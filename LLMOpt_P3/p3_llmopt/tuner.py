"""P3 -- Auto-tuner for the fused attention kernel.

Grid search over the launch configuration of a FlashAttention-style kernel:

    BLOCK_M     query rows per thread block           {16, 32, 64, 128}
    BLOCK_N     KV tokens per inner-loop tile         {16, 32, 64, 128}
    num_warps   warps per block (block size = 32*w)   {2, 4, 8}
    num_stages  software-pipeline depth (prefetch)    {1, 2, 3, 4}

That is 192 configurations. Every configuration is scored with an analytical
model of the target GPU (model-guided search, the same idea as TVM's
cost-model-based tuning). It runs on any machine, no GPU needed. P4's
benchmark is the project's measured data; numbers here are estimates.

What the model captures (each term is one physical effect):
  feasibility   shared memory for the Q tile + num_stages x (K,V) tiles must
                fit in one block's limit; register estimate must stay <= 255
                per thread (otherwise spills)
  occupancy     resident blocks/SM limited by shared memory, threads,
                registers and the hardware block limit
  waves         blocks spread over SMs (a 2nd, half-empty wave wastes time)
  tensor cores  an MMA instruction covers 16 rows per warp; fewer rows waste it
  softmax       online softmax: per score element ~5 FP32 ops, plus rescaling
                the output accumulator once per KV tile (small BLOCK_N = more
                rescales), on CUDA cores, which are much slower than tensor cores
  K/V reloads   prefill: every query tile re-reads K and V; the L2 cache
                absorbs part of it (bigger BLOCK_M = fewer reloads)
  latency       Little's law: enough bytes must be in flight per SM to cover
  hiding        DRAM latency (more stages / bigger tiles = more in flight)
  pipelining    num_stages = 1 cannot overlap loads with math: time = c + m;
                otherwise time = max(c, m)
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field

from p3 import perf
from p3.irview import P3View, make_candidate, view
from passes.schedule import choose_num_splits

BLOCK_M_CHOICES = (16, 32, 64, 128)
BLOCK_N_CHOICES = (16, 32, 64, 128)
WARPS_CHOICES = (2, 4, 8)
STAGES_CHOICES = (1, 2, 3, 4)

DRAM_LATENCY_NS = 500           # typical global-memory latency
LITTLE_SAFETY = 2.0             # kernels need ~2x the Little's-law minimum in flight
REG_BASE = 40                   # registers for indices, pointers, softmax stats
MAX_REGS = 255
REG_GRANULE = 8                 # register allocation granularity
MMA_ROWS_PER_WARP = 16          # m16n8k16 tensor-core instruction
MIN_WARPS_PER_SM_FOR_TC = 8     # 2 warps per tensor-core sub-partition to hide MMA latency
SOFTMAX_FLOPS_PER_SCORE = 5     # max, subtract, exp, sum, scale
CAUSAL_FRACTION = 0.5           # prefill: causal mask skips ~half the tiles
L2_MAX_HIT = 0.95
TIE_TOLERANCE = 0.01            # keep the default if it is within 1% of the best


@dataclass(frozen=True)
class TileConfig:
    block_m: int
    block_n: int
    num_warps: int
    num_stages: int

    @property
    def threads(self) -> int:
        return self.num_warps * 32

    def label(self) -> str:
        return f"BM={self.block_m} BN={self.block_n} w={self.num_warps} s={self.num_stages}"


DEFAULT = TileConfig(64, 64, 4, 2)      # what a hand-written kernel would typically hardcode


@dataclass
class Estimate:
    cfg: TileConfig
    feasible: bool
    why: str = ""                   # infeasibility reason
    why_kind: str = ""              # "smem" | "regs" | "occupancy"
    smem_kb: float = 0.0
    regs: int = 0
    blocks_per_sm: int = 0
    grid: int = 0
    l2_hit: float = 0.0
    compute_ms: float = 0.0         # per layer
    memory_ms: float = 0.0          # per layer
    time_ms: float = math.inf       # per layer

    @property
    def bound(self) -> str:
        return "compute" if self.compute_ms >= self.memory_ms else "memory"


@dataclass
class TuneResult:
    gpu: str
    phase: str
    n_layers: int
    splits: int
    total: int
    ranked: list[Estimate] = field(default_factory=list)       # feasible, best first
    infeasible: list[Estimate] = field(default_factory=list)
    baseline: Estimate | None = None
    baseline_note: str = ""
    chosen: Estimate | None = None   # best, or the baseline if it is within TIE_TOLERANCE

    @property
    def best(self) -> Estimate | None:
        return self.ranked[0] if self.ranked else None

    @property
    def keeps_baseline(self) -> bool:
        return self.chosen is not None and self.chosen is self.baseline


def search_space() -> list[TileConfig]:
    return [TileConfig(m, n, w, s) for m in BLOCK_M_CHOICES for n in BLOCK_N_CHOICES
            for w in WARPS_CHOICES for s in STAGES_CHOICES]


def estimate(cfg: TileConfig, v: P3View, splits: int = 1) -> Estimate:
    hw, d, b = v.hw, v.head_dim, v.dtype_bytes
    BM, BN = cfg.block_m, cfg.block_n

    # ---- feasibility and occupancy
    smem = (BM * d + cfg.num_stages * 2 * BN * d) * b
    if smem > hw.smem_per_block_kb * 1024:
        return Estimate(cfg, False, f"needs {smem / 1024:.0f} KB shared mem > {hw.smem_per_block_kb:g} KB",
                        "smem", smem_kb=smem / 1024)
    regs = math.ceil((BM * d + BM * BN) / cfg.threads) + REG_BASE   # O accumulator + score tile
    if regs > MAX_REGS:
        return Estimate(cfg, False, f"~{regs} registers/thread > {MAX_REGS} (spills)", "regs",
                        smem_kb=smem / 1024, regs=regs)
    regs_alloc = math.ceil(regs / REG_GRANULE) * REG_GRANULE
    bps = min(int(hw.smem_per_sm_kb * 1024 // smem), hw.max_threads_per_sm // cfg.threads,
              hw.regs_per_sm // (regs_alloc * cfg.threads), hw.max_blocks_per_sm)
    if bps < 1:
        return Estimate(cfg, False, "no block fits on an SM", "occupancy", smem_kb=smem / 1024, regs=regs)

    # ---- work decomposition
    B, H, Hkv = v.batch, v.n_heads, v.n_kv_heads
    group = max(1, H // Hkv)                      # query heads sharing one KV head (GQA)
    if v.is_decode:
        kv = v.kv_len
        m_tiles = math.ceil(group * v.q_len / BM)
        grid = B * Hkv * splits * m_tiles
        score_elems = B * Hkv * m_tiles * BM * kv          # padded rows are computed too
        kv_tiles = math.ceil(math.ceil(kv / splits) / BN)
        rescale = grid * kv_tiles * BM * d
        kv_head = 2 * kv * d * b
        kv_dram = B * Hkv * kv_head * m_tiles
        l2_hit = 0.0
    else:
        S = v.q_len
        m_tiles = math.ceil(S / BM)
        grid = B * H * m_tiles
        score_elems = B * H * m_tiles * BM * S * CAUSAL_FRACTION
        kv_tiles = math.ceil(S * CAUSAL_FRACTION / BN)
        rescale = grid * kv_tiles * BM * d
        kv_head = 2 * S * d * b
        kv_once = B * Hkv * kv_head
        kv_reads = B * H * m_tiles * kv_head * CAUSAL_FRACTION
        resident = min(grid, hw.sms * bps)
        kv_heads_in_flight = max(1.0, resident / m_tiles / group)
        l2_hit = min(L2_MAX_HIT, hw.l2_mb * 1e6 / (kv_heads_in_flight * kv_head))
        kv_dram = kv_once + max(0.0, kv_reads - kv_once) * (1 - l2_hit)
    qo_bytes = 2 * B * H * v.q_len * d * b
    tc_flops = 4 * score_elems * d                # QK^T and PV

    # ---- utilisation
    sm_util = perf.wave_util(grid, hw.sms)
    resident_per_sm = min(bps, math.ceil(grid / hw.sms))
    tc_eff = min(1.0, (BM / cfg.num_warps) / MMA_ROWS_PER_WARP)
    tc_occ = min(1.0, resident_per_sm * cfg.num_warps / MIN_WARPS_PER_SM_FOR_TC)
    compute = (tc_flops / (v.tc_per_ms * tc_eff * tc_occ)
               + (SOFTMAX_FLOPS_PER_SCORE * score_elems + rescale) / (hw.fp32_tflops * 1e9)) / sm_util

    in_flight = resident_per_sm * cfg.num_stages * 2 * BN * d * b
    needed = (hw.mem_bw_gbps * 1e9 / hw.sms) * DRAM_LATENCY_NS * 1e-9 * LITTLE_SAFETY
    hide = min(1.0, in_flight / needed)
    memory = (kv_dram + qo_bytes) / (v.bw_per_ms * sm_util * hide)

    t = max(compute, memory) if cfg.num_stages >= 2 else compute + memory
    return Estimate(cfg, True, smem_kb=smem / 1024, regs=regs, blocks_per_sm=bps, grid=grid,
                    l2_hit=l2_hit, compute_ms=compute, memory_ms=memory, time_ms=t)


def _baseline(v: P3View, splits: int) -> tuple[Estimate, str]:
    """The hand-written default; if it doesn't fit this GPU, shrink it the way
    a developer would (halve BLOCK_N, then stages, then BLOCK_M) until it does."""
    est = estimate(DEFAULT, v, splits)
    if est.feasible:
        return est, f"default {DEFAULT.label()}"
    cfg = DEFAULT
    while not est.feasible:
        if cfg.block_n > 16:
            cfg = TileConfig(cfg.block_m, cfg.block_n // 2, cfg.num_warps, cfg.num_stages)
        elif cfg.num_stages > 1:
            cfg = TileConfig(cfg.block_m, cfg.block_n, cfg.num_warps, cfg.num_stages - 1)
        elif cfg.block_m > 16:
            cfg = TileConfig(cfg.block_m // 2, cfg.block_n, cfg.num_warps, cfg.num_stages)
        else:
            break
        est = estimate(cfg, v, splits)
    first = estimate(DEFAULT, v, splits)
    return est, (f"default {DEFAULT.label()} does not launch on {v.hw.name} ({first.why}); "
                 f"hand-shrunk fallback {cfg.label()}")


def tune(ir) -> TuneResult:
    v = view(ir)
    splits = choose_num_splits(v)
    space = search_space()
    ests = [estimate(c, v, splits) for c in space]
    ranked = sorted((e for e in ests if e.feasible), key=lambda e: (e.time_ms, e.smem_kb))
    baseline, note = _baseline(v, splits)
    chosen = ranked[0] if ranked else None
    if chosen and baseline.feasible and baseline.time_ms <= chosen.time_ms * (1 + TIE_TOLERANCE):
        chosen = baseline
    return TuneResult(v.hw.name, v.mode, v.n_layers, splits, len(space), ranked,
                      [e for e in ests if not e.feasible], baseline, note, chosen)


def propose(ir) -> list:
    r = tune(ir)
    if r.chosen is None or r.baseline is None or not r.baseline.feasible:
        return []
    n_smem = sum(e.why_kind == "smem" for e in r.infeasible)
    n_regs = sum(e.why_kind == "regs" for e in r.infeasible)
    unit = "decode step" if r.phase == "decode" else "prefill pass"
    tail = (f"{len(r.ranked)}/{r.total} configs feasible "
            f"({n_smem} rejected for shared memory, {n_regs} for registers)")
    if r.keeps_baseline:
        reason = (f"keep {r.baseline_note}: within {TIE_TOLERANCE:.0%} of the best "
                  f"({r.baseline.bound}-bound, tile choice barely changes the time); {tail}")
    else:
        reason = (f"best {r.chosen.cfg.label()} vs {r.baseline_note}: attention "
                  f"{perf.fmt_ms(r.baseline.time_ms * r.n_layers)} -> "
                  f"{perf.fmt_ms(r.chosen.time_ms * r.n_layers)} per {unit}; {tail}")
    d = (r.chosen.time_ms - r.baseline.time_ms) * r.n_layers
    return [make_candidate("autotune", d, 0.0, 0, "none", reason)]


def format_table(r: TuneResult, top_k: int = 10) -> str:
    base_t = r.baseline.time_ms if r.baseline and r.baseline.feasible else None
    rows = []
    for i, e in enumerate(r.ranked[:top_k], 1):
        vs = f"{base_t / e.time_ms:.2f}x" if base_t else "-"
        rows.append([f"{i}{'*' if e is r.chosen else ''}", e.cfg.block_m, e.cfg.block_n, e.cfg.num_warps, e.cfg.num_stages,
                     f"{e.smem_kb:.0f}", e.regs, e.blocks_per_sm, e.bound,
                     f"{e.time_ms * 1000:.1f}", vs])
    out = [perf.table(["#", "BM", "BN", "warps", "stages", "smem KB", "regs", "blk/SM", "bound",
                       "us/layer", "vs base"], rows)]
    if r.baseline and r.baseline.feasible:
        out.append(f"  baseline: {r.baseline_note} -> {r.baseline.time_ms * 1000:.1f} us/layer")
    if r.chosen is not None:
        pick = "baseline kept (within 1% of best)" if r.keeps_baseline else r.chosen.cfg.label()
        out.append(f"  chosen  : {pick}")
    kinds = {}
    for e in r.infeasible:
        kinds.setdefault(e.why_kind, e)
    for k, e in kinds.items():
        n = sum(x.why_kind == k for x in r.infeasible)
        out.append(f"  rejected {n:3d} configs [{k}], e.g. {e.cfg.label()}: {e.why}")
    return "\n".join(out)


def print_table(r: TuneResult, top_k: int = 10) -> None:
    print(format_table(r, top_k))


if __name__ == "__main__":                 # quick look: python tuner.py --scenario 3
    from p3.mock import scenario_ir
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=int, default=1)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    title, ir = scenario_ir(a.scenario)
    print(title)
    print_table(tune(ir), a.top)
