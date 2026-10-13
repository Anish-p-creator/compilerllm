"""P3 -- Decode scheduling pass.                  propose(ir) -> list[Candidate]

Two scheduling transforms from PersistentKV / Flash-Decoding. Both are decode
only (prefill already has plenty of parallelism and few launches per token),
and neither changes the maths, so accuracy risk is "none".

split_kv
    A decode attention kernel normally gives one thread block to each
    (sequence, KV head) pair: batch x n_kv_heads blocks. GQA-aware kernels
    share a KV head between its query heads, so KV heads are the real unit
    of parallelism (Mistral-7B: 8 KV heads, not 32). With batch 1 that leaves
    most of the GPU idle. Splitting each sequence's KV into chunks multiplies
    the block count; a small extra kernel then merges the partial results
    exactly (log-sum-exp rescaling).
    Rule from the team doc: split when blocks < 2 x SMs.
    Costs: +1 launch per layer and a tiny FP32 workspace.
    Flash-Decoding (Dao et al., 2023) reports up to 8x faster decode attention
    for very long sequences with this transform.

persistent_kernel
    Launch one kernel per decode step that loops over all layers and ops,
    instead of one launch per op (PersistentKV cuts 16 launches to 2). It
    saves (launches - 1) x launch overhead. It costs a dependency wait between
    consecutive ops, plus some efficiency on every op, because one generic
    kernel can't use each op's best grid/tile choice. It pays off only when
    launch overhead is a large share of the step: small batch, short context.

When split_kv cannot help (enough blocks already), the pass still emits it
with its honest, positive (worse) delta, so P4's report can show WHY it was
rejected instead of the candidate silently disappearing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from p3 import perf
from p3.irview import P3View, make_candidate, view

# ---- split_kv
SPLIT_TARGET_BLOCKS_PER_SM = 2     # team-doc rule: split when blocks < 2 x SMs
MIN_CHUNK_TOKENS = 256             # don't create KV chunks smaller than this
MAX_SPLITS = 64
FP32_BYTES = 4

# ---- persistent kernel
PERSIST_SYNC_US = 1.0              # wait between dependent ops inside the kernel
                                   # (flag/counter based), vs ~4 us for a new launch
PERSIST_BW_LOSS = 0.02             # 2% lower bandwidth on memory-bound ops
PERSIST_GEMM_LOSS = 0.15           # 15% lower tensor-core efficiency on compute-bound ops


# ------------------------------------------------------------------ split_kv
def _units(v: P3View) -> int:
    return v.batch * v.n_kv_heads


def _target(v: P3View) -> int:
    return SPLIT_TARGET_BLOCKS_PER_SM * v.hw.sms


def _parallel_eff(blocks: int, target: int) -> float:
    """Fraction of peak bandwidth reached with `blocks` thread blocks: the GPU
    needs ~target blocks in flight to saturate memory; beyond that, more
    blocks don't help (wave/tail effects are left to the auto-tuner)."""
    return min(1.0, blocks / target)


def choose_num_splits(ir_or_view) -> int:
    """KV chunks per (sequence, KV head) for decode attention (1 = no split).
    Shared with the auto-tuner so both reason about the same grid."""
    v = ir_or_view if isinstance(ir_or_view, P3View) else view(ir_or_view)
    if not v.is_decode:
        return 1
    units, target = _units(v), _target(v)
    if units >= target:
        return 1
    return max(1, min(math.ceil(target / units),
                      math.ceil(v.kv_len / MIN_CHUNK_TOKENS),
                      MAX_SPLITS))


@dataclass
class SplitPlan:
    units: int
    target: int
    splits: int
    forced: bool                    # True = no split needed; costed at 2 to show the loss
    attn_bytes_per_layer: float
    t_unsplit_ms: float             # per layer
    t_split_ms: float               # per layer
    t_reduce_ms: float              # per layer
    workspace_bytes: float
    d_latency_no_launch_ms: float   # whole model
    d_launches: int
    score_ms: float

    @property
    def emitted_latency_ms(self) -> float:
        return self.score_ms if perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY else self.d_latency_no_launch_ms


def plan_split_kv(ir) -> SplitPlan | None:
    v = view(ir)
    if not v.is_decode:
        return None
    b = v.dtype_bytes
    units, target = _units(v), _target(v)
    splits = choose_num_splits(v)
    forced = splits == 1
    if forced:
        splits = 2

    attn_bytes = 2 * v.batch * v.kv_len * v.n_kv_heads * v.head_dim * b       # K and V
    t_unsplit = attn_bytes / (v.bw_per_ms * _parallel_eff(units, target))
    t_split = attn_bytes / (v.bw_per_ms * _parallel_eff(units * splits, target))
    # partial O (FP32) + running max + running sum per (seq, query head, split): write + read
    workspace = v.batch * v.n_heads * splits * (v.head_dim + 2) * FP32_BYTES
    t_reduce = 2 * workspace / v.bw_per_ms

    d_no_launch = v.n_layers * (t_split + t_reduce - t_unsplit)
    d_launches = v.n_layers                                   # one reduction kernel per layer
    return SplitPlan(units, target, splits, forced, attn_bytes, t_unsplit, t_split, t_reduce,
                     workspace, d_no_launch, d_launches, d_no_launch + d_launches * v.launch_ms)


# --------------------------------------------------------- persistent kernel
@dataclass
class PersistentPlan:
    launches: int
    work_ms: float                  # roofline time of the step, no launches
    launch_overhead_ms: float
    launch_share: float             # launch overhead / (work + launches)
    sync_ms: float
    penalty_ms: float
    d_latency_no_launch_ms: float
    d_launches: int
    score_ms: float

    @property
    def emitted_latency_ms(self) -> float:
        return self.score_ms if perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY else self.d_latency_no_launch_ms


def plan_persistent(ir) -> PersistentPlan | None:
    v = view(ir)
    if not v.is_decode:
        return None
    launches = v.launches_per_step
    work = v.step_ms
    overhead = launches * v.launch_ms
    penalty = 0.0
    for op in v.ops:
        c, m = v.op_ms(op)
        penalty += max(c / (1 - PERSIST_GEMM_LOSS), m / (1 - PERSIST_BW_LOSS)) - max(c, m)
    sync = (launches - 1) * PERSIST_SYNC_US / 1000.0
    d_no_launch = sync + penalty
    d_launches = 1 - launches       # relative to the unoptimised IR; P4 should clamp total >= 1
    return PersistentPlan(launches, work, overhead, overhead / (work + overhead), sync, penalty,
                          d_no_launch, d_launches, d_no_launch + d_launches * v.launch_ms)


# -------------------------------------------------------------------- propose
def propose(ir) -> list:
    v = view(ir)
    out = []

    sp = plan_split_kv(ir)
    if sp is not None:
        if sp.forced:
            reason = (f"not needed: {sp.units} blocks (batch x KV heads) already >= {sp.target} "
                      f"(2 x {v.hw.sms} SMs); splitting only adds a reduction pass")
        else:
            reason = (f"{sp.units} blocks (batch x KV heads) < {sp.target} (2 x {v.hw.sms} SMs): "
                      f"split KV into {sp.splits} chunks -> {sp.units * sp.splits} blocks; "
                      f"attention {perf.fmt_ms(sp.t_unsplit_ms)} -> "
                      f"{perf.fmt_ms(sp.t_split_ms + sp.t_reduce_ms)} per layer, +1 reduction launch/layer")
        out.append(make_candidate("split_kv", sp.emitted_latency_ms, sp.workspace_bytes / 1e9,
                                  sp.d_launches, "none", reason))

    pp = plan_persistent(ir)
    if pp is not None:
        saves = -pp.d_launches * v.launch_ms
        reason = (f"launch overhead is {pp.launch_share:.1%} of the step "
                  f"({pp.launches} launches x {v.hw.launch_us:g} us vs {perf.fmt_ms(pp.work_ms)} of work); "
                  f"1 launch saves {perf.fmt_ms(saves)}, costs {perf.fmt_ms(pp.sync_ms + pp.penalty_ms)} "
                  f"(op-to-op syncs + generic-kernel efficiency loss)")
        out.append(make_candidate("persistent_kernel", pp.emitted_latency_ms, 0.0,
                                  pp.d_launches, "none", reason))
    return out
