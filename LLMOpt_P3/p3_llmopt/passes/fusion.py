"""P3 -- Kernel fusion pass.                       propose(ir) -> list[Candidate]

Idea (FlashAttention-style). Unfused, attention writes the score matrix
S = QK^T to GPU memory, softmax reads S and writes P, and the PV matmul reads
P back: about 4|S| bytes of traffic per layer, and the S and P buffers are
both alive while softmax runs (+2|S| peak memory). A fused kernel computes
softmax online (streaming, tile by tile) so S and P never leave on-chip
memory. The maths is unchanged, so accuracy risk is "none".

Three levels are considered. They are mutually exclusive, and the shared
Candidate has no "exclusive group" field, so the pass evaluates all three and
emits ONLY the best one (the others appear in its reason string):

  L1  Attention + Softmax                     (FlashAttention)
  L2  QKV projection -> Attention + Softmax   (prologue fusion)
  L3  L2 + output projection                  (epilogue fusion)

Why the best level depends on the workload:
  * Each level removes kernel launches (L1: -1, L2: -2, L3: -3 per layer).
  * L2/L3 force the projection GEMM onto the attention kernel's tiling, so the
    GEMM loses tensor-core efficiency (FUSED_GEMM_EFFICIENCY). That only costs
    time when the GEMM is compute-bound (prefill, very large batches).
  * L3 needs a cross-head reduction: the output projection sums over all
    heads but each attention block owns one head, so partial outputs are
    combined with FP32 atomic adds (traffic ~ heads x tokens x hidden).

Each level is costed with the same roofline P4 uses: the fused kernel's time
is max(sum of compute, remaining bytes / bandwidth), compared with the sum of
the unfused ops' roofline times.
"""
from __future__ import annotations

from dataclasses import dataclass

from p3 import perf
from p3.irview import P3View, make_candidate, view

# Assumptions (named so they can be defended / changed in one place)
FUSED_GEMM_EFFICIENCY = 0.75   # projection GEMM inside a fused kernel reaches 75% of a
                               # standalone cuBLAS GEMM's tensor-core efficiency
ATOMIC_RMW_FACTOR = 2          # atomic add = read + write of the FP32 partial
FP32_BYTES = 4

LEVELS = {
    1: ("attention", "softmax"),
    2: ("qkv", "attention", "softmax"),
    3: ("qkv", "attention", "softmax", "out_proj"),
}
LABELS = {
    1: "Attention+Softmax",
    2: "QKV->Attention+Softmax",
    3: "QKV->Attention+Softmax->OutProj",
}
_GEMM_KINDS = ("qkv", "out_proj")


@dataclass
class FusionLevel:
    level: int
    label: str
    launches_saved_per_layer: int
    traffic_saved_bytes: float      # per layer
    extra_bytes: float              # per layer (L3 atomics)
    unfused_ms: float               # per layer, roofline of the separate ops
    fused_ms: float                 # per layer, roofline of the fused kernel
    mem_saved_bytes: float          # peak activation memory, counted once
    d_latency_no_launch_ms: float   # whole model, per step/pass
    d_launches: int                 # whole model, per step/pass
    score_ms: float                 # incl. launch overhead; used to pick the level

    @property
    def emitted_latency_ms(self) -> float:
        return self.score_ms if perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY else self.d_latency_no_launch_ms


def evaluate_levels(ir) -> list[FusionLevel]:
    v = view(ir)
    return _evaluate(v)


def _evaluate(v: P3View) -> list[FusionLevel]:
    b = v.dtype_bytes
    s_bytes = v.batch * v.n_heads * v.q_len * v.kv_len * b        # score matrix S
    q_bytes = v.batch * v.q_len * v.n_heads * v.head_dim * b       # Q (and O) activations
    kv_bytes = 2 * v.batch * v.kv_len * v.n_kv_heads * v.head_dim * b
    tokens = v.batch * v.q_len
    per_layer = v.kinds_per_layer

    out: list[FusionLevel] = []
    for level, kinds in LEVELS.items():
        if any(per_layer.get(k, 0) == 0 for k in kinds):
            continue                                   # IR doesn't have these ops separately
        ops = [op for k in kinds for op in v.layer_ops(k)]
        launches_saved = len(ops) - 1

        saved = 4 * s_bytes                            # write S, read S, write P, read P
        mem_saved = 2 * s_bytes                        # S and P buffers
        extra = 0.0
        if "qkv" in kinds:
            saved += 2 * q_bytes                       # Q no longer round-trips memory
            mem_saved += q_bytes
        if "out_proj" in kinds:
            saved += 2 * q_bytes                       # attention output O stays on-chip
            mem_saved += q_bytes
            extra += ATOMIC_RMW_FACTOR * v.n_heads * tokens * v.hidden * FP32_BYTES

        unfused_ms = sum(v.roofline_ms(op) for op in ops)
        compute_ms = 0.0
        for op in ops:
            c, _ = v.op_ms(op)
            compute_ms += c / FUSED_GEMM_EFFICIENCY if op.kind in _GEMM_KINDS else c
        total_bytes = sum(op.bytes for op in ops)
        floor = kv_bytes + 2 * q_bytes                 # must still read Q, K, V and write O
        fused_bytes = max(total_bytes - saved, floor) + extra
        fused_ms = max(compute_ms, fused_bytes / v.bw_per_ms)

        d_no_launch = v.n_layers * (fused_ms - unfused_ms)
        d_launches = -launches_saved * v.n_layers
        out.append(FusionLevel(
            level, LABELS[level], launches_saved, total_bytes - (fused_bytes - extra), extra,
            unfused_ms, fused_ms, mem_saved, d_no_launch, d_launches,
            d_no_launch + d_launches * v.launch_ms))
    return out


def propose(ir) -> list:
    v = view(ir)
    levels = _evaluate(v)
    if not levels:
        return []
    best = min(levels, key=lambda lv: lv.score_ms)
    base = v.launches_per_step
    others = ", ".join(f"L{lv.level} {perf.fmt_ms(lv.score_ms - best.score_ms, signed=True)}"
                       for lv in levels if lv is not best)
    reason = (f"L{best.level} {best.label}: launches {base}->{base + best.d_launches} per {v.unit}, "
              f"{perf.fmt_bytes(best.traffic_saved_bytes)} less memory traffic per layer"
              + (f" (other levels vs this: {others})" if others else ""))
    return [make_candidate(
        name="fusion",
        d_latency_ms=best.emitted_latency_ms,
        d_mem_gb=-best.mem_saved_bytes / 1e9,
        d_launches=best.d_launches,
        accuracy_risk="none",          # online softmax is exact; only FP summation order changes
        reason=reason,
    )]
