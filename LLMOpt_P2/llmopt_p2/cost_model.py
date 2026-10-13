"""Analytical (roofline-style) cost model used by the P2 passes and the planner.

Everything here is a closed-form estimate from model shape + hardware specs, NOT a GPU
measurement. Decode is treated as memory-bound (time = bytes moved / achievable HBM
bandwidth); prefill is treated as compute-bound (time = FLOPs / achievable FLOPs)."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ir import GIB, Context, ModelSpec


# ----------------------------------------------------------------------- KV sizing
def kv_reserved_bytes(ctx: Context) -> float:
    """Baseline: contiguous KV buffer reserved for max_context per request."""
    return ctx.workload.batch_size * ctx.workload.max_context * ctx.model.kv_bytes_per_token()


def kv_pressure(ctx: Context) -> float:
    return kv_reserved_bytes(ctx) / ctx.hw.memory_bytes


def paged_tokens(ctx: Context) -> int:
    """Token slots a paged allocator needs: only blocks actually touched (prompt + generation)."""
    w, bs = ctx.workload, ctx.opts.block_size
    return w.batch_size * math.ceil((w.prompt_length + w.generation_length) / bs) * bs


def int8_elem_bytes(model: ModelSpec) -> float:
    """Bytes per KV element in INT8 incl. one fp16 scale per (token, head, K|V) vector."""
    return 1.0 + 2.0 / model.head_dim


def int8_ratio(model: ModelSpec) -> float:
    return int8_elem_bytes(model) / model.dtype_bytes


def prefill_flops(model: ModelSpec, n: int) -> float:
    """FLOPs to prefill n tokens: dense matmuls (2*params/token) + causal attention."""
    return 2.0 * model.params_b * 1e9 * n + 2.0 * model.layers * model.hidden * n * (n + 1)


# --------------------------------------------------------- memory-bound analysis
def attention_intensity(model: ModelSpec) -> float:
    """Decode-attention FLOPs per byte of KV read: 2 FLOPs/element x GQA group size."""
    return 2.0 * (model.heads / model.kv_heads) / model.dtype_bytes


def ridge_point(ctx: Context) -> float:
    return ctx.hw.fp16_tflops * 1e12 / (ctx.hw.bandwidth_gbps * 1e9)


def decode_attention_memory_bound(ctx: Context) -> bool:
    return attention_intensity(ctx.model) < ridge_point(ctx)


def int8_latency_delta_pct(ctx: Context) -> float:
    """Change in attention latency from INT8 KV for the workload's mode."""
    ov = ctx.opts.int8_dequant_overhead_pct / 100.0
    dec = (int8_ratio(ctx.model) * (1 + ov) - 1) * 100 if decode_attention_memory_bound(ctx) else ov * 100
    pre = ov * 100  # prefill attention is compute-bound: no byte savings, only dequant cost
    f = {"decode": 1.0, "prefill": 0.0, "mixed": ctx.opts.decode_fraction}[ctx.workload.mode]
    return f * dec + (1 - f) * pre


# --------------------------------------------------------------------- full metrics
@dataclass
class Metrics:
    kv_gib: float
    weights_gib: float
    total_gib: float
    fits: bool
    max_batch: int
    ttft_ms: float
    tpot_ms: float
    throughput_tok_s: float
    kv_read_gib_per_step: float
    hbm_traffic_gib_per_step: float
    prefill_tokens: int


def evaluate(ctx: Context, paged: bool = False, int8: bool = False,
             prefix: Optional[dict] = None) -> Metrics:
    """Estimate memory + latency for a combination of P2 optimizations.

    prefix = {'token_reuse_ratio': r, 'flops_saved_fraction': f} from the prefix pass."""
    m, w, hw, o = ctx.model, ctx.workload, ctx.hw, ctx.opts
    B, P, G, C = w.batch_size, w.prompt_length, w.generation_length, w.max_context
    bpt = m.kv_bytes_per_token()
    bpt_eff = m.kv_bytes_per_token(int8_elem_bytes(m)) if int8 else bpt

    # ---- memory
    alloc_tokens = paged_tokens(ctx) if paged else B * C
    shared_tokens = prefix["token_reuse_ratio"] * B * P if prefix else 0.0
    kv_bytes = max(alloc_tokens - shared_tokens, 0.0) * bpt_eff
    usable = hw.memory_bytes * o.usable_memory_fraction
    weights = m.weight_bytes
    per_req = kv_bytes / B
    avail = usable - weights
    max_batch = int(avail // per_req) if avail > 0 and per_req > 0 else 0

    # ---- decode step (memory-bound): weights + KV attention read
    eff_bw = hw.bandwidth_gbps * 1e9 * o.mem_bw_efficiency
    avg_ctx = P + G / 2.0
    kv_read = B * avg_ctx * bpt_eff
    t_attn = B * avg_ctx * bpt / eff_bw
    if int8:
        if decode_attention_memory_bound(ctx):
            t_attn *= int8_ratio(m)
        t_attn *= 1 + o.int8_dequant_overhead_pct / 100.0
    if paged:
        t_attn *= 1 + o.paged_overhead_pct / 100.0
    t_w = weights / eff_bw
    tpot = t_w + t_attn

    # ---- prefill (compute-bound)
    f_saved = prefix["flops_saved_fraction"] if prefix else 0.0
    flops = B * prefill_flops(m, P) * (1 - f_saved)
    ttft = max(flops / (hw.fp16_tflops * 1e12 * o.prefill_mfu), weights / eff_bw)

    return Metrics(
        kv_gib=kv_bytes / GIB, weights_gib=weights / GIB, total_gib=(kv_bytes + weights) / GIB,
        fits=(kv_bytes + weights) <= usable, max_batch=max_batch,
        ttft_ms=ttft * 1e3, tpot_ms=tpot * 1e3, throughput_tok_s=B / tpot,
        kv_read_gib_per_step=kv_read / GIB, hbm_traffic_gib_per_step=(kv_read + weights) / GIB,
        prefill_tokens=int(round(B * P * (1 - (prefix["token_reuse_ratio"] if prefix else 0.0)))),
    )
