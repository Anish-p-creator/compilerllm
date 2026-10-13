"""P2 pass 1 - KV-cache optimization.

Candidates
  paged_kv : allocate KV in fixed blocks instead of reserving max_context per request.
             Saves the unused-context memory, costs ~3% attention latency.
  int8_kv  : store K/V as INT8 (+ one fp16 scale per token/head vector). Halves KV memory,
             faster on memory-bound decode, low accuracy risk (MEASURED below).

Both are only proposed under memory pressure: reserved KV > 60% of GPU memory."""
from __future__ import annotations

from typing import Optional

import numpy as np

from cost_model import (attention_intensity, decode_attention_memory_bound, int8_elem_bytes,
                        int8_latency_delta_pct, int8_ratio, kv_pressure, kv_reserved_bytes,
                        paged_tokens, ridge_point)
from ir import GIB, Candidate, Context, PassResult, Rejection


# ------------------------------------------------------- measured INT8 accuracy proxy
def _quantize_per_token(x: np.ndarray) -> np.ndarray:
    """Symmetric INT8, one fp16 scale per row (= per token, per head). Returns dequantized x."""
    scale = np.abs(x).max(axis=1, keepdims=True) / 127.0
    scale = scale.astype(np.float16).astype(np.float32)
    scale[scale == 0] = 1.0
    q = np.clip(np.round(x / scale), -127, 127)
    return (q * scale).astype(np.float32)


def _attention(q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
    s = q @ k.T / np.sqrt(k.shape[1])
    s -= s.max(axis=1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(axis=1, keepdims=True)
    return p @ v


def simulate_int8_error(head_dim: int, seq_len: int = 1024, n_queries: int = 64, seed: int = 0,
                        K: Optional[np.ndarray] = None, V: Optional[np.ndarray] = None,
                        Q: Optional[np.ndarray] = None) -> dict:
    """Run real attention with FP32 K/V vs INT8-quantized K/V and compare the outputs.

    Without K/V/Q it uses synthetic heads: per-channel scale spread, 2 K outlier channels
    (hard case for per-token quantization) and V with a shared component (as in real models;
    zero-mean random V makes the attention output cancel and inflates *relative* error).
    Headline = query scale 2 (moderately peaked attention); `worst_rel_error` is the max over
    query scales 1/2/4. Pass real K/V/Q tensors to calibrate on a real model."""
    synthetic = K is None
    if synthetic:
        rng = np.random.default_rng(seed)
        ch = np.exp(rng.normal(0, 0.4, head_dim))
        K = rng.normal(size=(seq_len, head_dim)) * ch
        K[:, rng.choice(head_dim, size=2, replace=False)] *= 6.0   # outlier channels
        V = rng.normal(size=(seq_len, head_dim)) * 0.5 + rng.normal(size=(1, head_dim))
        Q0 = rng.normal(size=(n_queries, head_dim))
        sweep = {s: Q0 * s for s in (1.0, 2.0, 4.0)}
        Q = sweep[2.0]
    else:
        sweep = {}
    K, V, Q = (np.asarray(a, dtype=np.float32) for a in (K, V, Q))
    Kq, Vq = _quantize_per_token(K), _quantize_per_token(V)

    def rel(q):
        q = np.asarray(q, dtype=np.float32)
        r, o = _attention(q, K, V), _attention(q, Kq, Vq)
        return r, o, float(np.linalg.norm(o - r) / np.linalg.norm(r))

    ref, out, headline = rel(Q)
    worst = max([headline] + [rel(q)[2] for q in sweep.values()])
    cos = (out * ref).sum(axis=1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1))
    return dict(rel_error=headline, worst_rel_error=worst, max_abs_error=float(np.abs(out - ref).max()),
                mean_cosine=float(cos.mean()), seq_len=int(K.shape[0]),
                data="synthetic heads" if synthetic else "user-supplied tensors")


def risk_from_error(rel_error: float) -> str:
    return "low" if rel_error < 0.01 else "medium" if rel_error < 0.03 else "high"


# ------------------------------------------------------------------------- the pass
class KVCachePass:
    name = "kv_cache"

    def run(self, ctx: Context) -> PassResult:
        res = PassResult(self.name)
        m, w, o = ctx.model, ctx.workload, ctx.opts
        reserved = kv_reserved_bytes(ctx)
        pressure = kv_pressure(ctx)
        res.info.update(kv_bytes_per_token=m.kv_bytes_per_token(), reserved_kv_gib=reserved / GIB,
                        memory_pressure=pressure, pressure_threshold=o.pressure_threshold,
                        high_pressure=pressure > o.pressure_threshold)

        if pressure <= o.pressure_threshold:
            why = (f"low memory pressure: reserved KV = {reserved / GIB:.1f} GiB = {pressure * 100:.1f}% of GPU "
                   f"memory (proposed only above {o.pressure_threshold * 100:.0f}%)")
            res.rejections += [Rejection("paged_kv", why), Rejection("int8_kv", why)]
            return res

        # ---- paged_kv
        reserved_tokens = w.batch_size * w.max_context
        used_tokens = paged_tokens(ctx)
        saved_tokens = reserved_tokens - used_tokens
        if saved_tokens <= 0:
            res.rejections.append(Rejection("paged_kv", "no unused context: prompt+generation already fills max_context"))
        else:
            bpt = m.kv_bytes_per_token()
            res.candidates.append(Candidate(
                name="paged_kv", pass_name=self.name,
                description=f"paged KV cache, {o.block_size}-token blocks",
                memory_saved_bytes=saved_tokens * bpt,
                latency_delta_pct=o.paged_overhead_pct,
                accuracy_risk="none",
                rationale=(f"contiguous allocation reserves {w.max_context} tokens/request but only "
                           f"{w.prompt_length + w.generation_length} are ever used "
                           f"({used_tokens / reserved_tokens * 100:.0f}% utilization)"),
                details=dict(block_size=o.block_size, reserved_gib=reserved / GIB,
                             paged_gib=used_tokens * bpt / GIB, utilization=used_tokens / reserved_tokens,
                             saved_gib=saved_tokens * bpt / GIB)))

        # ---- int8_kv
        if m.dtype_bytes <= 1:
            res.rejections.append(Rejection("int8_kv", f"KV is already {m.precision.upper()} (1 byte/element)"))
        else:
            ratio = int8_ratio(m)
            acc = simulate_int8_error(m.head_dim, seed=w.seed)
            risk = risk_from_error(acc["rel_error"])
            if risk == "high":
                res.rejections.append(Rejection(
                    "int8_kv", f"accuracy risk too high: attention-output error {acc['rel_error'] * 100:.2f}%"))
            else:
                bpt = m.kv_bytes_per_token()
                mb = decode_attention_memory_bound(ctx)
                res.candidates.append(Candidate(
                    name="int8_kv", pass_name=self.name,
                    description=f"INT8 KV cache (per-token symmetric, fp16 scales)",
                    memory_saved_bytes=reserved * (1 - ratio),
                    latency_delta_pct=int8_latency_delta_pct(ctx),
                    accuracy_risk=risk,
                    rationale=("decode attention is memory-bound "
                               f"({attention_intensity(m):.1f} FLOP/B << ridge {ridge_point(ctx):.0f} FLOP/B), "
                               "so reading half the bytes cuts attention time"
                               if mb else "attention is compute-bound here: saves memory only"),
                    details=dict(bytes_per_element=int8_elem_bytes(m), size_ratio=ratio,
                                 kv_gib_baseline=reserved / GIB, kv_gib_int8=reserved * ratio / GIB,
                                 saved_gib=reserved * (1 - ratio) / GIB,
                                 saved_gib_if_paged_too=(used_tokens * bpt * (1 - ratio) / GIB),
                                 memory_bound=mb, accuracy=acc)))
        return res
