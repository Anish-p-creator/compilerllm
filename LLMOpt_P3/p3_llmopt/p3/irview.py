"""IR adapter -- the ONLY place where P3 touches P1's IR.

P1's exact field names are not final yet, so every read goes through view(),
which accepts dataclasses or dicts and tries the common spellings
(n_kv_heads / num_kv_heads / num_key_value_heads, ...). Strings such as
"1935 GB/s" or "80GB" are parsed. Hardware fields P1 does not provide
(shared memory, registers, ...) come from p3/gpu_specs.py.

If integration day reveals a name this file doesn't know, add it to the
relevant list below -- nothing else in P3 needs to change.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from p3.gpu_specs import lookup

try:                                    # P1's shared class, once pushed
    from common import Candidate as _Candidate
    CANDIDATE_SOURCE = "common.py (P1)"
except ImportError:                     # until then, the stand-in
    from p3.mock import Candidate as _Candidate
    CANDIDATE_SOURCE = "p3/mock.py (stand-in until P1 pushes)"

_MISSING = object()


def _get(obj, names, default=None):
    """First non-None attribute or dict key from `names`."""
    if obj is None:
        return default
    for n in names:
        if isinstance(obj, dict):
            v = obj.get(n)
        else:
            v = getattr(obj, n, None)
        if callable(v):
            try:
                v = v()
            except TypeError:
                continue
        if v is not None:
            return v
    return default


def _num(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(x))
    return float(m.group()) if m else None


# ----------------------------------------------------------------- hardware
@dataclass
class HW:
    name: str
    sms: int
    mem_gb: float
    mem_bw_gbps: float
    fp16_tflops: float
    fp32_tflops: float
    smem_per_sm_kb: float
    smem_per_block_kb: float
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    l2_mb: float
    launch_us: float


# field -> (accepted names, scale used if P1 stored raw SI units)
_HW_FIELDS = {
    "sms": (["sms", "num_sms", "n_sms", "sm_count", "streaming_multiprocessors"], None),
    "mem_gb": (["mem_gb", "gpu_memory_gb", "gpu_memory", "memory_gb", "gpu_mem_gb", "vram_gb"], 1e9),
    "mem_bw_gbps": (["mem_bw_gbps", "memory_bandwidth_gbps", "memory_bandwidth", "bandwidth_gbps",
                     "bandwidth", "mem_bw", "bw_gbps"], 1e9),
    "fp16_tflops": (["fp16_tflops", "peak_fp16_tflops", "tensor_tflops", "peak_tflops", "tflops"], 1e12),
    "fp32_tflops": (["fp32_tflops", "peak_fp32_tflops"], 1e12),
    "smem_per_sm_kb": (["smem_per_sm_kb", "shared_mem_per_sm_kb"], None),
    "smem_per_block_kb": (["smem_per_block_kb", "shared_mem_per_block_kb"], None),
    "max_threads_per_sm": (["max_threads_per_sm"], None),
    "max_blocks_per_sm": (["max_blocks_per_sm"], None),
    "regs_per_sm": (["regs_per_sm", "registers_per_sm"], None),
    "l2_mb": (["l2_mb", "l2_cache_mb"], None),
    "launch_us": (["launch_us", "launch_overhead_us", "kernel_launch_us"], None),
}
_INT_FIELDS = {"sms", "max_threads_per_sm", "max_blocks_per_sm", "regs_per_sm"}


def _hw(ir) -> HW:
    obj = _get(ir, ["hw", "hardware", "target", "gpu", "device"])
    if obj is None:
        raise ValueError("P3: IR has no hardware object (looked for ir.hw / ir.hardware / ir.target)")
    if isinstance(obj, str):
        name, obj = obj, {}
    else:
        name = str(_get(obj, ["name", "device", "gpu", "gpu_name", "model"], ""))
    base = lookup(name)
    vals, missing = {}, []
    for f, (names, scale) in _HW_FIELDS.items():
        v = _num(_get(obj, names))
        if v is not None and scale and v > 1e5:      # stored in bytes/s or FLOP/s
            v /= scale
        if v is None and base is not None:
            v = getattr(base, f)
        if v is None and f == "launch_us":
            v = 4.0
        if v is None:
            missing.append(f)
        else:
            vals[f] = int(v) if f in _INT_FIELDS else float(v)
    if missing:
        raise ValueError(f"P3 needs hardware fields {missing} for GPU '{name}'. "
                         f"Add them to P1's preset or to p3/gpu_specs.py")
    return HW(name=name or (base.name if base else "unknown GPU"), **vals)


# ---------------------------------------------------------------------- ops
@dataclass
class OpInfo:
    kind: str
    layer: int | None
    flops: float
    bytes: float
    launches: int = 1


def _kind(op) -> str:
    s = str(_get(op, ["kind", "op_type", "type", "op", "name"], "")).lower()
    s = re.sub(r"[^a-z]", "", s)
    if "qkv" in s:
        return "qkv"
    if "softmax" in s:
        return "softmax"
    if "outproj" in s or "oproj" in s or s in ("out", "output", "dense"):
        return "out_proj"
    if "attn" in s or "attention" in s:
        return "attention"
    if "mlp" in s or "ffn" in s or "feedforward" in s:
        return "mlp"
    return s or "other"


# --------------------------------------------------------------------- view
@dataclass
class P3View:
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    batch: int
    mode: str
    q_len: int
    kv_len: int
    dtype_bytes: int
    hw: HW
    ops: list[OpInfo] = field(default_factory=list)

    # -- derived ------------------------------------------------------------
    @property
    def is_decode(self) -> bool:
        return self.mode == "decode"

    @property
    def unit(self) -> str:
        return "decode step" if self.is_decode else "prefill pass"

    @property
    def launch_ms(self) -> float:
        return self.hw.launch_us / 1000.0

    @property
    def bw_per_ms(self) -> float:            # bytes per millisecond
        return self.hw.mem_bw_gbps * 1e6

    @property
    def tc_per_ms(self) -> float:            # tensor-core FLOPs per millisecond
        return self.hw.fp16_tflops * 1e9

    @property
    def launches_per_step(self) -> int:
        return sum(op.launches for op in self.ops)

    def _first_layer_ops(self) -> list[OpInfo]:
        layers = [op.layer for op in self.ops if op.layer is not None]
        if layers:
            first = min(layers)
            return [op for op in self.ops if op.layer == first]
        n = max(1, round(len(self.ops) / self.n_layers))
        return self.ops[:n]

    @property
    def kinds_per_layer(self) -> Counter:
        return Counter(op.kind for op in self._first_layer_ops())

    def layer_ops(self, kind: str) -> list[OpInfo]:
        return [op for op in self._first_layer_ops() if op.kind == kind]

    def op_ms(self, op: OpInfo) -> tuple[float, float]:
        """(compute_ms, memory_ms) of one op -- the same roofline P4 uses."""
        return op.flops / self.tc_per_ms, op.bytes / self.bw_per_ms

    def roofline_ms(self, op: OpInfo) -> float:
        return max(self.op_ms(op))

    @property
    def step_ms(self) -> float:
        """Roofline time of the whole step/pass, launch overhead excluded."""
        return sum(self.roofline_ms(op) for op in self.ops)


_PRECISION_BYTES = {"fp32": 4, "float32": 4, "fp16": 2, "float16": 2, "bf16": 2,
                    "bfloat16": 2, "int8": 1, "fp8": 1}


def view(ir) -> P3View:
    model = _get(ir, ["model", "model_cfg", "model_config", "cfg"])
    wl = _get(ir, ["workload", "wl", "work"])

    def m(names, default=_MISSING):
        v = _get(model, names)
        v = _get(ir, names) if v is None else v
        if v is None:
            if default is _MISSING:
                raise ValueError(f"P3: IR model field missing (tried {names})")
            return default
        return int(_num(v))

    def w(names, default=_MISSING):
        v = _get(wl, names)
        v = _get(ir, names) if v is None else v
        if v is None:
            if default is _MISSING:
                raise ValueError(f"P3: IR workload field missing (tried {names})")
            return default
        return v

    n_layers = m(["n_layers", "num_layers", "layers", "num_hidden_layers"])
    hidden = m(["hidden", "hidden_size", "d_model", "hidden_dim"])
    n_heads = m(["n_heads", "num_heads", "heads", "num_attention_heads"])
    n_kv_heads = m(["n_kv_heads", "num_kv_heads", "kv_heads", "num_key_value_heads"], n_heads)
    head_dim = m(["head_dim", "d_head"], hidden // n_heads)

    batch = int(_num(w(["batch", "batch_size"])))
    mode = str(w(["mode", "phase"], "decode")).lower()
    mode = "prefill" if "prefill" in mode else "decode"
    prompt = int(_num(w(["prompt_len", "prompt_length", "seq_len", "sequence_length", "input_len"])))
    gen = int(_num(w(["gen_len", "generation_length", "output_len", "max_new_tokens"], 0)))
    prec = w(["dtype_bytes", "bytes_per_elem", "precision", "dtype"], 2)
    dtype_bytes = _PRECISION_BYTES.get(str(prec).lower(), None) or int(_num(prec) or 2)

    if mode == "prefill":
        q_len, kv_len = prompt, prompt
    else:
        q_len = 1
        kv = _get(wl, ["kv_len", "context_len", "ctx_len"])
        kv_len = int(_num(kv)) if kv is not None else prompt + gen // 2

    raw_ops = _get(ir, ["ops", "operations", "nodes"])
    if not raw_ops:
        raise ValueError("P3: IR has no ops list (looked for ir.ops / ir.operations / ir.nodes)")
    ops = []
    for op in raw_ops:
        layer = _get(op, ["layer", "layer_idx", "layer_id"])
        ops.append(OpInfo(
            kind=_kind(op),
            layer=int(layer) if layer is not None else None,
            flops=float(_num(_get(op, ["flops", "FLOPs", "n_flops"], 0)) or 0),
            bytes=float(_num(_get(op, ["bytes", "mem_bytes", "bytes_moved", "io_bytes"], 0)) or 0),
            launches=int(_num(_get(op, ["launches", "n_launches", "kernels"], 1)) or 1),
        ))

    return P3View(n_layers, hidden, n_heads, n_kv_heads, head_dim, batch, mode,
                  q_len, kv_len, dtype_bytes, _hw(ir), ops)


def make_candidate(name: str, d_latency_ms: float, d_mem_gb: float, d_launches: int,
                   accuracy_risk: str, reason: str):
    """Build the shared Candidate (P1's class if available, else the stand-in)."""
    kw = dict(name=name, d_latency_ms=round(float(d_latency_ms), 4),
              d_mem_gb=round(float(d_mem_gb), 4), d_launches=int(d_launches),
              accuracy_risk=accuracy_risk, reason=reason)
    try:
        return _Candidate(**kw)
    except TypeError:                       # positional-only constructor
        return _Candidate(*kw.values())
