"""Shared types for LLMOpt (P2 slice): model/workload/hardware specs, the input
loader, and the Candidate / Rejection / PassResult types every pass returns."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

GIB = 1024 ** 3
DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2, "fp8": 1, "int8": 1}

MODEL_PRESETS = {
    "llama-7b": dict(layers=32, hidden=4096, heads=32, kv_heads=32, head_dim=128, params_b=6.74),
    "llama-13b": dict(layers=40, hidden=5120, heads=40, kv_heads=40, head_dim=128, params_b=13.0),
    "llama-3-8b": dict(layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128, params_b=8.03),
    "mistral-7b": dict(layers=32, hidden=4096, heads=32, kv_heads=8, head_dim=128, params_b=7.24),
}

# memory in GiB, bandwidth in GB/s, dense FP16 tensor-core TFLOPs
HARDWARE_PRESETS = {
    "a10080gb": dict(name="NVIDIA A100-80GB", memory_gib=80, bandwidth_gbps=1935, fp16_tflops=312),
    "a10040gb": dict(name="NVIDIA A100-40GB", memory_gib=40, bandwidth_gbps=1555, fp16_tflops=312),
    "h100": dict(name="NVIDIA H100-80GB", memory_gib=80, bandwidth_gbps=3350, fp16_tflops=989),
    "l4": dict(name="NVIDIA L4", memory_gib=24, bandwidth_gbps=300, fp16_tflops=121),
    "rtx4090": dict(name="NVIDIA RTX 4090", memory_gib=24, bandwidth_gbps=1008, fp16_tflops=82.6),
}


# --------------------------------------------------------------------------- specs
@dataclass
class ModelSpec:
    name: str
    layers: int
    hidden: int
    heads: int
    kv_heads: int
    head_dim: int
    params_b: float
    precision: str = "fp16"

    @property
    def dtype_bytes(self) -> int:
        return DTYPE_BYTES[self.precision]

    def kv_bytes_per_token(self, elem_bytes: Optional[float] = None) -> float:
        """K and V, all layers, all KV heads, for ONE token."""
        b = self.dtype_bytes if elem_bytes is None else elem_bytes
        return 2 * self.layers * self.kv_heads * self.head_dim * b

    @property
    def weight_bytes(self) -> float:
        return self.params_b * 1e9 * self.dtype_bytes


@dataclass
class Hardware:
    name: str
    memory_gib: float
    bandwidth_gbps: float
    fp16_tflops: float

    @property
    def memory_bytes(self) -> float:
        return self.memory_gib * GIB


@dataclass
class Workload:
    batch_size: int
    prompt_length: int
    generation_length: int
    max_context: int
    mode: str = "decode"                       # decode | prefill | mixed
    prompts: Optional[list] = None             # list[str | list[int]]
    prompt_spec: Optional[dict] = None         # synthetic prompt generator
    seed: int = 0


@dataclass
class Options:
    pressure_threshold: float = 0.60           # KV / GPU memory above which KV passes fire
    block_size: int = 16                       # paged-KV block (tokens)
    chunk_size: int = 16                       # prefix-sharing granularity (tokens)
    paged_overhead_pct: float = 3.0            # attention latency cost of paging
    int8_dequant_overhead_pct: float = 2.0     # dequant cost on top of the byte savings
    decode_fraction: float = 0.7               # used for mode == "mixed"
    usable_memory_fraction: float = 0.92       # fraction of GPU memory usable (runtime overhead)
    mem_bw_efficiency: float = 0.75            # achievable fraction of peak HBM bandwidth
    prefill_mfu: float = 0.5                   # achievable fraction of peak FLOPs in prefill
    objective: Dict[str, float] = field(default_factory=dict)  # latency / memory / ttft weights


@dataclass
class Context:
    model: ModelSpec
    workload: Workload
    hw: Hardware
    opts: Options
    _prompts_cache: Any = None


# ------------------------------------------------------------------- pass results
@dataclass
class Candidate:
    name: str
    pass_name: str
    description: str
    memory_saved_bytes: float                  # stand-alone saving vs the baseline
    latency_delta_pct: float                   # change in attention latency (+ = slower)
    accuracy_risk: str                         # none | low | medium | high
    prefill_compute_saved_flops: float = 0.0
    rationale: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Rejection:
    name: str
    reason: str


@dataclass
class PassResult:
    pass_name: str
    candidates: List[Candidate] = field(default_factory=list)
    rejections: List[Rejection] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------------ loading
def parse_num(v: Any, field_name: str) -> float:
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[-+]?\d*\.?\d+", v)
        if m:
            return float(m.group())
    raise ValueError(f"cannot read a number from {field_name}={v!r}")


def _norm_device(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower()).replace("nvidia", "")


def build_model(spec: Any, precision: Optional[str]) -> ModelSpec:
    if isinstance(spec, str):
        spec = {"preset": spec}
    if not isinstance(spec, dict):
        raise ValueError("'model' must be a preset name or an object")
    base: Dict[str, Any] = {}
    name = spec.get("name")
    if "preset" in spec:
        key = spec["preset"].lower()
        if key not in MODEL_PRESETS:
            raise ValueError(f"unknown model preset {key!r}; choose from {sorted(MODEL_PRESETS)}")
        base = dict(MODEL_PRESETS[key])
        name = name or key
    base.update({k: v for k, v in spec.items() if k not in ("preset", "name", "precision")})
    missing = [k for k in ("layers", "hidden", "heads", "kv_heads", "head_dim", "params_b") if k not in base]
    if missing:
        raise ValueError(f"model is missing fields: {missing}")
    prec = (precision or spec.get("precision") or "fp16").lower()
    if prec not in DTYPE_BYTES:
        raise ValueError(f"unsupported precision {prec!r}; choose from {sorted(DTYPE_BYTES)}")
    return ModelSpec(name=name or "custom", precision=prec,
                     **{k: (float(base[k]) if k == "params_b" else int(base[k])) for k in
                        ("layers", "hidden", "heads", "kv_heads", "head_dim", "params_b")})


def build_hardware(spec: Any) -> Hardware:
    if isinstance(spec, str):
        spec = {"device": spec}
    device = spec.get("device", "custom")
    key = _norm_device(device)
    preset = None
    for k, v in HARDWARE_PRESETS.items():
        if key == k or key.startswith(k) or k.startswith(key):
            preset = dict(v)
            break
    mem = spec.get("gpu_memory", spec.get("memory_gib"))
    bw = spec.get("memory_bandwidth", spec.get("bandwidth_gbps"))
    tf = spec.get("fp16_tflops")
    if preset is None:
        if mem is None or bw is None:
            raise ValueError(f"unknown device {device!r}: give gpu_memory and memory_bandwidth "
                             f"(or use one of {[v['name'] for v in HARDWARE_PRESETS.values()]})")
        preset = dict(name=device, memory_gib=0, bandwidth_gbps=0, fp16_tflops=100.0)
    if mem is not None:
        preset["memory_gib"] = parse_num(mem, "gpu_memory")
    if bw is not None:
        preset["bandwidth_gbps"] = parse_num(bw, "memory_bandwidth")
    if tf is not None:
        preset["fp16_tflops"] = parse_num(tf, "fp16_tflops")
    return Hardware(**preset)


def build_workload(spec: Dict[str, Any]) -> Workload:
    try:
        w = Workload(
            batch_size=int(spec["batch_size"]),
            prompt_length=int(spec["prompt_length"]),
            generation_length=int(spec["generation_length"]),
            max_context=int(spec.get("max_context", int(spec["prompt_length"]) + int(spec["generation_length"]))),
            mode=str(spec.get("mode", "decode")).lower(),
            prompts=spec.get("prompts"),
            prompt_spec=spec.get("prompt_spec"),
            seed=int(spec.get("seed", 0)),
        )
    except KeyError as e:
        raise ValueError(f"workload is missing field {e}") from None
    if w.mode not in ("decode", "prefill", "mixed"):
        raise ValueError("workload.mode must be decode, prefill or mixed")
    if min(w.batch_size, w.prompt_length, w.generation_length) < 1:
        raise ValueError("batch_size, prompt_length and generation_length must be >= 1")
    if w.prompt_length + w.generation_length > w.max_context:
        raise ValueError(f"prompt_length + generation_length ({w.prompt_length + w.generation_length}) "
                         f"exceeds max_context ({w.max_context})")
    return w


def load_context(data: Dict[str, Any]) -> Context:
    for k in ("model", "hardware", "workload"):
        if k not in data:
            raise ValueError(f"input is missing the '{k}' section")
    wl = build_workload(data["workload"])
    model = build_model(data["model"], data["workload"].get("precision"))
    hw = build_hardware(data["hardware"])
    known = set(Options.__dataclass_fields__)
    raw = data.get("options", {})
    bad = set(raw) - known
    if bad:
        raise ValueError(f"unknown options: {sorted(bad)}")
    return Context(model=model, workload=wl, hw=hw, opts=Options(**raw))


def load_context_file(path: str) -> Context:
    with open(path) as f:
        return load_context(json.load(f))


# ------------------------------------------------------------------------ prompts
_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def resolve_prompts(wl: Workload) -> Optional[Dict[str, Any]]:
    """Return {'sequences': [[int,...],...], 'source': str, 'vocab_rev': dict|None} or None."""
    if wl.prompts:
        vocab: Dict[str, int] = {}
        seqs: List[List[int]] = []
        text = False
        for p in wl.prompts:
            if isinstance(p, str):
                text = True
                seqs.append([vocab.setdefault(t, len(vocab) + 1) for t in _TOKEN_RE.findall(p)])
            else:
                seqs.append([int(t) for t in p])
        return dict(sequences=seqs, source="text prompts (regex word tokenizer)" if text else "token ids",
                    vocab_rev={v: k for k, v in vocab.items()} if text else None, synthetic=False)
    if wl.prompt_spec is not None:
        return dict(sequences=generate_prompts(wl), source="synthetic prompt_spec", vocab_rev=None, synthetic=True)
    return None


def generate_prompts(wl: Workload) -> List[List[int]]:
    """batch_size prompts of prompt_length tokens: [system][few-shot group][unique tail]."""
    s = wl.prompt_spec or {}
    sys_n = int(s.get("system_prompt_tokens", 0))
    groups = int(s.get("fewshot_groups", 0))
    fs_n = int(s.get("fewshot_tokens", 0)) if groups > 0 else 0
    if sys_n + fs_n > wl.prompt_length:
        raise ValueError("prompt_spec: system_prompt_tokens + fewshot_tokens exceeds prompt_length")
    rng = random.Random(wl.seed)
    out = []
    for r in range(wl.batch_size):
        toks = list(range(1, sys_n + 1))
        if groups:
            g = r % groups
            toks += [1_000_000 * (g + 1) + k for k in range(fs_n)]
        tail = wl.prompt_length - len(toks)
        toks += [100_000_000 + rng.randrange(10 ** 8) for _ in range(tail)]
        out.append(toks)
    return out
