"""Stand-in for P1's common.py + ir/builder.py, used ONLY so P3 can be built,
tested and demoed before P1 pushes. It mirrors the team-split doc:

  * Candidate(name, d_latency_ms, d_mem_gb, d_launches, accuracy_risk, reason)
  * model presets LLaMA-7B, Mistral-7B; hardware presets A100/H100/T4/RTX4090
  * build_ir(): per-layer ops QKV, Attention, Softmax, OutProj, MLP with
    flops and bytes, plus kv_cache_gb and weights_gb

The ops describe the UNOPTIMISED (naive) execution: attention writes the
score matrix S to memory, softmax reads it and writes P, the PV matmul reads P.

P3's passes never import this directly; they read any IR through
p3/irview.py, so at integration this file can simply be ignored or deleted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from p3.gpu_specs import SPECS, GPUSpec


@dataclass
class Candidate:
    name: str
    d_latency_ms: float
    d_mem_gb: float
    d_launches: int
    accuracy_risk: str
    reason: str


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layers: int
    hidden: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    ffn: int
    vocab: int

    @property
    def params(self) -> int:
        qkv = self.hidden * (self.n_heads + 2 * self.n_kv_heads) * self.head_dim
        out = self.n_heads * self.head_dim * self.hidden
        mlp = 3 * self.hidden * self.ffn            # SwiGLU: gate, up, down
        norms = 2 * self.hidden
        return self.n_layers * (qkv + out + mlp + norms) + 2 * self.vocab * self.hidden


MODELS = {
    "LLaMA-7B": ModelConfig("LLaMA-7B", 32, 4096, 32, 32, 128, 11008, 32000),
    "Mistral-7B": ModelConfig("Mistral-7B", 32, 4096, 32, 8, 128, 14336, 32000),  # GQA: 8 KV heads
}


@dataclass
class Workload:
    batch: int
    prompt_len: int
    gen_len: int
    mode: str = "decode"          # "decode" or "prefill"
    dtype_bytes: int = 2          # FP16

    @property
    def kv_len(self) -> int:
        """KV length the attention kernel reads: the prompt in prefill, the
        average context over the generation in decode."""
        if self.mode == "prefill":
            return self.prompt_len
        return self.prompt_len + self.gen_len // 2


@dataclass
class Op:
    name: str
    kind: str
    layer: int
    flops: float
    bytes: float


@dataclass
class IR:
    model: ModelConfig
    workload: Workload
    hw: GPUSpec
    ops: list[Op] = field(default_factory=list)
    kv_cache_gb: float = 0.0
    weights_gb: float = 0.0


def build_ir(model: str | ModelConfig, workload: Workload, hw: str | GPUSpec) -> IR:
    m = MODELS[model] if isinstance(model, str) else model
    g = SPECS[hw] if isinstance(hw, str) else hw
    w = workload
    b = w.dtype_bytes
    B, L, D, F = w.batch, m.n_layers, m.hidden, m.ffn
    H, Hkv, d = m.n_heads, m.n_kv_heads, m.head_dim
    sq = 1 if w.mode == "decode" else w.prompt_len    # query tokens per sequence
    kv = w.kv_len
    T = B * sq                                          # tokens this pass
    qkv_out = (H + 2 * Hkv) * d

    ops: list[Op] = []
    for l in range(L):
        s_elems = B * H * sq * kv                       # attention score matrix
        q_b = T * H * d * b
        kv_b = 2 * B * kv * Hkv * d * b
        ops += [
            Op("QKV", "qkv", l, 2 * T * D * qkv_out, b * (D * qkv_out + T * D + T * qkv_out)),
            # QK^T and PV matmuls: read Q, K, V, write S, read P, write O
            Op("Attention", "attention", l, 4 * B * H * sq * kv * d, q_b + kv_b + 2 * s_elems * b + q_b),
            Op("Softmax", "softmax", l, 5 * s_elems, 2 * s_elems * b),   # read S, write P
            Op("OutProj", "out_proj", l, 2 * T * H * d * D, b * (H * d * D + 2 * T * D)),
            Op("MLP", "mlp", l, 2 * T * D * F * 3, b * (3 * D * F + 2 * T * D + 3 * T * F)),
        ]

    kv_max = w.prompt_len + (w.gen_len if w.mode == "decode" else 0)
    kv_cache_gb = 2 * L * B * kv_max * Hkv * d * b / 1e9
    return IR(m, w, g, ops, kv_cache_gb, m.params * b / 1e9)


# Three scenarios chosen so that each P3 optimisation is selected in some
# and rejected in others. All of them fit in GPU memory once optimised.
SCENARIOS = {
    1: dict(title="Chatbot, latency-critical (small batch decode)",
            model="LLaMA-7B", gpu="A100", batch=1, prompt_len=1024, gen_len=256, mode="decode"),
    2: dict(title="High-throughput serving (large batch decode)",
            model="Mistral-7B", gpu="A100", batch=256, prompt_len=1024, gen_len=256, mode="decode"),
    3: dict(title="Long-prompt prefill on a small GPU",
            model="Mistral-7B", gpu="T4", batch=2, prompt_len=4096, gen_len=0, mode="prefill"),
}


def make_ir(model: str, gpu: str, batch: int, prompt_len: int, gen_len: int, mode: str) -> IR:
    return build_ir(model, Workload(batch, prompt_len, gen_len, mode), gpu)


def scenario_ir(n: int) -> tuple[str, IR]:
    s = dict(SCENARIOS[n])
    title = s.pop("title")
    return title, make_ir(**s)
