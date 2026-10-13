"""
Shared data structures for the LLM IR + Analyzer (P1).

This module has NO imports from other project modules, so every other file
(hardware.py, models.py, ir_builder.py, analyzer.py, ...) can import from it
without creating circular dependencies.

Unit conventions used across the whole project
----------------------------------------------
* FLOPs  : one multiply OR one add counts as 1 FLOP (so a multiply-add = 2 FLOPs).
* Bytes  : plain bytes.
* GB     : DECIMAL gigabytes, 1 GB = 10**9 bytes (NOT GiB = 2**30 bytes).
           GPU vendors also use decimal units for bandwidth (GB/s, TB/s).
* Time   : seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

BYTES_PER_GB = 10**9  # decimal GB, used everywhere in this project
BITS_PER_BYTE = 8


# --------------------------------------------------------------------------- #
# Precision handling
# --------------------------------------------------------------------------- #

# Storing bits (not bytes) lets INT4 (= 0.5 byte) be handled exactly with
# integer arithmetic instead of floating-point "0.5 bytes".
PRECISION_BITS: dict[str, int] = {
    "fp32": 32,
    "fp16": 16,
    "bf16": 16,
    "int8": 8,
    "int4": 4,
}


def normalize_precision(precision: str) -> str:
    """Return the canonical lower-case precision name or raise ValueError."""
    name = precision.strip().lower()
    if name not in PRECISION_BITS:
        valid = ", ".join(PRECISION_BITS)
        raise ValueError(f"Unknown precision '{precision}'. Valid options: {valid}")
    return name


def bytes_per_element(precision: str) -> float:
    """Bytes needed to store one value (e.g. fp16 -> 2.0, int4 -> 0.5)."""
    return PRECISION_BITS[normalize_precision(precision)] / BITS_PER_BYTE


def bytes_for_elements(num_elements: int, precision: str) -> int:
    """
    Exact number of bytes needed to store `num_elements` values.

    bytes = ceil(num_elements * bits / 8), done in integer arithmetic so that
    INT4 (4 bits) never suffers from floating-point rounding.
    """
    bits = PRECISION_BITS[normalize_precision(precision)]
    total_bits = num_elements * bits
    return (total_bits + BITS_PER_BYTE - 1) // BITS_PER_BYTE


# --------------------------------------------------------------------------- #
# IR
# --------------------------------------------------------------------------- #

class Op(Enum):
    """Operation types that appear once per transformer layer in the IR."""

    QKV = "QKV"              # fused Q, K, V projections
    ATTENTION = "Attention"  # Q @ K^T  and  P @ V  (the two attention matmuls)
    SOFTMAX = "Softmax"      # row-wise softmax over the attention scores
    OUT_PROJ = "OutProj"     # attention output projection (W_o)
    MLP = "MLP"              # feed-forward network (gate/up/down for SwiGLU)


class Bound(Enum):
    """Roofline classification of an operation."""

    COMPUTE_BOUND = "COMPUTE_BOUND"
    MEMORY_BOUND = "MEMORY_BOUND"


@dataclass
class IROp:
    """
    One node of the IR: a single operation in a single transformer layer.

    Attributes:
        layer: transformer layer index (0-based).
        op:    which operation this is.
        flops: estimated floating-point operations.
        bytes: estimated bytes moved between GPU memory (DRAM/HBM) and the chip.
    """

    layer: int
    op: Op
    flops: int
    bytes: int

    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs performed per byte moved (FLOPs / byte)."""
        return self.flops / self.bytes if self.bytes else float("inf")


@dataclass
class Candidate:
    """
    A candidate version of the program for later optimization passes.

    P1 only produces the baseline candidate. Later stages (e.g. operator
    fusion, FlashAttention, quantization) can create new Candidates with a
    transformed IR and compare their estimated latency.
    """

    name: str
    ir: list[IROp] = field(default_factory=list)
    description: str = ""
    estimated_latency_s: float | None = None


# --------------------------------------------------------------------------- #
# Configurations
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ModelConfig:
    """
    Architecture hyper-parameters of a decoder-only transformer LLM.

    Attributes:
        name:                 display name, e.g. "LLaMA-7B".
        num_layers:           number of transformer blocks (L).
        hidden_size:          model dimension (d).
        num_attention_heads:  number of query heads (h).
        num_kv_heads:         number of key/value heads (h_kv).
                              h_kv == h  -> standard Multi-Head Attention
                              h_kv <  h  -> Grouped Query Attention (GQA)
        intermediate_size:    FFN hidden dimension (f).
        vocab_size:           vocabulary size (V).
        reported_params:      parameter count published by the model authors,
                              used only as a sanity check.
        gated_mlp:            True for SwiGLU-style MLP (3 weight matrices:
                              gate, up, down). False for classic 2-matrix MLP.
        tied_embeddings:      True if the LM head shares the embedding matrix.
    """

    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    reported_params: int
    gated_mlp: bool = True
    tied_embeddings: bool = False

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    @property
    def head_dim(self) -> int:
        """d_h = d / h"""
        return self.hidden_size // self.num_attention_heads

    @property
    def kv_dim(self) -> int:
        """Total width of K (or V): d_kv = h_kv * d_h. Equals d for MHA."""
        return self.num_kv_heads * self.head_dim

    @property
    def mlp_matrices(self) -> int:
        """Number of d x f weight matrices in the MLP (3 for SwiGLU, else 2)."""
        return 3 if self.gated_mlp else 2


@dataclass(frozen=True)
class HardwareConfig:
    """
    GPU characteristics used by the roofline model.

    Attributes:
        name:                   display name.
        memory_gb:              memory capacity in decimal GB, as marketed.
        memory_bandwidth_gbps:  peak DRAM/HBM bandwidth in GB/s (decimal).
        peak_fp16_tflops:       DENSE FP16 Tensor Core throughput
                                (FP16 inputs, FP32 accumulate, NO sparsity),
                                in TFLOP/s. This is the single compute basis
                                used for every GPU (see hardware.py).
        notes:                  where the numbers come from.
    """

    name: str
    memory_gb: float
    memory_bandwidth_gbps: float
    peak_fp16_tflops: float
    notes: str = ""

    @property
    def memory_bytes(self) -> float:
        return self.memory_gb * BYTES_PER_GB

    @property
    def peak_flops_per_s(self) -> float:
        return self.peak_fp16_tflops * 1e12

    @property
    def bandwidth_bytes_per_s(self) -> float:
        return self.memory_bandwidth_gbps * 1e9

    @property
    def ridge_point(self) -> float:
        """
        Arithmetic intensity (FLOPs/byte) where the roofline "bends".
        Ops with AI below this are memory-bound, above it compute-bound.
        """
        return self.peak_flops_per_s / self.bandwidth_bytes_per_s
