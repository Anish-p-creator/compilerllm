"""
Analyzer: combines the IR, the model and the hardware to estimate
memory requirements and a roofline-style latency.

NOTE: everything here is an ANALYTICAL ESTIMATE from formulas, not a
benchmark. Real kernels never reach 100% of peak compute or bandwidth.
"""

from dataclasses import dataclass

from common import (BYTES_PER_GB, Bound, HardwareConfig, IROp, ModelConfig,
                    bytes_for_elements, normalize_precision)
from ir_builder import build_ir
from models import count_parameters

KV_TENSORS = 2  # one K tensor and one V tensor are cached per layer


# --------------------------------------------------------------------------- #
# Memory footprint
# --------------------------------------------------------------------------- #

def weights_bytes(model: ModelConfig, precision: str) -> int:
    """Weight memory = parameter_count * bytes_per_parameter."""
    return bytes_for_elements(count_parameters(model), precision)


def weights_gb(model: ModelConfig, precision: str) -> float:
    """Weight memory in decimal GB (1 GB = 10^9 bytes)."""
    return weights_bytes(model, precision) / BYTES_PER_GB


def kv_cache_bytes(model: ModelConfig, sequence_length: int, batch_size: int,
                   precision: str) -> int:
    """
    KV cache = 2 * L * B * S * h_kv * d_h * bytes_per_element

    2     : one K and one V tensor
    L     : layers (every layer has its own cache)
    B * S : one cached vector per token
    h_kv * d_h : K (or V) width. With GQA h_kv < h, so the cache shrinks
                 by a factor h / h_kv (4x for Mistral-7B).
    """
    elements = (KV_TENSORS * model.num_layers * batch_size * sequence_length
                * model.num_kv_heads * model.head_dim)
    return bytes_for_elements(elements, precision)


def kv_cache_gb(model: ModelConfig, sequence_length: int, batch_size: int,
                precision: str) -> float:
    """KV cache size in decimal GB."""
    return kv_cache_bytes(model, sequence_length, batch_size, precision) / BYTES_PER_GB


def fits_in_memory(required_bytes: float, hardware: HardwareConfig) -> bool:
    """True if `required_bytes` fits in the GPU's memory capacity."""
    return required_bytes <= hardware.memory_bytes


# --------------------------------------------------------------------------- #
# Roofline estimate
# --------------------------------------------------------------------------- #

@dataclass
class OpEstimate:
    """Roofline estimate for one IR op."""

    op: IROp
    compute_time_s: float
    memory_time_s: float

    @property
    def estimated_time_s(self) -> float:
        # Compute and memory traffic are assumed to overlap perfectly,
        # so the slower of the two determines the time.
        return max(self.compute_time_s, self.memory_time_s)

    @property
    def bound(self) -> Bound:
        if self.compute_time_s >= self.memory_time_s:
            return Bound.COMPUTE_BOUND
        return Bound.MEMORY_BOUND


def estimate_op(op: IROp, hardware: HardwareConfig) -> OpEstimate:
    """
    compute_time = FLOPs / peak FLOP/s
    memory_time  = bytes / peak bytes/s
    estimated    = max(compute_time, memory_time)
    """
    return OpEstimate(
        op=op,
        compute_time_s=op.flops / hardware.peak_flops_per_s,
        memory_time_s=op.bytes / hardware.bandwidth_bytes_per_s,
    )


# --------------------------------------------------------------------------- #
# Full analysis
# --------------------------------------------------------------------------- #

@dataclass
class AnalysisReport:
    """Everything the analyzer computed for one (model, hardware, workload)."""

    model: ModelConfig
    hardware: HardwareConfig
    sequence_length: int
    batch_size: int
    precision: str
    estimates: list[OpEstimate]
    parameter_count: int
    weights_bytes: int
    kv_cache_bytes: int

    @property
    def total_flops(self) -> int:
        return sum(e.op.flops for e in self.estimates)

    @property
    def total_bytes(self) -> int:
        return sum(e.op.bytes for e in self.estimates)

    @property
    def arithmetic_intensity(self) -> float:
        return self.total_flops / self.total_bytes

    @property
    def total_memory_bytes(self) -> int:
        """Weights + KV cache (activations/framework overhead not included)."""
        return self.weights_bytes + self.kv_cache_bytes

    @property
    def fits(self) -> bool:
        return fits_in_memory(self.total_memory_bytes, self.hardware)

    @property
    def estimated_latency_s(self) -> float:
        """Sum of per-op times: ops run one after another, no overlap between ops."""
        return sum(e.estimated_time_s for e in self.estimates)


def analyze(model: ModelConfig, hardware: HardwareConfig, sequence_length: int,
            batch_size: int = 1, precision: str = "fp16") -> AnalysisReport:
    """Build the IR and run every analysis on it."""
    precision = normalize_precision(precision)
    ir = build_ir(model, sequence_length, batch_size, precision)
    return AnalysisReport(
        model=model,
        hardware=hardware,
        sequence_length=sequence_length,
        batch_size=batch_size,
        precision=precision,
        estimates=[estimate_op(op, hardware) for op in ir],
        parameter_count=count_parameters(model),
        weights_bytes=weights_bytes(model, precision),
        kv_cache_bytes=kv_cache_bytes(model, sequence_length, batch_size, precision),
    )
