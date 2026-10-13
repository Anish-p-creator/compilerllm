"""Detailed GPU specs used by P3 (fusion, scheduling, auto-tuner).

P1's hardware.py owns the hardware presets. P3 only falls back to this table
for fields P1 does not provide: shared memory, registers, occupancy limits,
L2 size, FP32 rate and launch overhead. Values are public datasheet /
CUDA occupancy-calculator numbers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class GPUSpec:
    name: str
    sms: int                    # streaming multiprocessors
    mem_gb: float               # device memory
    mem_bw_gbps: float          # peak DRAM bandwidth, GB/s
    fp16_tflops: float          # dense tensor-core FP16 (FP32 accumulate)
    fp32_tflops: float          # CUDA-core FP32 (softmax / exp / rescale work)
    smem_per_sm_kb: float       # shared memory per SM
    smem_per_block_kb: float    # max shared memory one block may use
    max_threads_per_sm: int
    max_blocks_per_sm: int
    regs_per_sm: int
    l2_mb: float
    launch_us: float = 4.0      # typical back-to-back CUDA kernel launch cost


SPECS: dict[str, GPUSpec] = {
    # key order matters for lookup(): longer / more specific keys first
    "RTX4090": GPUSpec("NVIDIA RTX 4090", 128, 24, 1008, 165, 82.6, 100, 99, 1536, 24, 65536, 72),
    "A100": GPUSpec("NVIDIA A100 80GB PCIe", 108, 80, 1935, 312, 19.5, 164, 163, 2048, 32, 65536, 40),
    "H100": GPUSpec("NVIDIA H100 SXM", 132, 80, 3350, 989, 67.0, 228, 227, 2048, 32, 65536, 50),
    "T4": GPUSpec("NVIDIA T4", 40, 16, 320, 65, 8.1, 64, 64, 1024, 16, 65536, 4),
}


def lookup(name: str | None) -> GPUSpec | None:
    """Match a free-form device name ('NVIDIA A100', 'Tesla T4', 'RTX 4090')."""
    if not name:
        return None
    key = re.sub(r"[^A-Z0-9]", "", str(name).upper())
    for k, spec in SPECS.items():
        if k in key:
            return spec
    return None
