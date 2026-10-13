"""
GPU hardware presets.

COMPUTE BASIS (important!)
--------------------------
Every `peak_fp16_tflops` value below is the DENSE FP16 Tensor Core
throughput (FP16 inputs, FP32 accumulate), WITHOUT structured sparsity.
NVIDIA datasheets often advertise a 2x larger "with sparsity" number; those
are NOT used here. Using one basis for every GPU keeps comparisons fair.

The same FP16 number is used as the compute roof for all precisions
(fp32, int8, int4, ...). See README "Assumptions" for why.

Memory capacities are the marketed sizes, treated as decimal GB. (Real HBM
capacity is usually a little larger, e.g. 80 GiB ~= 85.9 GB, so fit
decisions here are slightly conservative.)

Sources: NVIDIA product datasheets / architecture whitepapers
(A100 80GB SXM datasheet, H100 SXM5 datasheet, T4 datasheet,
Ada Lovelace (RTX 4090) architecture whitepaper, appendix table).
"""

from common import HardwareConfig

HARDWARE_PRESETS: dict[str, HardwareConfig] = {
    "a100": HardwareConfig(
        name="NVIDIA A100 80GB SXM",
        memory_gb=80,
        memory_bandwidth_gbps=2039,   # HBM2e
        peak_fp16_tflops=312,         # dense FP16 TC (624 with sparsity)
        notes="A100 80GB SXM datasheet",
    ),
    "h100": HardwareConfig(
        name="NVIDIA H100 80GB SXM5",
        memory_gb=80,
        memory_bandwidth_gbps=3350,   # HBM3
        peak_fp16_tflops=989.4,       # dense FP16 TC (1979 with sparsity)
        notes="H100 SXM5 datasheet",
    ),
    "t4": HardwareConfig(
        name="NVIDIA T4",
        memory_gb=16,
        memory_bandwidth_gbps=320,    # GDDR6
        peak_fp16_tflops=65,          # FP16/FP32 mixed-precision TC (Turing, no sparsity)
        notes="T4 datasheet",
    ),
    "rtx4090": HardwareConfig(
        name="NVIDIA GeForce RTX 4090",
        memory_gb=24,
        memory_bandwidth_gbps=1008,   # GDDR6X
        # GeForce cards run FP16-with-FP32-accumulate at half the rate of
        # FP16-accumulate. 165.2 is dense FP32-accumulate, matching the
        # data-center numbers above (330.3 = FP16 accumulate or sparsity).
        peak_fp16_tflops=165.2,
        notes="Ada Lovelace architecture whitepaper",
    ),
}


def get_hardware(name: str) -> HardwareConfig:
    """Look up a hardware preset by (case-insensitive) key, e.g. 'a100'."""
    key = name.strip().lower().replace("-", "").replace("_", "")
    if key not in HARDWARE_PRESETS:
        valid = ", ".join(HARDWARE_PRESETS)
        raise ValueError(f"Unknown hardware '{name}'. Valid options: {valid}")
    return HARDWARE_PRESETS[key]
