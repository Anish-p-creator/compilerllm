"""
Pretty-printing of the IR and the analysis summary as plain-text tables.
All units are decimal (1 KB = 10^3 bytes, 1 GFLOP = 10^9 FLOPs).
"""

from analyzer import AnalysisReport, OpEstimate
from common import Op

FLOP_UNITS = ["FLOPs", "KFLOPs", "MFLOPs", "GFLOPs", "TFLOPs", "PFLOPs"]
BYTE_UNITS = ["B", "KB", "MB", "GB", "TB"]
UNIT_STEP = 1000.0


def _scale(value: float, units: list[str]) -> str:
    """Divide by 1000 until the value is < 1000, then attach the unit."""
    for unit in units[:-1]:
        if abs(value) < UNIT_STEP:
            return f"{value:.2f} {unit}"
        value /= UNIT_STEP
    return f"{value:.2f} {units[-1]}"


def format_flops(flops: float) -> str:
    return _scale(flops, FLOP_UNITS)


def format_bytes(num_bytes: float) -> str:
    return _scale(num_bytes, BYTE_UNITS)


def format_time(seconds: float) -> str:
    if seconds >= 1:
        return f"{seconds:.3f} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3f} ms"
    return f"{seconds * 1e6:.3f} us"


def _print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip()
    print(line)
    print("-" * len(line))
    for row in rows:
        print(" | ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())


def _estimate_row(label: str, est: OpEstimate) -> list[str]:
    return [label, est.op.op.value, format_flops(est.op.flops),
            format_bytes(est.op.bytes), f"{est.op.arithmetic_intensity:.1f}",
            format_time(est.estimated_time_s), est.bound.value]


TABLE_HEADERS = ["Layer", "Operation", "FLOPs", "Bytes", "AI (FLOP/B)", "Est. Time", "Bound"]


def print_header(report: AnalysisReport) -> None:
    print(f"Model: {report.model.name}")
    print(f"Sequence Length: {report.sequence_length}")
    print(f"Batch Size: {report.batch_size}")
    print(f"Precision: {report.precision.upper()}")
    print(f"Hardware: {report.hardware.name}")
    print()


def print_ir_table(report: AnalysisReport, max_layers: int | None = None) -> None:
    """Print one row per IR op. `max_layers` limits how many layers are shown."""
    shown = [e for e in report.estimates
             if max_layers is None or e.op.layer < max_layers]
    _print_table(TABLE_HEADERS, [_estimate_row(str(e.op.layer), e) for e in shown])

    hidden_layers = report.model.num_layers - (max_layers or report.model.num_layers)
    if hidden_layers > 0:
        print(f"... ({hidden_layers} more layers with identical ops not shown)")
    print()


def print_op_totals(report: AnalysisReport) -> None:
    """Aggregate each op type over all layers, to show where time goes."""
    print("Per-operation totals (all layers):")
    total_time = report.estimated_latency_s
    rows = []
    for op_type in Op:
        ests = [e for e in report.estimates if e.op.op is op_type]
        flops = sum(e.op.flops for e in ests)
        nbytes = sum(e.op.bytes for e in ests)
        time_s = sum(e.estimated_time_s for e in ests)
        rows.append([op_type.value, format_flops(flops), format_bytes(nbytes),
                     f"{flops / nbytes:.1f}", format_time(time_s),
                     f"{100 * time_s / total_time:.1f}%", ests[0].bound.value])
    _print_table(["Operation", "FLOPs", "Bytes", "AI (FLOP/B)", "Est. Time", "% Time", "Bound"], rows)
    print()


def print_summary(report: AnalysisReport) -> None:
    model, hw = report.model, report.hardware
    bar = "=" * 40
    print(bar)
    print("MODEL SUMMARY")
    print(bar)
    print()
    print(f"Model: {model.name}")
    print(f"Layers: {model.num_layers}")
    print(f"Hidden Size: {model.hidden_size}")
    print(f"Attention Heads: {model.num_attention_heads} (KV heads: {model.num_kv_heads}, "
          f"head dim: {model.head_dim})")
    print(f"Parameters: {report.parameter_count / 1e9:.3f} B "
          f"(reported ~{model.reported_params / 1e9:.2f} B)")
    print(f"Sequence Length: {report.sequence_length}")
    print(f"Batch Size: {report.batch_size}")
    print(f"Precision: {report.precision.upper()}")
    print()
    print(f"Weights: {format_bytes(report.weights_bytes)}")
    print(f"KV Cache: {format_bytes(report.kv_cache_bytes)}")
    print(f"Total Memory (weights + KV): {format_bytes(report.total_memory_bytes)}")
    print()
    print(f"GPU: {hw.name}")
    print(f"GPU Memory: {format_bytes(hw.memory_bytes)}")
    print(f"Fits in GPU Memory: {'YES' if report.fits else 'NO'}")
    print(f"Peak Compute (dense FP16 Tensor Core): {hw.peak_fp16_tflops} TFLOP/s")
    print(f"Memory Bandwidth: {hw.memory_bandwidth_gbps} GB/s")
    print(f"Ridge Point: {hw.ridge_point:.1f} FLOP/B")
    print()
    print(f"Total FLOPs: {format_flops(report.total_flops)}")
    print(f"Total Estimated Traffic: {format_bytes(report.total_bytes)}")
    print(f"Overall Arithmetic Intensity: {report.arithmetic_intensity:.1f} FLOP/B")
    print(f"Estimated Latency (prefill, roofline): {format_time(report.estimated_latency_s)}")
    print()
    print("Note: analytical roofline estimate at 100% of peak, not a measured benchmark.")


def print_report(report: AnalysisReport, max_layers: int | None = None) -> None:
    print_header(report)
    print_ir_table(report, max_layers)
    print_op_totals(report)
    print_summary(report)
