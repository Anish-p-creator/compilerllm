"""
Command-line entry point.

Example:
    python main.py --model llama-7b --hardware a100 --seq-len 2048 --batch-size 1 --precision fp16
"""

import argparse
import sys

from analyzer import analyze
from common import PRECISION_BITS, normalize_precision
from hardware import HARDWARE_PRESETS, get_hardware
from models import MODEL_PRESETS, get_model
from report import print_report


def positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive integer, got {value}")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hardware-aware IR builder and roofline analyzer for LLM inference (prefill).")
    parser.add_argument("--model", default="llama-7b",
                        help=f"model preset ({', '.join(MODEL_PRESETS)})")
    parser.add_argument("--hardware", default="a100",
                        help=f"GPU preset ({', '.join(HARDWARE_PRESETS)})")
    parser.add_argument("--seq-len", type=positive_int, default=2048,
                        help="sequence length in tokens (default: 2048)")
    parser.add_argument("--batch-size", type=positive_int, default=1,
                        help="batch size (default: 1)")
    parser.add_argument("--precision", default="fp16",
                        help=f"numeric precision ({', '.join(PRECISION_BITS)})")
    parser.add_argument("--max-layers", type=positive_int, default=None,
                        help="only print the IR for the first N layers (default: all)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        model = get_model(args.model)
        hardware = get_hardware(args.hardware)
        precision = normalize_precision(args.precision)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    report = analyze(model, hardware, args.seq_len, args.batch_size, precision)
    print_report(report, args.max_layers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
