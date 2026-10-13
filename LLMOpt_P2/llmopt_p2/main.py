#!/usr/bin/env python3
"""LLMOpt P2 CLI:  python main.py --input samples/high_pressure.json [--json out.json]"""
import argparse
import glob
import json
import os
import sys

from ir import load_context_file
from planner import build_plan, run_passes
from report import render_text, to_json

HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(path: str, json_out: str = None) -> None:
    ctx = load_context_file(path)
    results = run_passes(ctx)
    plan = build_plan(ctx, results)
    print(f"\n>>> input: {path}")
    print(render_text(ctx, results, plan))
    if json_out:
        with open(json_out, "w") as f:
            json.dump(to_json(ctx, results, plan), f, indent=2, default=str)
        print(f"JSON report written to {json_out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LLMOpt P2: KV-cache + prefix-sharing optimization passes")
    ap.add_argument("-i", "--input", help="input JSON (default: samples/high_pressure.json)")
    ap.add_argument("--all", action="store_true", help="run every file in samples/")
    ap.add_argument("--json", help="also write the full report as JSON to this path")
    args = ap.parse_args(argv)
    try:
        if args.all:
            for p in sorted(glob.glob(os.path.join(HERE, "samples", "*.json"))):
                run_one(p)
        else:
            run_one(args.input or os.path.join(HERE, "samples", "high_pressure.json"), args.json)
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
