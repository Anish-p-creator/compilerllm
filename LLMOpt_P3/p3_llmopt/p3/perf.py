"""Shared helpers for P3: the integration switch, GPU-utilisation math and
plain-ASCII formatting (ASCII so output works in any terminal, incl. Windows).
"""
from __future__ import annotations

import math

# --------------------------------------------------------------------------
# Integration switch -- the ONLY setting that depends on how P4 stacks
# candidates. Every P3 candidate changes the launch count (d_launches).
#
#   True : d_latency_ms ALREADY includes the launch-overhead change implied
#          by d_launches. Use if P4 simply adds up d_latency_ms.
#   False: d_latency_ms EXCLUDES it. Use if P4's cost model charges
#          "launch overhead" itself from the final launch count
#          (otherwise the saving is counted twice).
#
# P3 always uses launch overhead internally to pick the best fusion level;
# this flag only changes the number that is handed to P4.
# --------------------------------------------------------------------------
COUNT_LAUNCH_OVERHEAD_IN_LATENCY = True


def wave_util(blocks: float, slots: float) -> float:
    """Useful fraction of `slots` when `blocks` equal blocks are spread over
    them (wave quantisation).

    32 blocks on 216 slots -> 0.15 (most of the GPU idle)
    300 blocks on 216 slots -> 300 / (2*216) = 0.69 (second wave half empty)
    """
    if blocks <= 0 or slots <= 0:
        return 0.0
    x = blocks / slots
    return x / math.ceil(x)


def fmt_bytes(n: float) -> str:
    n = float(n)
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f} {unit}"
    return f"{n:.0f} B"


def fmt_ms(ms: float, signed: bool = False) -> str:
    sign = "+" if signed and ms > 0 else ""
    if abs(ms) < 1:
        return f"{sign}{ms * 1000:.1f} us"
    return f"{sign}{ms:.2f} ms"


def table(headers: list[str], rows: list[list], indent: str = "  ") -> str:
    """Minimal fixed-width table: text left-aligned, numbers right-aligned."""
    cells = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cells)) if cells else len(h)
              for i, h in enumerate(headers)]

    def is_num(s: str) -> bool:
        return bool(s) and (s[0].isdigit() or s[0] in "+-.") and any(ch.isdigit() for ch in s)

    def line(r: list[str]) -> str:
        return indent + "  ".join(
            (c.rjust(w) if is_num(c) else c.ljust(w)) for c, w in zip(r, widths)).rstrip()

    sep = indent + "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), sep] + [line(r) for r in cells])
