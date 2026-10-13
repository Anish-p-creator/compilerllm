"""Stand-in for the team's cost model / optimizer, scoped to the P2 candidates.

It tries every subset of the proposed candidates, scores each with
    cost = w_lat * TPOT/TPOT0 + w_mem * Mem/Mem0 + w_ttft * TTFT/TTFT0
(lower is better), keeps only subsets that fit in GPU memory (if any do), and picks the best."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, List

from cost_model import Metrics, evaluate
from ir import Candidate, Context, PassResult
from passes import ALL_PASSES

DEFAULT_WEIGHTS = {
    "decode": {"latency": 1.0, "memory": 1.0, "ttft": 0.25},
    "prefill": {"latency": 0.25, "memory": 1.0, "ttft": 1.0},
    "mixed": {"latency": 1.0, "memory": 1.0, "ttft": 0.5},
}


@dataclass
class Trial:
    names: List[str]
    metrics: Metrics
    cost: float


@dataclass
class Plan:
    selected: List[str]
    baseline: Metrics
    optimized: Metrics
    weights: Dict[str, float]
    trials: List[Trial] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def run_passes(ctx: Context) -> List[PassResult]:
    return [p.run(ctx) for p in ALL_PASSES]


def _eval(ctx: Context, names, cands: Dict[str, Candidate]) -> Metrics:
    prefix = None
    if "prefix_share" in names:
        d = cands["prefix_share"].details
        prefix = dict(token_reuse_ratio=d["token_reuse_ratio"], flops_saved_fraction=d["flops_saved_fraction"])
    return evaluate(ctx, paged="paged_kv" in names, int8="int8_kv" in names, prefix=prefix)


def build_plan(ctx: Context, results: List[PassResult]) -> Plan:
    cands = {c.name: c for r in results for c in r.candidates}
    weights = dict(DEFAULT_WEIGHTS[ctx.workload.mode])
    weights.update(ctx.opts.objective)
    base = evaluate(ctx)

    trials: List[Trial] = []
    for k in range(len(cands) + 1):
        for names in combinations(sorted(cands), k):
            m = _eval(ctx, names, cands)
            cost = (weights["latency"] * m.tpot_ms / base.tpot_ms
                    + weights["memory"] * m.total_gib / base.total_gib
                    + weights["ttft"] * m.ttft_ms / base.ttft_ms)
            trials.append(Trial(list(names), m, cost))

    notes: List[str] = []
    feasible = [t for t in trials if t.metrics.fits]
    if feasible:
        pool = feasible
        if not base.fits:
            notes.append("baseline does NOT fit in usable GPU memory; only fitting plans were considered")
    else:
        pool = trials
        notes.append("no combination fits in usable GPU memory; picked the lowest-cost plan anyway")
    best = min(pool, key=lambda t: (t.cost, len(t.names)))
    return Plan(selected=best.names, baseline=base, optimized=best.metrics, weights=weights,
                trials=sorted(trials, key=lambda t: t.cost), notes=notes)
