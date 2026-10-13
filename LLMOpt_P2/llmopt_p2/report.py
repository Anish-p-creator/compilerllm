"""Console + JSON rendering of the optimization report."""
from __future__ import annotations

from dataclasses import asdict
from typing import List

from cost_model import Metrics
from ir import GIB, Context, PassResult
from planner import Plan

W = 78


def _pct(a: float, b: float) -> str:
    if a == 0:
        return "-"
    d = (b - a) / a * 100
    return f"{'↑' if d > 0 else '↓' if d < 0 else '='} {abs(d):.1f}%"


def _row(label: str, a: str, b: str, change: str) -> str:
    return f"  {label:<30}{a:>14}{b:>14}{change:>14}"


def render_text(ctx: Context, results: List[PassResult], plan: Plan) -> str:
    m, w, hw, o = ctx.model, ctx.workload, ctx.hw, ctx.opts
    L: List[str] = []
    add = L.append
    add("=" * W)
    add("LLMOpt · P2 · KV-cache + prefix-sharing passes".center(W))
    add("=" * W)
    add(f"Model     : {m.name}  ({m.layers} layers, {m.kv_heads} KV heads, head_dim {m.head_dim}, {m.precision.upper()})")
    add(f"Hardware  : {hw.name}  ({hw.memory_gib:g} GiB, {hw.bandwidth_gbps:g} GB/s)")
    add(f"Workload  : batch {w.batch_size} | prompt {w.prompt_length} | generate {w.generation_length} "
        f"| max_ctx {w.max_context} | mode {w.mode}")

    kv = next(r for r in results if r.pass_name == "kv_cache")
    add("")
    add("1. KV-CACHE ANALYSIS")
    add("-" * W)
    add(f"  KV size per token        : {kv.info['kv_bytes_per_token'] / 1024:.0f} KiB")
    add(f"  Reserved KV (baseline)   : {kv.info['reserved_kv_gib']:.2f} GiB = "
        f"{kv.info['memory_pressure'] * 100:.1f}% of GPU memory "
        f"[{'HIGH' if kv.info['high_pressure'] else 'low'} pressure, threshold {o.pressure_threshold * 100:.0f}%]")

    add("")
    add("2. CANDIDATES PROPOSED")
    add("-" * W)
    n = 0
    for r in results:
        for c in r.candidates:
            n += 1
            add(f"  [{n}] {c.name:<12} {c.description}")
            add(f"      memory saved   : {c.memory_saved_bytes / GIB:.2f} GiB (stand-alone)")
            if c.name == "prefix_share":
                d = c.details
                add(f"      compute saved  : {d['prefill_tokens_saved']:,.0f} prefill tokens skipped, "
                    f"{d['compute_saved_tflop']:,.1f} TFLOP ({d['flops_saved_fraction'] * 100:.1f}% of prefill)")
                add(f"      longest prefix : {d['longest_shared_prefix']} tokens shared by "
                    f"{d['longest_shared_prefix_prompts']}/{d['num_prompts']} prompts; common to ALL: {d['common_prefix_all']}")
                if d["top_segments"]:
                    t = d["top_segments"][0]
                    add(f"      best segment   : tokens [{t['start']}, {t['start'] + t['length']}) shared by "
                        f"{t['count']} prompts -> {t['length'] * (t['count'] - 1):,} tokens deduplicated")
                add(f"      starts with    : \"{d['longest_shared_prefix_preview']}\"")
                add(f"      trie           : {d['total_prompt_tokens']:,} prompt tokens -> {d['trie_stored_tokens']:,} stored "
                    f"({d['prompt_source']})")
                if d["extrapolated"]:
                    add(f"      note           : ratios measured on {d['num_prompts']} sample prompts, applied to the workload")
            else:
                add(f"      attention lat. : {c.latency_delta_pct:+.1f}%    accuracy risk: {c.accuracy_risk}")
            if c.name == "paged_kv":
                add(f"      utilization    : {c.details['utilization'] * 100:.0f}% of reserved slots are ever used")
            if c.name == "int8_kv":
                a = c.details["accuracy"]
                add(f"      measured error : {a['rel_error'] * 100:.3f}% rel. attention-output error "
                    f"(worst of sweep {a['worst_rel_error'] * 100:.2f}%), cosine {a['mean_cosine']:.5f}")
                add(f"                       data: {a['data']}")
                add(f"      size           : {c.details['size_ratio'] * 100:.1f}% of FP16 "
                    f"({c.details['bytes_per_element']:.3f} B/elem incl. scales)")
            add(f"      why            : {c.rationale}")
    if n == 0:
        add("  (none)")
    rej = [x for r in results for x in r.rejections]
    if rej:
        add("")
        add("  Rejected:")
        for x in rej:
            add(f"   ✗ {x.name:<12} {x.reason}")

    add("")
    add("3. COST-MODEL SEARCH  (cost = {latency:g}·TPOT + {memory:g}·Mem + {ttft:g}·TTFT, normalized)".format(**plan.weights))
    add("-" * W)
    for t in plan.trials[:8]:
        mark = "◀ selected" if t.names == plan.selected else ""
        add(f"  cost {t.cost:5.3f}  {'fits' if t.metrics.fits else 'OOM '}  "
            f"{('+'.join(t.names) or 'baseline'):<40}{mark}")

    add("")
    add("4. RESULT")
    add("-" * W)
    add(f"  Selected: {' + '.join(plan.selected) if plan.selected else 'none (baseline is already best)'}")
    for note in plan.notes:
        add(f"  ! {note}")
    b, p = plan.baseline, plan.optimized
    add("")
    add(_row("Metric", "Baseline", "Optimized", "Change"))
    add("  " + "-" * 70)
    rows = [
        ("KV cache (GiB)", f"{b.kv_gib:.2f}", f"{p.kv_gib:.2f}", _pct(b.kv_gib, p.kv_gib)),
        ("Peak GPU memory (GiB)", f"{b.total_gib:.2f}", f"{p.total_gib:.2f}", _pct(b.total_gib, p.total_gib)),
        ("Fits in usable memory", "yes" if b.fits else "NO (OOM)", "yes" if p.fits else "NO (OOM)", ""),
        ("Max concurrent requests", f"{b.max_batch}", f"{p.max_batch}", _pct(b.max_batch, p.max_batch)),
        ("KV read / decode step (GiB)", f"{b.kv_read_gib_per_step:.2f}", f"{p.kv_read_gib_per_step:.2f}",
         _pct(b.kv_read_gib_per_step, p.kv_read_gib_per_step)),
        ("TPOT (ms)", f"{b.tpot_ms:.2f}", f"{p.tpot_ms:.2f}", _pct(b.tpot_ms, p.tpot_ms)),
        ("Throughput (tok/s)", f"{b.throughput_tok_s:,.0f}", f"{p.throughput_tok_s:,.0f}",
         _pct(b.throughput_tok_s, p.throughput_tok_s)),
        ("TTFT, batch prefill (ms)", f"{b.ttft_ms:,.0f}", f"{p.ttft_ms:,.0f}", _pct(b.ttft_ms, p.ttft_ms)),
        ("Prefill tokens computed", f"{b.prefill_tokens:,}", f"{p.prefill_tokens:,}", _pct(b.prefill_tokens, p.prefill_tokens)),
    ]
    for r in rows:
        add(_row(*r))
    int8 = next((c for r_ in results for c in r_.candidates if c.name == "int8_kv"), None)
    if int8 and "int8_kv" in plan.selected:
        a = int8.details["accuracy"]
        add(_row("Attention-output error (INT8)", "0", f"{a['rel_error'] * 100:.3f}%", "measured"))
    else:
        add(_row("Attention-output error", "0", "0", "exact"))
    add("")
    add("  Memory/latency/TTFT are analytical roofline estimates; trie statistics and the")
    add("  INT8 error are computed (not estimated). See README for assumptions.")
    add("=" * W)
    return "\n".join(L)


def to_json(ctx: Context, results: List[PassResult], plan: Plan) -> dict:
    w = asdict(ctx.workload)
    w.pop("prompts", None)
    return dict(
        model=asdict(ctx.model), hardware=asdict(ctx.hw), workload=w, options=asdict(ctx.opts),
        passes=[dict(pass_name=r.pass_name, info=r.info,
                     candidates=[asdict(c) for c in r.candidates],
                     rejections=[asdict(x) for x in r.rejections]) for r in results],
        plan=dict(selected=plan.selected, weights=plan.weights, notes=plan.notes,
                  baseline=asdict(plan.baseline), optimized=asdict(plan.optimized),
                  trials=[dict(names=t.names, cost=t.cost, metrics=asdict(t.metrics)) for t in plan.trials]))
