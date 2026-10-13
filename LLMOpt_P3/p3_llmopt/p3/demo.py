"""P3 standalone demo: Fusion + Scheduling + Auto-tuner.

    python -m p3.demo                      # all three scenarios
    python -m p3.demo --scenario 3         # one scenario
    python -m p3.demo --sweep              # auto-tuner on T4 / RTX4090 / A100 / H100
    python -m p3.demo --model Mistral-7B --gpu A100 --batch 1 --prompt 8192 --gen 256 --mode decode
    python -m p3.demo --json               # candidates as JSON (what P4 receives)
    python -m p3.demo --launch-in-p4       # d_latency_ms without launch overhead (see p3/perf.py)
"""
from __future__ import annotations

import argparse
import json
import sys

import tuner
from p3 import perf
from p3.irview import CANDIDATE_SOURCE, view
from p3.mock import MODELS, SCENARIOS, make_ir, scenario_ir
from passes import fusion, schedule

RULE = "=" * 78


def _cand_dict(c) -> dict:
    return {k: getattr(c, k) for k in ("name", "d_latency_ms", "d_mem_gb", "d_launches",
                                       "accuracy_risk", "reason")}


def _verdict(c, launch_ms: float) -> str:
    total = c.d_latency_ms + (0.0 if perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY else c.d_launches * launch_ms)
    if abs(total) < 1e-9:
        return "no change"
    return "faster" if total < 0 else "slower -> reject"


def run(title: str, ir, top_k: int) -> list:
    v = view(ir)
    print(f"\n{RULE}\n {title}\n{RULE}")
    print(f"  {getattr(ir.model, 'name', '?')} on {v.hw.name} | {v.mode} | batch {v.batch} | "
          f"{'KV length' if v.is_decode else 'prompt'} {v.kv_len} tokens")
    print(f"  {v.launches_per_step} kernel launches per {v.unit} | roofline work "
          f"{perf.fmt_ms(v.step_ms)} + launch overhead {perf.fmt_ms(v.launches_per_step * v.launch_ms)}")
    w_gb, kv_gb = getattr(ir, "weights_gb", None), getattr(ir, "kv_cache_gb", None)
    if w_gb is not None and kv_gb is not None:
        print(f"  memory: weights {w_gb:.1f} GB + KV cache {kv_gb:.1f} GB = {w_gb + kv_gb:.1f} GB "
              f"of {v.hw.mem_gb:g} GB")

    # ---------------------------------------------------------------- fusion
    print("\n[1] Kernel fusion (passes/fusion.py)")
    levels = fusion.evaluate_levels(ir)
    best = min(levels, key=lambda lv: lv.score_ms) if levels else None
    base = v.launches_per_step
    rows = [[f"L{lv.level}{' <-' if lv is best else ''}", lv.label,
             f"{base} -> {base + lv.d_launches}",
             perf.fmt_bytes(lv.traffic_saved_bytes),
             f"{perf.fmt_ms(lv.unfused_ms)} -> {perf.fmt_ms(lv.fused_ms)}",
             f"{lv.mem_saved_bytes / 1e9:.3f}",
             f"{lv.score_ms:+.3f}"] for lv in levels]
    print(perf.table(["level", "fused ops", "launches/step", "traffic saved/layer",
                      "time per layer", "mem saved GB", f"delta ms/{v.unit.split()[-1]}"], rows))
    if best and w_gb is not None and kv_gb is not None and best.mem_saved_bytes > 1e9:
        unfused_peak = w_gb + kv_gb + best.mem_saved_bytes / 1e9
        print(f"  unfused attention adds {best.mem_saved_bytes / 1e9:.2f} GB of score/prob buffers: "
              f"{unfused_peak:.1f} GB {'>' if unfused_peak > v.hw.mem_gb else '<='} {v.hw.mem_gb:g} GB"
              f"{' -> only the fused plan fits on this GPU' if unfused_peak > v.hw.mem_gb >= w_gb + kv_gb else ''}")

    # ------------------------------------------------------------ scheduling
    print("\n[2] Decode scheduling (passes/schedule.py)")
    sp, pp = schedule.plan_split_kv(ir), schedule.plan_persistent(ir)
    if sp is None:
        print("  not applicable in prefill: thousands of query tiles already fill the GPU "
              "and launches are a tiny share of the pass")
    else:
        if sp.forced:
            print(f"  split_kv   : {sp.units} blocks (batch {v.batch} x {v.n_kv_heads} KV heads) >= "
                  f"target {sp.target} (2 x {v.hw.sms} SMs) -> no split needed; costing 2 splits anyway: "
                  f"{sp.score_ms:+.3f} ms/step")
        else:
            print(f"  split_kv   : {sp.units} blocks (batch {v.batch} x {v.n_kv_heads} KV heads) < "
                  f"target {sp.target} (2 x {v.hw.sms} SMs) -> {sp.splits} splits -> "
                  f"{sp.units * sp.splits} blocks")
            print(f"               attention per layer {perf.fmt_ms(sp.t_unsplit_ms)} -> "
                  f"{perf.fmt_ms(sp.t_split_ms)} + {perf.fmt_ms(sp.t_reduce_ms)} reduction; "
                  f"+{sp.d_launches} launches; {sp.score_ms:+.3f} ms/step")
        print(f"  persistent : launch overhead = {pp.launch_share:.1%} of the step; one launch saves "
              f"{perf.fmt_ms(pp.launches * v.launch_ms - v.launch_ms)}, costs "
              f"{perf.fmt_ms(pp.sync_ms)} syncs + {perf.fmt_ms(pp.penalty_ms)} efficiency loss -> "
              f"{pp.score_ms:+.3f} ms/step")

    # ---------------------------------------------------------------- tuner
    r = tuner.tune(ir)
    print(f"\n[3] Auto-tuner (tuner.py): fused attention kernel, {r.total} configs, "
          f"{len(r.ranked)} feasible{f', {r.splits} KV splits' if r.splits > 1 else ''}")
    print(tuner.format_table(r, top_k))

    # ------------------------------------------------------------ candidates
    cands = fusion.propose(ir) + schedule.propose(ir) + tuner.propose(ir)
    print("\nCandidates returned to P4  (propose(ir) -> list[Candidate]; P4's optimizer decides)")
    rows = [[c.name, f"{c.d_latency_ms:+.3f}", f"{c.d_mem_gb:+.4f}", f"{c.d_launches:+d}",
             c.accuracy_risk, _verdict(c, v.launch_ms)] for c in cands]
    print(perf.table(["name", "d_latency_ms", "d_mem_gb", "d_launches", "risk", "on its own"], rows))
    for c in cands:
        print(f"  - {c.name}: {c.reason}")
    return cands


def sweep(ir_kwargs: dict, top_k: int) -> None:
    print(f"\n{RULE}\n Auto-tuner hardware sweep: same workload, four GPUs\n{RULE}")
    rows = []
    for g in ("T4", "RTX4090", "A100", "H100"):
        r = tuner.tune(make_ir(**{**ir_kwargs, "gpu": g}))
        default_ok = tuner.estimate(tuner.DEFAULT, view(make_ir(**{**ir_kwargs, "gpu": g})), r.splits).feasible
        rows.append([g, "yes" if default_ok else "NO", f"{len(r.ranked)}/{r.total}",
                     "keep default" if r.keeps_baseline else r.chosen.cfg.label(),
                     f"{r.chosen.time_ms * 1000:.1f}",
                     f"{r.baseline.time_ms / r.chosen.time_ms:.2f}x"])
    print(f"  workload: {ir_kwargs}")
    print(perf.table(["GPU", "default fits", "feasible", "chosen config", "us/layer", "vs hand-written"], rows))
    print("  'vs hand-written' = default config, or its hand-shrunk version where the default does not fit")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=int, choices=sorted(SCENARIOS))
    ap.add_argument("--model", choices=sorted(MODELS))
    ap.add_argument("--gpu", choices=["A100", "H100", "T4", "RTX4090"])
    ap.add_argument("--batch", type=int)
    ap.add_argument("--prompt", type=int)
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--mode", choices=["decode", "prefill"], default="decode")
    ap.add_argument("--sweep", action="store_true", help="auto-tuner across four GPUs")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--json", action="store_true", help="print candidates as JSON only")
    ap.add_argument("--launch-in-p4", action="store_true",
                    help="exclude launch overhead from d_latency_ms (P4 charges it)")
    a = ap.parse_args(argv)
    if a.launch_in_p4:
        perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY = False

    custom = a.model or a.gpu or a.batch or a.prompt
    if custom:
        kw = dict(model=a.model or "LLaMA-7B", gpu=a.gpu or "A100", batch=a.batch or 1,
                  prompt_len=a.prompt or 1024, gen_len=a.gen if a.mode == "decode" else 0, mode=a.mode)
        jobs = [("Custom workload", make_ir(**kw), kw)]
    else:
        nums = [a.scenario] if a.scenario else sorted(SCENARIOS)
        jobs = []
        for n in nums:
            title, ir = scenario_ir(n)
            kw = {k: v for k, v in SCENARIOS[n].items() if k != "title"}
            jobs.append((f"Scenario {n}: {title}", ir, kw))

    if a.json:
        out = {t: [_cand_dict(c) for c in fusion.propose(ir) + schedule.propose(ir) + tuner.propose(ir)]
               for t, ir, _ in jobs}
        print(json.dumps(out, indent=2))
        return 0

    print(f"P3 -- Fusion + Scheduling + Auto-tuner | Candidate class: {CANDIDATE_SOURCE}")
    print(f"d_latency_ms {'includes' if perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY else 'excludes'} "
          f"launch overhead | all numbers are model estimates (P4's benchmark is the measured data)")
    if a.sweep:
        sweep(jobs[0][2], a.top)
        return 0
    for title, ir, _ in jobs:
        run(title, ir, a.top)
    return 0


if __name__ == "__main__":
    sys.exit(main())
