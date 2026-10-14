"""
Unified Optimization Pipeline — combines all three approaches into a single
compiler-style pass pipeline for LLM inference optimization.

Phase 1 (P1): IR Construction + Roofline Analysis
Phase 2 (P2): KV-Cache + Prefix-Sharing Passes
Phase 3 (P3): Decode Scheduling + Kernel Fusion + Auto-Tuner

Uses importlib to load modules from each approach's directory without
polluting sys.path (avoids "passes" name collisions between P2 and P3).
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Absolute directories for the three approaches
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_P1_DIR = os.path.join(_ROOT, "LLMOpt_P1", "compiler-project")
_P2_DIR = os.path.join(_ROOT, "LLMOpt_P2", "llmopt_p2")
_P3_DIR = os.path.join(_ROOT, "LLMOpt_P3", "p3_llmopt")


def _load_module(name: str, directory: str, filename: str | None = None):
    """Load a single .py module from *directory* under the given *name*."""
    if filename is None:
        filename = name + ".py"
    path = os.path.join(directory, filename)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # make it importable by sub-imports
    spec.loader.exec_module(mod)
    return mod


def _load_package(pkg_name: str, directory: str):
    """Load a package (directory with __init__.py) under *pkg_name*."""
    init = os.path.join(directory, "__init__.py")
    spec = importlib.util.spec_from_file_location(
        pkg_name, init,
        submodule_search_locations=[directory],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Phase 1 imports (no sub-package collision risk, but use importlib for safety)
# ---------------------------------------------------------------------------
# P1 modules have simple flat imports among themselves, so we add P1 to path.
if _P1_DIR not in sys.path:
    sys.path.insert(0, _P1_DIR)

from common import (
    BYTES_PER_GB, PRECISION_BITS, HardwareConfig, IROp, ModelConfig, Op, Bound,
    bytes_for_elements, normalize_precision,
)
from ir_builder import build_ir as p1_build_ir
from models import MODEL_PRESETS as P1_MODELS, get_model as p1_get_model, count_parameters
from hardware import HARDWARE_PRESETS as P1_HARDWARE, get_hardware as p1_get_hardware
from analyzer import (
    analyze as p1_analyze, AnalysisReport, OpEstimate,
    weights_bytes, weights_gb, kv_cache_bytes, kv_cache_gb,
)
import report as p1_report_mod
format_flops = p1_report_mod.format_flops
format_bytes = p1_report_mod.format_bytes
format_time = p1_report_mod.format_time

# ---------------------------------------------------------------------------
# Phase 2 imports — need to load its "passes" package under a unique name
# ---------------------------------------------------------------------------
# P2's code imports its passes as the top-level package "passes". Import P2
# with its directory first on sys.path, then evict "passes*" from
# sys.modules so P3 can load its own (different) "passes" package.
if _P2_DIR not in sys.path:
    sys.path.insert(0, _P2_DIR)

from ir import (
    ModelSpec, Hardware as P2Hardware, Workload as P2Workload, Options as P2Options,
    Context as P2Context, Candidate as P2Candidate, PassResult, Rejection,
    MODEL_PRESETS as P2_MODELS, HARDWARE_PRESETS as P2_HARDWARE,
    load_context, build_model as p2_build_model, build_hardware as p2_build_hardware,
    build_workload as p2_build_workload, GIB,
)
from cost_model import evaluate as p2_evaluate, Metrics as P2Metrics, kv_pressure
from planner import run_passes as p2_run_passes, build_plan as p2_build_plan, Plan as P2Plan

for _k in [k for k in sys.modules if k == "passes" or k.startswith("passes.")]:
    sys.modules["p2_" + _k] = sys.modules.pop(_k)
sys.path.remove(_P2_DIR)

# ---------------------------------------------------------------------------
# Phase 3 imports — its "passes" is separate from P2's
# ---------------------------------------------------------------------------
if _P3_DIR not in sys.path:
    sys.path.insert(0, _P3_DIR)

from p3.mock import build_ir as p3_build_ir, Workload as P3Workload, MODELS as P3_MODELS
from p3.gpu_specs import SPECS as P3_GPU_SPECS
from p3.irview import view as p3_view
import p3.irview as _p3_irview
import p3.mock as _p3_mock
# P1's common.Candidate has a different signature; P3 must use its own.
_p3_irview._Candidate = _p3_mock.Candidate
from passes import fusion as p3_fusion, schedule as p3_schedule

import tuner as p3_tuner


# ---------------------------------------------------------------------------
# Unified input schema
# ---------------------------------------------------------------------------
@dataclass
class PipelineInput:
    model: str = "llama-7b"
    hardware: str = "a100"
    batch_size: int = 1
    sequence_length: int = 2048
    generation_length: int = 256
    max_context: int = 0
    precision: str = "fp16"
    mode: str = "decode"
    pressure_threshold: float = 0.60
    block_size: int = 16
    chunk_size: int = 16
    system_prompt_tokens: int = 0
    fewshot_groups: int = 0
    fewshot_tokens: int = 0

    def __post_init__(self):
        if self.max_context <= 0:
            self.max_context = self.sequence_length + self.generation_length


# ---------------------------------------------------------------------------
# Unified output
# ---------------------------------------------------------------------------
@dataclass
class PassCandidate:
    phase: str
    pass_name: str
    name: str
    description: str
    memory_delta_gb: float
    latency_delta_ms: float
    accuracy_risk: str
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    selected: bool = False


@dataclass
class PipelineReport:
    input_config: Dict[str, Any]
    p1_parameter_count: int
    p1_weights_gb: float
    p1_kv_cache_gb: float
    p1_total_memory_gb: float
    p1_fits_in_memory: bool
    p1_total_flops: int
    p1_total_bytes: int
    p1_arithmetic_intensity: float
    p1_estimated_latency_ms: float
    p1_op_breakdown: List[Dict[str, Any]]
    p2_kv_pressure: float
    p2_baseline_metrics: Dict[str, Any]
    p2_optimized_metrics: Dict[str, Any]
    p2_selected_passes: List[str]
    p2_plan_notes: List[str]
    p3_fusion_best_level: Optional[int]
    p3_fusion_levels: List[Dict[str, Any]]
    p3_split_kv: Optional[Dict[str, Any]]
    p3_persistent: Optional[Dict[str, Any]]
    p3_autotune_chosen: Optional[Dict[str, Any]]
    p3_autotune_baseline: Optional[Dict[str, Any]]
    p3_autotune_top: List[Dict[str, Any]]
    candidates: List[Dict[str, Any]]
    summary: Dict[str, Any]


# ---------------------------------------------------------------------------
# Name mapping
# ---------------------------------------------------------------------------
_MODEL_MAP = {
    "llama-7b": ("llama-7b", "llama-7b", "LLaMA-7B"),
    "mistral-7b": ("mistral-7b", "mistral-7b", "Mistral-7B"),
}
_HW_MAP = {
    "a100": "A100", "h100": "H100", "t4": "T4", "rtx4090": "RTX4090",
}

def _norm_hw(name: str) -> str:
    return name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_p1(inp: PipelineInput) -> AnalysisReport:
    model = p1_get_model(inp.model)
    hardware = p1_get_hardware(inp.hardware)
    precision = normalize_precision(inp.precision)
    return p1_analyze(model, hardware, inp.sequence_length, inp.batch_size, precision)


def run_p2(inp: PipelineInput):
    p2_data = {
        "model": {"preset": inp.model},
        "hardware": {"device": inp.hardware},
        "workload": {
            "batch_size": inp.batch_size,
            "prompt_length": inp.sequence_length,
            "generation_length": inp.generation_length,
            "max_context": inp.max_context,
            "mode": inp.mode if inp.mode != "prefill" else "prefill",
            "precision": inp.precision,
        },
        "options": {
            "pressure_threshold": inp.pressure_threshold,
            "block_size": inp.block_size,
            "chunk_size": inp.chunk_size,
        },
    }
    if inp.system_prompt_tokens > 0 or inp.fewshot_groups > 0:
        p2_data["workload"]["prompt_spec"] = {
            "system_prompt_tokens": inp.system_prompt_tokens,
            "fewshot_groups": inp.fewshot_groups,
            "fewshot_tokens": inp.fewshot_tokens,
        }
    ctx = load_context(p2_data)
    results = p2_run_passes(ctx)
    plan = p2_build_plan(ctx, results)
    return ctx, results, plan


def run_p3(inp: PipelineInput) -> dict:
    model_name = _MODEL_MAP.get(inp.model.lower(), (None, None, "LLaMA-7B"))[2]
    hw_key = _norm_hw(inp.hardware)
    gpu_name = _HW_MAP.get(hw_key, "A100")

    mode = "prefill" if inp.mode == "prefill" else "decode"
    gen = inp.generation_length if mode == "decode" else 0
    ir = p3_build_ir(model_name, P3Workload(
        batch=inp.batch_size, prompt_len=inp.sequence_length,
        gen_len=gen, mode=mode,
        dtype_bytes=2 if inp.precision.lower() in ("fp16", "bf16") else 4,
    ), gpu_name)

    fusion_levels = p3_fusion.evaluate_levels(ir)
    fusion_candidates = p3_fusion.propose(ir)
    split_plan = p3_schedule.plan_split_kv(ir)
    persist_plan = p3_schedule.plan_persistent(ir)
    schedule_candidates = p3_schedule.propose(ir)
    tune_result = p3_tuner.tune(ir)
    tune_candidates = p3_tuner.propose(ir)

    return {
        "ir": ir,
        "fusion_levels": fusion_levels,
        "fusion_candidates": fusion_candidates,
        "split_plan": split_plan,
        "persist_plan": persist_plan,
        "schedule_candidates": schedule_candidates,
        "tune_result": tune_result,
        "tune_candidates": tune_candidates,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(inp: PipelineInput) -> PipelineReport:
    # ======================== Phase 1 ========================
    p1 = run_p1(inp)
    p1_op_breakdown = []
    for op_type in Op:
        ests = [e for e in p1.estimates if e.op.op is op_type]
        if not ests:
            continue
        flops = sum(e.op.flops for e in ests)
        nbytes = sum(e.op.bytes for e in ests)
        time_s = sum(e.estimated_time_s for e in ests)
        pct = 100 * time_s / p1.estimated_latency_s if p1.estimated_latency_s > 0 else 0
        p1_op_breakdown.append({
            "operation": op_type.value,
            "flops": flops,
            "flops_fmt": format_flops(flops),
            "bytes": nbytes,
            "bytes_fmt": format_bytes(nbytes),
            "arithmetic_intensity": round(flops / nbytes, 1) if nbytes else 0,
            "time_s": time_s,
            "time_fmt": format_time(time_s),
            "pct_time": round(pct, 1),
            "bound": ests[0].bound.value,
        })

    # ======================== Phase 2 ========================
    try:
        p2_ctx, p2_results, p2_plan = run_p2(inp)
        p2_pressure = kv_pressure(p2_ctx)
        p2_baseline = asdict(p2_plan.baseline)
        p2_optimized = asdict(p2_plan.optimized)
        p2_selected = p2_plan.selected
        p2_notes = p2_plan.notes
    except Exception as e:
        p2_pressure = 0.0
        p2_baseline = {}
        p2_optimized = {}
        p2_selected = []
        p2_notes = [f"P2 error: {e}"]
        p2_results = []
        p2_plan = None

    # ======================== Phase 3 ========================
    try:
        p3 = run_p3(inp)
        p3_fl = []
        best_level = None
        if p3["fusion_levels"]:
            best_lv = min(p3["fusion_levels"], key=lambda lv: lv.score_ms)
            best_level = best_lv.level
            for lv in p3["fusion_levels"]:
                p3_fl.append({
                    "level": lv.level,
                    "label": lv.label,
                    "launches_saved_per_layer": lv.launches_saved_per_layer,
                    "traffic_saved_bytes": lv.traffic_saved_bytes,
                    "unfused_ms": round(lv.unfused_ms, 4),
                    "fused_ms": round(lv.fused_ms, 4),
                    "mem_saved_gb": round(lv.mem_saved_bytes / 1e9, 4),
                    "delta_ms": round(lv.score_ms, 4),
                    "is_best": lv.level == best_level,
                })

        p3_split = None
        sp = p3["split_plan"]
        if sp is not None:
            p3_split = {
                "units": sp.units, "target": sp.target, "splits": sp.splits,
                "forced": sp.forced, "delta_ms": round(sp.score_ms, 4),
                "attention_unsplit_ms": round(sp.t_unsplit_ms, 4),
                "attention_split_ms": round(sp.t_split_ms, 4),
                "reduction_ms": round(sp.t_reduce_ms, 4),
            }

        p3_persist = None
        pp = p3["persist_plan"]
        if pp is not None:
            p3_persist = {
                "launches": pp.launches, "launch_share": round(pp.launch_share, 4),
                "delta_ms": round(pp.score_ms, 4),
                "sync_ms": round(pp.sync_ms, 4),
                "penalty_ms": round(pp.penalty_ms, 4),
            }

        tr = p3["tune_result"]
        p3_tune_chosen = None
        p3_tune_baseline = None
        if tr.chosen:
            p3_tune_chosen = {
                "config": tr.chosen.cfg.label(),
                "block_m": tr.chosen.cfg.block_m, "block_n": tr.chosen.cfg.block_n,
                "num_warps": tr.chosen.cfg.num_warps, "num_stages": tr.chosen.cfg.num_stages,
                "time_us_per_layer": round(tr.chosen.time_ms * 1000, 1),
                "bound": tr.chosen.bound,
                "smem_kb": round(tr.chosen.smem_kb, 1),
                "blocks_per_sm": tr.chosen.blocks_per_sm,
            }
        if tr.baseline and tr.baseline.feasible:
            p3_tune_baseline = {
                "config": tr.baseline_note,
                "time_us_per_layer": round(tr.baseline.time_ms * 1000, 1),
                "bound": tr.baseline.bound,
            }
        p3_tune_top = []
        for i, e in enumerate(tr.ranked[:10]):
            speedup = (tr.baseline.time_ms / e.time_ms) if (tr.baseline and tr.baseline.feasible) else 0
            p3_tune_top.append({
                "rank": i + 1,
                "config": e.cfg.label(),
                "time_us_per_layer": round(e.time_ms * 1000, 1),
                "bound": e.bound,
                "smem_kb": round(e.smem_kb, 1),
                "blocks_per_sm": e.blocks_per_sm,
                "speedup_vs_baseline": round(speedup, 2),
                "is_chosen": e is tr.chosen,
            })
    except Exception as e:
        p3_fl = []
        best_level = None
        p3_split = None
        p3_persist = None
        p3_tune_chosen = None
        p3_tune_baseline = None
        p3_tune_top = []
        p3 = {"fusion_candidates": [], "schedule_candidates": [], "tune_candidates": []}

    # ======================== Collect all candidates ========================
    all_candidates: List[PassCandidate] = []

    if p2_plan:
        for r in p2_results:
            for c in r.candidates:
                mem_delta = -c.memory_saved_bytes / 1e9
                lat_delta = c.latency_delta_pct
                all_candidates.append(PassCandidate(
                    phase="P2", pass_name=r.pass_name, name=c.name,
                    description=c.description,
                    memory_delta_gb=round(mem_delta, 4),
                    latency_delta_ms=round(lat_delta, 4),
                    accuracy_risk=c.accuracy_risk,
                    reason=c.rationale,
                    details=c.details,
                    selected=c.name in p2_selected,
                ))

    for c in p3.get("fusion_candidates", []):
        all_candidates.append(PassCandidate(
            phase="P3", pass_name="fusion", name=c.name,
            description=c.reason[:80],
            memory_delta_gb=round(c.d_mem_gb, 4),
            latency_delta_ms=round(c.d_latency_ms, 4),
            accuracy_risk=c.accuracy_risk, reason=c.reason,
            selected=c.d_latency_ms < 0,
        ))
    for c in p3.get("schedule_candidates", []):
        all_candidates.append(PassCandidate(
            phase="P3", pass_name="scheduling", name=c.name,
            description=c.reason[:80],
            memory_delta_gb=round(getattr(c, 'd_mem_gb', 0), 4),
            latency_delta_ms=round(c.d_latency_ms, 4),
            accuracy_risk=c.accuracy_risk, reason=c.reason,
            selected=c.d_latency_ms < 0,
        ))
    for c in p3.get("tune_candidates", []):
        all_candidates.append(PassCandidate(
            phase="P3", pass_name="autotune", name=c.name,
            description=c.reason[:80],
            memory_delta_gb=round(getattr(c, 'd_mem_gb', 0), 4),
            latency_delta_ms=round(c.d_latency_ms, 4),
            accuracy_risk=c.accuracy_risk, reason=c.reason,
            selected=c.d_latency_ms < 0,
        ))

    # ======================== Summary ========================
    total_mem_saved_gb = sum(
        abs(c.memory_delta_gb) for c in all_candidates if c.selected and c.memory_delta_gb < 0
    )
    total_lat_improvement_ms = sum(
        abs(c.latency_delta_ms) for c in all_candidates if c.selected and c.latency_delta_ms < 0
    )
    selected_names = [c.name for c in all_candidates if c.selected]

    summary = {
        "model": p1.model.name,
        "hardware": p1.hardware.name,
        "mode": inp.mode,
        "total_passes_run": 7,
        "total_candidates_proposed": len(all_candidates),
        "selected_optimizations": selected_names,
        "total_memory_saved_gb": round(total_mem_saved_gb, 3),
        "total_latency_saved_ms": round(total_lat_improvement_ms, 3),
        "baseline_latency_ms": round(p1.estimated_latency_s * 1000, 3),
        "baseline_memory_gb": round(p1.total_memory_bytes / 1e9, 3),
    }

    return PipelineReport(
        input_config=asdict(inp) if hasattr(inp, '__dataclass_fields__') else inp.__dict__,
        p1_parameter_count=p1.parameter_count,
        p1_weights_gb=round(p1.weights_bytes / 1e9, 3),
        p1_kv_cache_gb=round(p1.kv_cache_bytes / 1e9, 3),
        p1_total_memory_gb=round(p1.total_memory_bytes / 1e9, 3),
        p1_fits_in_memory=p1.fits,
        p1_total_flops=p1.total_flops,
        p1_total_bytes=p1.total_bytes,
        p1_arithmetic_intensity=round(p1.arithmetic_intensity, 1),
        p1_estimated_latency_ms=round(p1.estimated_latency_s * 1000, 3),
        p1_op_breakdown=p1_op_breakdown,
        p2_kv_pressure=round(p2_pressure, 4),
        p2_baseline_metrics=p2_baseline,
        p2_optimized_metrics=p2_optimized,
        p2_selected_passes=p2_selected,
        p2_plan_notes=p2_notes,
        p3_fusion_best_level=best_level,
        p3_fusion_levels=p3_fl,
        p3_split_kv=p3_split,
        p3_persistent=p3_persist,
        p3_autotune_chosen=p3_tune_chosen,
        p3_autotune_baseline=p3_tune_baseline,
        p3_autotune_top=p3_tune_top,
        candidates=[asdict(c) for c in all_candidates],
        summary=summary,
    )


if __name__ == "__main__":
    inp = PipelineInput(
        model="llama-7b", hardware="a100",
        batch_size=1, sequence_length=2048,
        generation_length=256, precision="fp16", mode="decode",
    )
    report = run_pipeline(inp)
    import json
    print(json.dumps(asdict(report), indent=2, default=str))
