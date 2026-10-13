"""P3 tests -- run from the repo root:   python -m pytest tests/test_p3.py -q"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import tuner
from p3 import perf
from p3.irview import view
from p3.mock import make_ir, scenario_ir
from passes import fusion, schedule

S1, S2, S3 = (scenario_ir(n)[1] for n in (1, 2, 3))
FIELDS = ("name", "d_latency_ms", "d_mem_gb", "d_launches", "accuracy_risk", "reason")


def all_candidates(ir):
    return fusion.propose(ir) + schedule.propose(ir) + tuner.propose(ir)


def by_name(ir, name):
    return next(c for c in all_candidates(ir) if c.name == name)


# ------------------------------------------------------------ interface
@pytest.mark.parametrize("ir", [S1, S2, S3])
def test_candidates_follow_shared_interface(ir):
    cands = all_candidates(ir)
    assert cands
    for c in cands:
        for f in FIELDS:
            assert hasattr(c, f), f
        assert isinstance(c.d_launches, int)
        assert isinstance(c.d_latency_ms, float) and math.isfinite(c.d_latency_ms)
        assert c.accuracy_risk in ("none", "low", "medium", "high")
        assert c.reason


def test_names_and_phase_coverage():
    assert {c.name for c in all_candidates(S1)} == {"fusion", "split_kv", "persistent_kernel", "autotune"}
    assert {c.name for c in all_candidates(S3)} == {"fusion", "autotune"}     # no decode scheduling


def test_deterministic():
    assert [vars(c) for c in all_candidates(S1)] == [vars(c) for c in all_candidates(S1)]


# --------------------------------------------------------------- fusion
def test_fusion_level_depends_on_workload():
    assert by_name(S1, "fusion").reason.startswith("L3")   # tiny batch: fuse everything
    assert by_name(S2, "fusion").reason.startswith("L2")   # big batch: OutProj atomics too costly
    assert by_name(S3, "fusion").reason.startswith("L1")   # prefill: projection GEMMs compute-bound


def test_fusion_launch_counts():
    L = view(S1).n_layers
    assert by_name(S1, "fusion").d_launches == -3 * L
    assert by_name(S2, "fusion").d_launches == -2 * L
    assert by_name(S3, "fusion").d_launches == -1 * L


def test_fusion_removes_prefill_score_buffers():
    v = view(S3)
    s_gb = v.batch * v.n_heads * v.q_len * v.kv_len * v.dtype_bytes / 1e9
    c = by_name(S3, "fusion")
    assert c.d_mem_gb == pytest.approx(-2 * s_gb, rel=1e-3)
    assert c.d_latency_ms < 0


# --------------------------------------------------------------- split_kv
def test_split_kv_helps_small_batch():
    c = by_name(S1, "split_kv")
    assert schedule.choose_num_splits(S1) > 1
    assert c.d_latency_ms < 0
    assert c.d_launches == view(S1).n_layers               # one reduction kernel per layer


def test_split_kv_rejected_when_gpu_already_full():
    c = by_name(S2, "split_kv")
    assert schedule.choose_num_splits(S2) == 1
    assert c.d_latency_ms > 0 and c.reason.startswith("not needed")


def test_split_kv_counts_kv_heads_for_gqa():
    llama = make_ir("LLaMA-7B", "A100", 1, 8192, 256, "decode")
    mistral = make_ir("Mistral-7B", "A100", 1, 8192, 256, "decode")     # 8 KV heads
    assert schedule.choose_num_splits(mistral) > schedule.choose_num_splits(llama)


# ------------------------------------------------------- persistent kernel
def test_persistent_kernel_flips_with_workload():
    v = view(S1)
    small, big = by_name(S1, "persistent_kernel"), by_name(S2, "persistent_kernel")
    assert small.d_latency_ms < 0 < big.d_latency_ms        # launch-bound vs memory/compute-bound
    assert small.d_launches == 1 - v.launches_per_step


# ------------------------------------------------------------------ tuner
@pytest.mark.parametrize("ir", [S1, S2, S3])
def test_tuner_ranking_is_sorted_and_feasible(ir):
    r = tuner.tune(ir)
    times = [e.time_ms for e in r.ranked]
    assert times == sorted(times)
    assert all(e.feasible for e in r.ranked) and r.chosen.feasible
    assert len(r.ranked) + len(r.infeasible) == r.total == 192


@pytest.mark.parametrize("ir", [S1, S2, S3])
def test_tuner_never_worse_than_hand_written(ir):
    assert by_name(ir, "autotune").d_latency_ms <= 0


def test_tuner_is_hardware_aware():
    big_tile = tuner.TileConfig(64, 128, 8, 3)             # ~208 KB shared memory
    a100 = view(make_ir("LLaMA-7B", "A100", 8, 2048, 0, "prefill"))
    h100 = view(make_ir("LLaMA-7B", "H100", 8, 2048, 0, "prefill"))
    assert not tuner.estimate(big_tile, a100).feasible      # 163 KB limit
    assert tuner.estimate(big_tile, h100).feasible          # 227 KB limit
    t4 = view(S3)
    assert not tuner.estimate(tuner.DEFAULT, t4).feasible   # 80 KB > 64 KB on T4
    assert tuner.tune(S3).chosen.cfg != tuner.tune(make_ir("Mistral-7B", "A100", 2, 4096, 0, "prefill")).chosen.cfg


# ------------------------------------------------------ integration switch
def test_launch_overhead_switch():
    v = view(S1)
    with_launch = by_name(S1, "persistent_kernel")
    perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY = False
    try:
        without = by_name(S1, "persistent_kernel")
    finally:
        perf.COUNT_LAUNCH_OVERHEAD_IN_LATENCY = True
    assert without.d_latency_ms > 0                         # only the costs remain
    assert without.d_latency_ms + without.d_launches * v.launch_ms == pytest.approx(with_launch.d_latency_ms, abs=1e-3)


# ------------------------------------------------------------ IR adapter
def _p1_style_dict(ir):
    """Same IR, but shaped the way P1 might write it: dicts, HF-style names, string units."""
    m, w = ir.model, ir.workload
    return {
        "model": {"num_layers": m.n_layers, "hidden_size": m.hidden, "num_attention_heads": m.n_heads,
                  "num_key_value_heads": m.n_kv_heads, "head_dim": m.head_dim},
        "workload": {"batch_size": w.batch, "prompt_length": w.prompt_len,
                     "generation_length": w.gen_len, "mode": w.mode, "precision": "FP16"},
        "hardware": {"device": "NVIDIA A100", "gpu_memory": "80GB", "memory_bandwidth": "1935 GB/s"},
        "ops": [{"name": f"L{op.layer}.{op.name}", "layer": op.layer, "flops": op.flops, "bytes": op.bytes}
                for op in ir.ops],
    }


@pytest.mark.parametrize("ir", [S1, S2])
def test_adapter_accepts_differently_shaped_ir(ir):
    a = [(c.name, c.d_latency_ms, c.d_launches) for c in all_candidates(ir)]
    b = [(c.name, c.d_latency_ms, c.d_launches) for c in all_candidates(_p1_style_dict(ir))]
    assert a == b


def test_adapter_reports_missing_hardware_fields():
    bad = _p1_style_dict(S1)
    bad["hardware"] = {"device": "Mystery GPU 9000"}
    with pytest.raises(ValueError, match="hardware fields"):
        view(bad)
