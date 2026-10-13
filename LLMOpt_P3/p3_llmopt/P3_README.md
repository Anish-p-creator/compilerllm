# P3: Fusion + Scheduling + Auto-tuner (LLMOpt)

Owned files from the team split: `passes/fusion.py`, `passes/schedule.py`, `tuner.py`.
P3-only support code lives in `p3/` and `tests/test_p3.py`.
Needs plain Python 3.9+ (pytest only for tests). No GPU required.

## Run it

```
python -m p3.demo                         # all 3 scenarios, every P3 component
python -m p3.demo --scenario 3            # one scenario
python -m p3.demo --sweep --scenario 3    # auto-tuner on T4 / RTX4090 / A100 / H100
python -m p3.demo --model Mistral-7B --gpu A100 --batch 1 --prompt 8192 --mode decode
python -m p3.demo --json                  # exactly what P4 receives
python tuner.py --scenario 3              # ranked tuner table only
python -m pytest tests/test_p3.py -q      # 23 tests
```

Every pass follows the agreed interface `propose(ir) -> list[Candidate]`, where Candidate is
`(name, d_latency_ms, d_mem_gb, d_launches, accuracy_risk, reason)`.

| Candidate | Source | Phase | d_launches | Accuracy risk |
|---|---|---|---|---|
| `fusion` | best of 3 levels: Attn+Softmax / +QKV / +OutProj | both | -1 / -2 / -3 per layer | none |
| `split_kv` | Flash-Decoding style KV splitting | decode | +1 per layer (reduction) | none |
| `persistent_kernel` | one launch per decode step (PersistentKV) | decode | 1 - baseline | none |
| `autotune` | best tile/warps/stages for the fused attention kernel | both | 0 | none |

## Results snapshot (model estimates, not measurements)

| Scenario | fusion | split_kv | persistent_kernel | autotune |
|---|---|---|---|---|
| 1. LLaMA-7B, A100, batch 1, decode | L3, -0.37 ms/step | -1.56 ms/step (32 -> 160 blocks) | -0.33 ms/step (launches = 8.4% of step) | keep default (memory-bound) |
| 2. Mistral-7B, A100, batch 256, decode | L2, -1.98 ms/step | +0.41, reject (GPU already full) | +1.98, reject (launches = 1.9% of step) | keep default |
| 3. Mistral-7B, T4, batch 2, 4K prefill | L1, -603 ms and -4.3 GB (only plan that fits in 16 GB) | n/a | n/a | -135 ms, 1.86x (default needs 80 KB shared memory, T4 has 64 KB) |

The same transform is selected in one workload and rejected in another, and the tuner
picks a different configuration on each GPU. That is the "workload- and hardware-aware
compiler" claim from the PRD.

## How the estimates work

Everything uses the roofline P4 uses: `time = max(compute, memory)`. Each module's docstring
explains its model in full.

- **fusion:** fusing Attention+Softmax removes about 4|S| bytes of score-matrix traffic and the S and P buffers (2|S|).
  Fusing the QKV projection or the output projection (OutProj) as well costs 25% GEMM efficiency, which only matters when that GEMM is compute-bound.
  Fusing OutProj additionally needs FP32 atomics across heads.
  The fusion pass costs the fused kernel conservatively, using the IR's FLOPs with no credit for causal-mask skipping.
- **split_kv:** decode attention has `batch x KV heads` blocks.
  Achieved bandwidth grows linearly with blocks up to `2 x SMs`, then flattens.
  Splitting adds a log-sum-exp merge kernel.
- **persistent_kernel:** saves `(launches - 1) x launch overhead`.
  It costs a 1 us sync between ops, plus 2% bandwidth and 15% tensor-core efficiency, because one generic kernel can't use each op's best configuration.
- **tuner:** 192 configs, each checked against:
  - shared-memory and register limits
  - occupancy and wave quantization
  - 16 rows per warp for the MMA instruction
  - CUDA-core softmax and rescale work
  - K/V reloads with L2 reuse (prefill)
  - Little's-law latency hiding
  - 1-stage vs pipelined execution

  If the default config doesn't fit the GPU, it is shrunk the way a developer would and used as the baseline.

All assumptions are named constants at the top of each file, so each one can be explained or changed.

## Integration notes (for P1 / P4)

1. **IR:** P3 reads the IR only through `p3/irview.py`, which accepts dataclasses or dicts and common field-name variants. Unknown names are fixed there and nowhere else. Hardware fields P1 doesn't provide (shared memory, registers, L2) come from `p3/gpu_specs.py`.
2. **Candidate:** imported from P1's `common.py` automatically once it exists. Until then `p3/mock.py` provides a stand-in, along with a mock `build_ir` for the demo and tests.
3. **Launch overhead:** set `COUNT_LAUNCH_OVERHEAD_IN_LATENCY` in `p3/perf.py`.
   - `True` (default): `d_latency_ms` already includes the launch saving.
   - `False`: use this if P4's cost model charges launch overhead from the launch count itself.
   - `python -m p3.demo --launch-in-p4` previews the `False` setting.
4. **Launch floor:** `persistent_kernel.d_launches` is relative to the unoptimized IR. Stacked with `fusion`, the sum can drop below 1, so P4 should clamp the final launch count at a minimum of 1.
5. **Fusion levels:** they are mutually exclusive and the contract has no exclusion field, so `fusion` emits only its best level. The other levels are compared in its `reason`.
6. **Reporting:** label P3's numbers as estimates in the report. P4's benchmark is the measured data.
