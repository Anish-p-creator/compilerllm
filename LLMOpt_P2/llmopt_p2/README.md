# LLMOpt — P2: KV-cache + Prefix-sharing passes

Part of *LLMOpt: Compiler Optimization Framework for LLM Inference*. This repo is the **P2 slice**:
the two optimization passes plus a small runnable pipeline (input → passes → cost-model search → report with metrics), so the work can be demoed on its own.

| File | Role |
|---|---|
| `passes/kv_cache.py` | **paged KV** and **INT8 KV** candidates, gated on memory pressure |
| `passes/prefix_share.py` | **radix-trie** prefix finder + **shared-prefix** candidate (GB and compute saved) |
| `ir.py` | model / workload / hardware types, input loader, `Candidate`/`Rejection`/`PassResult` |
| `cost_model.py` | roofline-style estimates (memory, TPOT, TTFT, throughput) |
| `planner.py` | stand-in for the team cost model: tries every subset of candidates, picks the lowest cost |
| `report.py`, `main.py` | console/JSON report and CLI |
| `samples/` | 3 ready-to-run inputs |
| `tests/test_p2.py` | 18 unit tests |

## 1. Run it

Requires Python 3.9+ and numpy (`pip install -r requirements.txt`). No GPU needed.

```bash
python main.py                                        # runs samples/high_pressure.json
python main.py --input samples/high_pressure.json     # explicit input
python main.py --input samples/high_pressure.json --json report.json   # also save JSON
python main.py --all                                  # run all three samples
python -m unittest discover -s tests -v               # run the tests
```

## 2. Input

One JSON file with `model`, `hardware`, `workload` (field names follow the PRD) and optional `options`.

```json
{
  "model": {"preset": "llama-7b"},
  "hardware": {"device": "NVIDIA A100", "gpu_memory": "80GB", "memory_bandwidth": "1935 GB/s"},
  "workload": {
    "batch_size": 32,
    "prompt_length": 2048,
    "generation_length": 512,
    "max_context": 4096,
    "mode": "decode",
    "precision": "FP16",
    "prompt_spec": {"system_prompt_tokens": 1024, "fewshot_groups": 2, "fewshot_tokens": 256}
  }
}
```

| Field | Meaning |
|---|---|
| `model` | preset name (`llama-7b`, `llama-13b`, `llama-3-8b`, `mistral-7b`) or an object with `layers, hidden, heads, kv_heads, head_dim, params_b` (a `preset` plus overrides also works) |
| `hardware` | `device` (A100 / A100-40GB / H100 / L4 / RTX 4090 presets) with optional overrides `gpu_memory`, `memory_bandwidth`, `fp16_tflops`; unknown devices need memory + bandwidth |
| `workload.mode` | `decode`, `prefill` or `mixed` (affects the latency effect of INT8 and the planner weights) |
| `workload.max_context` | what a contiguous allocator reserves per request (must be >= prompt + generation) |
| `workload.prompts` | optional list of prompts, as strings (regex word tokenizer) **or** lists of token ids |
| `workload.prompt_spec` | optional synthetic prompts: `system_prompt_tokens` shared by all, `fewshot_groups` × `fewshot_tokens` shared per group, unique tail up to `prompt_length` |
| `options` | `pressure_threshold` (0.60), `block_size` (16), `chunk_size` (16), `paged_overhead_pct` (3), `int8_dequant_overhead_pct` (2), `usable_memory_fraction` (0.92), `objective` (`latency`/`memory`/`ttft` weights) |

If neither `prompts` nor `prompt_spec` is given, prefix sharing is rejected with a reason (paged/INT8 do not need prompts).

## 3. What the passes do

**KV-cache pass** (`passes/kv_cache.py`)
1. Memory pressure = reserved KV (`batch × max_context × KV bytes/token`) ÷ GPU memory. Paged and INT8 are proposed **only if this is > 60%**; otherwise both are listed as rejected with the reason.
2. `paged_kv`: allocates 16-token blocks for what is actually used (prompt + generation) instead of `max_context`. Saves the unused-context memory; costs **+3% attention latency** (configurable).
3. `int8_kv`: INT8 values + one fp16 scale per token/head vector = 1.016 B/element vs 2 B, i.e. **50.8% of FP16 size** (the scales are why it is not exactly 50%). Decode attention is shown to be memory-bound (about 1–4 FLOP/byte vs a ridge point of 80–160), so reading half the bytes makes it faster. Accuracy is **measured**: real attention is run with FP32 vs INT8 K/V and the output error reported. A candidate is dropped if the error is in the "high" band (>= 3%).

**Prefix-sharing pass** (`passes/prefix_share.py`)
1. All prompts go into a compressed radix trie (`RadixTrie`) over token ids.
2. It reports the longest prefix shared by >= 2 prompts, the prefix common to *all* prompts, the most valuable shared segments, and how many tokens are stored once instead of N times.
3. Reuse is counted the way a prefix cache behaves: prompt *i* reuses what prompts *0..i-1* already cached, rounded down to `chunk_size`-token chunks (conservative).
4. The candidate reports **memory saved (GiB of KV)** and **compute saved** (prefill tokens skipped, TFLOP, % of prefill). Unlike the KV passes it is not gated on memory pressure: it saves prefill compute even when memory is plentiful.

**Planner** (stand-in for the team cost model): evaluates all 2^n subsets of proposed candidates with `cost = w_lat·TPOT/TPOT₀ + w_mem·Mem/Mem₀ + w_ttft·TTFT/TTFT₀`, prefers plans that fit in usable GPU memory, and picks the lowest cost. The search table is printed so you can show *why* a combination won.

## 4. Sample output (`samples/high_pressure.json`)

```text
>>> input: samples/high_pressure.json
==============================================================================
                LLMOpt · P2 · KV-cache + prefix-sharing passes                
==============================================================================
Model     : llama-7b  (32 layers, 32 KV heads, head_dim 128, FP16)
Hardware  : NVIDIA A100-80GB  (80 GiB, 1935 GB/s)
Workload  : batch 32 | prompt 2048 | generate 512 | max_ctx 4096 | mode decode

1. KV-CACHE ANALYSIS
------------------------------------------------------------------------------
  KV size per token        : 512 KiB
  Reserved KV (baseline)   : 64.00 GiB = 80.0% of GPU memory [HIGH pressure, threshold 60%]

2. CANDIDATES PROPOSED
------------------------------------------------------------------------------
  [1] paged_kv     paged KV cache, 16-token blocks
      memory saved   : 24.00 GiB (stand-alone)
      attention lat. : +3.0%    accuracy risk: none
      utilization    : 62% of reserved slots are ever used
      why            : contiguous allocation reserves 4096 tokens/request but only 2560 are ever used (62% utilization)
  [2] int8_kv      INT8 KV cache (per-token symmetric, fp16 scales)
      memory saved   : 31.50 GiB (stand-alone)
      attention lat. : -48.2%    accuracy risk: low
      measured error : 0.643% rel. attention-output error (worst of sweep 1.74%), cosine 0.99998
                       data: synthetic heads
      size           : 50.8% of FP16 (1.016 B/elem incl. scales)
      why            : decode attention is memory-bound (1.0 FLOP/B << ridge 161 FLOP/B), so reading half the bytes cuts attention time
  [3] prefix_share shared-prefix KV cache (16-token chunks, radix trie)
      memory saved   : 19.25 GiB (stand-alone)
      compute saved  : 39,424 prefill tokens skipped, 544.6 TFLOP (59.3% of prefill)
      longest prefix : 1280 tokens shared by 16/32 prompts; common to ALL: 1024
      best segment   : tokens [0, 1024) shared by 32 prompts -> 31,744 tokens deduplicated
      starts with    : "1 2 3 4 5 6 7 8 ..."
      trie           : 65,536 prompt tokens -> 26,112 stored (synthetic prompt_spec)
      why            : longest shared prefix = 1280 tokens (shared by 16 of 32 prompts); 60.2% of prompt tokens never need recomputing or re-storing

3. COST-MODEL SEARCH  (cost = 1·TPOT + 1·Mem + 0.25·TTFT, normalized)
------------------------------------------------------------------------------
  cost 1.058  fits  int8_kv+paged_kv+prefix_share           ◀ selected
  cost 1.205  fits  int8_kv+prefix_share                    
  cost 1.333  fits  int8_kv+paged_kv                        
  cost 1.481  fits  int8_kv                                 
  cost 1.559  fits  paged_kv+prefix_share                   
  cost 1.850  fits  prefix_share                            
  cost 1.959  fits  paged_kv                                
  cost 2.250  OOM   baseline                                

4. RESULT
------------------------------------------------------------------------------
  Selected: int8_kv + paged_kv + prefix_share
  ! baseline does NOT fit in usable GPU memory; only fitting plans were considered

  Metric                              Baseline     Optimized        Change
  ----------------------------------------------------------------------
  KV cache (GiB)                         64.00         10.54       ↓ 83.5%
  Peak GPU memory (GiB)                  76.55         23.09       ↓ 69.8%
  Fits in usable memory               NO (OOM)           yes              
  Max concurrent requests                   30           185      ↑ 516.7%
  KV read / decode step (GiB)            36.00         18.28       ↓ 49.2%
  TPOT (ms)                              35.92         23.50       ↓ 34.6%
  Throughput (tok/s)                       891         1,362       ↑ 52.9%
  TTFT, batch prefill (ms)               5,889         2,398       ↓ 59.3%
  Prefill tokens computed               65,536        26,112       ↓ 60.2%
  Attention-output error (INT8)              0        0.643%      measured

  Memory/latency/TTFT are analytical roofline estimates; trie statistics and the
  INT8 error are computed (not estimated). See README for assumptions.
==============================================================================
JSON report written to /tmp/hp.json
```

### All three samples at a glance

| Sample | Situation | What the passes decide |
|---|---|---|
| `high_pressure.json` | Llama-7B, A100-80GB, batch 32, max ctx 4096 (KV = 80% of GPU) | all three selected; baseline does not fit, optimized does |
| `consumer_gpu.json` | Llama-3-8B (GQA), RTX 4090 24 GiB, ctx 8192 | all three selected; baseline OOM → fits, 7 → 55 concurrent requests |
| `low_pressure.json` | Mistral-7B, H100, batch 8, **text** prompts (KV = 2.5%) | paged/INT8 **rejected** (low pressure); only prefix sharing proposed |

## 5. Metrics reported

| Metric | How it is obtained |
|---|---|
| KV cache GiB, peak GPU memory (weights + KV) | exact arithmetic from model shape and allocator rules |
| Max concurrent requests | (usable memory − weights) ÷ KV per request |
| KV read per decode step, TPOT, throughput | roofline estimate: bytes moved ÷ (peak bandwidth × 0.75) |
| TTFT (whole batch prefill) | FLOPs ÷ (peak FP16 FLOPs × 0.5); prefix sharing removes the reused tokens' FLOPs |
| Prefill tokens computed | exact, from the trie |
| Longest shared prefix, tokens stored vs total | exact, from the trie |
| INT8 attention-output error | computed by running attention in FP32 vs INT8 K/V |

## 6. Honest limitations (say these before someone asks)

- **Latency/TTFT/throughput are analytical estimates, not GPU measurements.** The memory numbers and the trie statistics are exact; the timing uses a roofline model with efficiency factors in `options`. Validating against a real GPU run (vLLM/Triton kernels) is the natural next step.
- **INT8 error is from a synthetic head** (outlier channels, shared V component) unless real tensors are passed to `simulate_int8_error(head_dim, K=, V=, Q=)`. The headline is 0.64%; the worst case across the query-scale sweep is 1.74%. Use real K/V from a model for a final accuracy claim.
- With **sample prompts** (`prompts`), the reuse ratio measured on the samples is *extrapolated* to the workload's `batch × prompt_length` (the report says so). With `prompt_spec` the numbers are exact for the workload.
- Prefix sharing assumes chunked KV storage so shared prefix chunks can be physically reused, and counts its saving against the KV that would otherwise be materialized.
- The 3% paged overhead and 2% INT8 dequant overhead are assumptions taken from the task spec/typical values; both are configurable.
- Activations and CUDA context are not modelled; `usable_memory_fraction` (0.92) stands in for them.

## 7. Plugging into the full compiler

Each pass is `run(ctx) -> PassResult` returning `Candidate`s (memory saved, latency delta %, accuracy risk, compute saved) and `Rejection`s (with reasons). Teammates' passes (fusion, scheduling, auto-tuning) can be appended to `ALL_PASSES` in `passes/__init__.py`; `planner.py` can be replaced by the real cost model, which only needs the `Candidate` fields.

## 8. Suggested 3-minute demo

1. `python main.py --input samples/high_pressure.json` — walk the report top to bottom: pressure 80% → three candidates → search table → metrics.
2. Point at the trie lines: 65,536 prompt tokens stored as 26,112; system prompt of 1,024 tokens shared by all 32.
3. `python main.py --input samples/low_pressure.json` — same code, low pressure, so paged/INT8 are **rejected with reasons** and only prefix sharing is proposed. This shows the pass is selective, not "always optimize".
4. Mention the tests (`python -m unittest discover -s tests`) and the limitations above.

**Likely questions:** *Why only above 60%?* Under low pressure paging adds 3% latency for memory nobody needs. *Why not exactly 50% for INT8?* Per-vector fp16 scales add 1.6%. *Is INT8 lossless?* No; measured error is reported, and a high-error candidate is rejected. *Is the speedup real?* It is a roofline estimate; decode attention is memory-bound so bytes read is the right first-order driver.

