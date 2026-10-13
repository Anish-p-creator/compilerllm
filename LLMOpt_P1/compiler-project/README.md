# P1: Hardware-Aware IR + Analyzer for LLM Inference

This is part P1 of a compiler-optimization / performance-analysis project for
Large Language Models. It does four things:

1. Converts a transformer LLM configuration (LLaMA-7B, Mistral-7B) into an
   **Intermediate Representation (IR)**: a list of operations, five per layer.
2. Estimates the **FLOPs** and **memory traffic (bytes)** of every operation.
3. Computes **weight memory**, **KV-cache memory**, and whether everything
   **fits** on a given GPU (A100, H100, T4, RTX 4090).
4. Uses a **roofline model** to estimate latency and classify each op as
   compute-bound or memory-bound.

It uses only the Python standard library (Python 3.10+).

---

## Quick start

```bash
# run the unit tests
python3 -m unittest discover -s tests -v

# analyze LLaMA-7B on an A100
python3 main.py --model llama-7b --hardware a100 --seq-len 2048 --batch-size 1 --precision fp16

# analyze Mistral-7B on a T4, only print the IR of the first 2 layers
python3 main.py --model mistral-7b --hardware t4 --seq-len 4096 --max-layers 2

# other examples
python3 main.py --model mistral-7b --hardware h100 --seq-len 4096
python3 main.py --model llama-7b --hardware rtx4090 --precision int4 --batch-size 4
python3 main.py --help
```

| Option         | Values                                 | Default    |
|----------------|----------------------------------------|------------|
| `--model`      | `llama-7b`, `mistral-7b`               | `llama-7b` |
| `--hardware`   | `a100`, `h100`, `t4`, `rtx4090`        | `a100`     |
| `--seq-len`    | positive integer                       | `2048`     |
| `--batch-size` | positive integer                       | `1`        |
| `--precision`  | `fp32`, `fp16`, `bf16`, `int8`, `int4` | `fp16`     |
| `--max-layers` | print only the first N layers' IR      | all        |

Invalid names print an error with the valid options and exit with code 2.

---

## Project structure

```text
compiler-project/
├── common.py          # shared types: Op, Bound, IROp, Candidate, ModelConfig,
│                      #   HardwareConfig, precision -> bytes helpers
├── hardware.py        # GPU presets (A100, H100, T4, RTX 4090) + get_hardware()
├── models.py          # model presets (LLaMA-7B, Mistral-7B) + count_parameters()
├── ir_builder.py      # build_ir(): FLOP and byte formulas for every op
├── analyzer.py        # weights / KV cache memory, fit check, roofline, analyze()
├── report.py          # pretty-printed tables and summary
├── main.py            # command-line interface (argparse)
├── tests/
│   └── test_analyzer.py   # 28 unit tests that check the formulas
└── README.md
```

Dependency direction (no cycles): `common` ← `hardware`, `models` ← `ir_builder`
← `analyzer` ← `report` ← `main`.

---

## Background concepts

### What is an IR?
An **Intermediate Representation** is the compiler's internal description of a
program, sitting between the source (here: a model config) and the hardware.
Our IR is a flat list of `IROp(layer, op, flops, bytes)` nodes in execution
order. Optimization passes (P2+) can transform this list, e.g. fuse
`Attention + Softmax` into one op, and the analyzer can re-score it. The
`Candidate` dataclass holds one such version of the IR plus its estimated
latency, so different optimized versions can be compared.

### Why count FLOPs?
FLOPs (floating-point operations) measure the **amount of math**. Divided by
the GPU's peak FLOP/s, they give a lower bound on time if the GPU were
limited only by its arithmetic units.

### Why count memory traffic?
Every operation must read its inputs and weights from GPU memory (HBM/GDDR)
and write its outputs back. Bytes divided by memory bandwidth gives a lower
bound on time if the GPU were limited only by memory. For LLMs this is often
the real bottleneck.

### Arithmetic intensity (AI)

```text
AI = FLOPs / bytes        [FLOP/byte]
```

High AI means much math per byte loaded (e.g. big matmuls); low AI means
little math per byte (e.g. softmax, elementwise ops).

### Roofline model, compute-bound vs memory-bound

```text
compute_time   = FLOPs / peak_FLOP_per_s
memory_time    = bytes / bandwidth_bytes_per_s
estimated_time = max(compute_time, memory_time)

ridge_point    = peak_FLOP_per_s / bandwidth_bytes_per_s      [FLOP/byte]
```

* `AI ≥ ridge point` → compute_time dominates → **COMPUTE_BOUND**
  (a faster GPU core would help).
* `AI < ridge point` → memory_time dominates → **MEMORY_BOUND**
  (less data movement would help, e.g. fusion or quantization).

The A100's ridge point is 312e12 / 2039e9 ≈ **153 FLOP/B**. A QKV projection
(AI ≈ 1229) is compute-bound; softmax (AI ≈ 1.2) is heavily memory-bound.

Total latency = sum of the per-op estimated times (ops run one after another).

### KV cache
During generation, each new token attends to all previous tokens. To avoid
recomputing their keys and values, every layer **caches K and V** for every
token seen so far. That cache grows linearly with sequence length and batch
size, and it must sit in GPU memory alongside the weights.

### Why GQA (Mistral) shrinks the KV cache
In standard Multi-Head Attention (LLaMA-7B) each of the 32 query heads has
its own K and V head (`h_kv = 32`). In **Grouped Query Attention** (Mistral-7B)
groups of 4 query heads share one K/V head, so `h_kv = 8`. The KV cache and
the K/V projection weights/FLOPs therefore shrink by `h / h_kv = 4×`.
The attention FLOPs (QKᵀ, PV) do *not* shrink, because every query head still
does its own matmuls.

---

## Formulas

### Notation

| Symbol | Meaning                          | LLaMA-7B | Mistral-7B |
|--------|----------------------------------|---------:|-----------:|
| L      | layers                           | 32       | 32         |
| d      | hidden size                      | 4096     | 4096       |
| h      | query heads                      | 32       | 32         |
| h_kv   | key/value heads                  | 32       | 8          |
| d_h    | head dim = d / h                 | 128      | 128        |
| d_kv   | K/V width = h_kv · d_h           | 4096     | 1024       |
| f      | FFN intermediate size            | 11008    | 14336      |
| V      | vocabulary size                  | 32000    | 32000      |
| B, S   | batch size, sequence length      |          |            |
| T      | tokens = B · S                   |          |            |
| b      | bytes per element (fp32 4, fp16/bf16 2, int8 1, int4 0.5) | | |

Matrix multiply `(M×K)·(K×N)`: each of M·N outputs needs K multiplies and K
adds, so **FLOPs = 2·M·K·N**.

### FLOPs per layer

| Op        | Formula                                    | Derivation |
|-----------|--------------------------------------------|------------|
| Q proj    | `2·T·d·d`                                  | (T×d)·(d×d) |
| K proj    | `2·T·d·d_kv`                               | (T×d)·(d×d_kv), smaller with GQA |
| V proj    | `2·T·d·d_kv`                               | same as K |
| **QKV**   | **`2·T·d·(d + 2·d_kv)`**                   | sum of the three |
| QKᵀ       | `2·B·h·S²·d_h`                             | per head (S×d_h)·(d_h×S) |
| P·V       | `2·B·h·S²·d_h`                             | per head (S×S)·(S×d_h) |
| **Attention** | **`4·B·h·S²·d_h`**                     | QKᵀ + P·V |
| **Softmax**   | **`5·B·h·S²`**                         | ≈5 FLOPs per score: max, subtract, exp, sum, divide |
| **OutProj**   | **`2·T·d²`**                           | (T×d)·(d×d) |
| **MLP (SwiGLU)** | **`3 · 2·T·d·f = 6·T·d·f`**         | gate, up: (T×d)·(d×f); down: (T×f)·(f×d) |

Sanity check: QKV + OutProj + MLP = `2 · T · (weights per layer)`, the well
known "2 FLOPs per parameter per token" rule (verified in the tests).

### Bytes per layer (multiply element counts by b)

| Op        | Elements moved |
|-----------|----------------|
| QKV       | `T·d` (input) + `d·(d+2d_kv)` (weights) + `T·(d+2d_kv)` (Q,K,V out) |
| Attention | `T·d` (Q) + `2·T·d_kv` (K,V) + `2·B·h·S²` (write scores, read probs) + `T·d` (out) |
| Softmax   | `2·B·h·S²` (read scores, write probs) |
| OutProj   | `T·d` + `d²` + `T·d` |
| MLP       | `T·d` + `3·d·f` + `T·d` + `2·T·f` (hidden written once, read once) |

### Weight memory

```text
P = V·d                            (embedding)
  + L · (d·(2d + 2d_kv)            (W_q, W_o, W_k, W_v)
         + 3·d·f                   (W_gate, W_up, W_down)
         + 2·d)                    (two RMSNorm weights)
  + d                              (final RMSNorm)
  + V·d                            (LM head, not tied)

weights_bytes = P · b
weights_GB    = weights_bytes / 10^9
```

LLaMA-7B: P = 6,738,415,616 → 13.48 GB in FP16.
Mistral-7B: P = 7,241,732,096 → 14.48 GB in FP16.
(The parameter count is computed from the architecture; the published value
is kept only to sanity-check it, and they agree within 0.1%.)

### KV cache

```text
KV_bytes = 2 · L · B · S · h_kv · d_h · b
           ^ K and V
```

| S = 4096, B = 1, FP16 | KV cache |
|-----------------------|---------:|
| LLaMA-7B (h_kv = 32)  | 2.15 GB  |
| Mistral-7B (h_kv = 8) | 0.54 GB (4× smaller) |

### Fit check

```text
total_memory = weights_bytes + KV_bytes
fits         = total_memory ≤ GPU memory (GB · 10^9)
```

---

## Hardware presets

All compute numbers use one basis: **dense FP16 Tensor Core throughput, FP16
inputs with FP32 accumulate, without sparsity** (vendor "with sparsity"
numbers are 2× higher and are *not* used).

| Key       | GPU             | Memory | Bandwidth  | Dense FP16 TC | Ridge point |
|-----------|-----------------|-------:|-----------:|--------------:|------------:|
| `a100`    | A100 80GB SXM   | 80 GB  | 2039 GB/s  | 312 TFLOP/s   | 153 FLOP/B  |
| `h100`    | H100 SXM5 80GB  | 80 GB  | 3350 GB/s  | 989.4 TFLOP/s | 295 FLOP/B  |
| `t4`      | T4              | 16 GB  | 320 GB/s   | 65 TFLOP/s    | 203 FLOP/B  |
| `rtx4090` | RTX 4090        | 24 GB  | 1008 GB/s  | 165.2 TFLOP/s | 164 FLOP/B  |

Sources: NVIDIA datasheets (A100, H100, T4) and the Ada Lovelace architecture
whitepaper (RTX 4090; 165.2 is the FP32-accumulate rate, 330 is FP16-accumulate).
To add a GPU, add one `HardwareConfig` entry to `HARDWARE_PRESETS` in
`hardware.py`. To add a model, add a `ModelConfig` to `MODEL_PRESETS` in `models.py`.

---

## Example output (abridged)

```text
$ python3 main.py --model llama-7b --hardware a100 --seq-len 2048 --batch-size 1 --precision fp16
Layer | Operation | FLOPs         | Bytes     | AI (FLOP/B) | Est. Time  | Bound
--------------------------------------------------------------------------------
0     | QKV       | 206.16 GFLOPs | 167.77 MB | 1228.8      | 660.764 us | COMPUTE_BOUND
0     | Attention | 68.72 GFLOPs  | 603.98 MB | 113.8       | 296.214 us | MEMORY_BOUND
0     | Softmax   | 671.09 MFLOPs | 536.87 MB | 1.2         | 263.301 us | MEMORY_BOUND
0     | OutProj   | 68.72 GFLOPs  | 67.11 MB  | 1024.0      | 220.255 us | COMPUTE_BOUND
0     | MLP       | 554.05 GFLOPs | 394.26 MB | 1405.3      | 1.776 ms   | COMPUTE_BOUND
...
Weights: 13.48 GB
KV Cache: 1.07 GB
Total Memory (weights + KV): 14.55 GB
Fits in GPU Memory: YES
Total FLOPs: 28.75 TFLOPs
Total Estimated Traffic: 56.64 GB
Estimated Latency (prefill, roofline): 102.923 ms
```

Observation: attention + softmax do only ~8% of the FLOPs but take ~17% of
the estimated time, because the S×S score matrix goes through memory twice.
That is exactly the problem that **operator fusion / FlashAttention** solves,
which makes it a natural first optimization pass for P2.

---

## Assumptions

1. **Prefill only.** The IR models one forward pass over the whole prompt
   (S tokens at once). Token-by-token decoding is not modelled yet.
2. **Unfused kernels.** Each op reads its inputs and weights from memory and
   writes its outputs back; weights are read exactly once per op (perfect
   on-chip reuse). The attention score matrix (B·h·S²) is materialized in memory.
3. **One precision for all tensors.** Weights, activations and KV cache all
   use `--precision`. INT4 is handled exactly as 4 bits (0.5 byte) using
   integer bit arithmetic.
4. **One compute roof.** Compute time always uses the dense FP16 Tensor Core
   rate, whatever the precision. For fp32 this is optimistic, and for
   int8/int4 it models weight-only quantization where the math runs in FP16.
5. **Perfect overlap within an op** (time = max(compute, memory)), and **no
   overlap between ops** (total = sum).
6. **Causal masking ignored** in attention FLOPs (dense kernels compute the full S×S).
7. Softmax ≈ 5 FLOPs per element (exp counted as 1 FLOP); the 1/√d_h scaling
   is assumed folded into W_q. SiLU and the elementwise multiply in the MLP
   are ignored (O(T·f), negligible next to the O(T·d·f) matmuls).
8. **Units:** 1 GB = 10⁹ bytes (decimal), the same convention vendors use for
   bandwidth. GPU capacity uses the marketed size, which is slightly
   conservative because e.g. "80 GB" HBM is really 80 GiB ≈ 85.9 GB.

## Limitations

* This is an analytical estimate, **not a benchmark**. Real kernels reach maybe
  50-80% of peak, so real latency will be higher.
* Not included in the IR: embedding lookup, RMSNorm, rotary embeddings,
  residual adds, and the final LM head (2·T·d·V FLOPs ≈ 0.5 TFLOPs at
  S = 2048). Their *weights* are included in the memory calculation.
* Memory fit counts only weights + KV cache, not activations, CUDA context
  or framework overhead (typically 1-2 GB extra).
* Mistral's sliding-window attention (4096 tokens) is not modelled; for
  S ≤ 4096 it makes no difference.
* No multi-GPU / tensor parallelism, no cache effects (L2), no kernel-launch overhead.
