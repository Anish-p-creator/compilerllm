# LLMOpt — Compiler Optimization for LLMs

## Overview

LLMOpt is an LLM-aware compiler optimization project focused on improving the execution efficiency of large language models on GPU hardware.

The project explores multiple optimization approaches targeting memory management, redundant attention computation, and autoregressive decoding performance.

## Optimization Approaches

### 1. KV-Cache Optimization
Explores efficient KV-cache memory management and layout optimization, inspired by vAttention.

### 2. Attention Optimization
Explores reducing redundant computation, including shared-prefix optimization inspired by ChunkAttention.

### 3. Decode Scheduling
Explores decode-stage scheduling, work splitting, and execution optimization inspired by PersistentKV.

## Repository Structure

- `approach-1-kv-cache/` — KV-cache optimization
- `approach-2-attention-optimization/` — Attention optimization
- `approach-3-decode-scheduling/` — Decode scheduling
- `integration/` — Combined compiler pipeline (in progress)

## Setup

Refer to the README and dependency files inside each approach directory for its specific environment and execution instructions.

## Evaluation

The project aims to evaluate:

- Time to first token (TTFT)
- Time per output token (TPOT)
- End-to-end latency
- Throughput
- Peak GPU memory usage
- Kernel launch overhead
- Correctness of model outputs

## Integration

The three approaches are maintained independently while the combined compiler pipeline is developed.

## Contributors

Add the names and GitHub profiles of all four team members here.
