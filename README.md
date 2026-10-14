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

- `LLMOpt_P1/` — IR construction and roofline analysis
- `LLMOpt_P2/` — KV-cache passes (INT8 KV, prefix sharing)
- `LLMOpt_P3/` — Kernel fusion, decode scheduling, auto-tuning
- `integration/` — Unified pipeline (`pipeline.py`), Flask API (`server.py`) and the LLMOpt Studio frontend (`frontend/`)

## Setup

```bash
pip install -r requirements.txt
python integration/server.py
```

Then open http://localhost:5000. The Studio re-runs the full P1 → P2 → P3 pipeline on every change; the Deploy tab turns the chosen plan into vLLM / TensorRT-LLM / TGI launch flags. The previous UI is at `/classic.html`.

API endpoints: `POST /api/optimize`, `POST /api/sweep`, `GET /api/presets`, `GET /api/detect` (reads the local NVIDIA GPU via pynvml or nvidia-smi), `GET /api/health`.

## Deployment

`render.yaml` deploys the app to Render: on render.com choose **New → Blueprint** and select this repository. It runs `gunicorn --chdir integration server:app`. (`/api/detect` reports no GPU on a hosted server; pick the target GPU manually.)

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

`integration/pipeline.py` loads the three approaches side by side (each keeps its own module layout) and runs them as one compiler pipeline: P1 builds the IR and roofline estimates, P2 applies KV-cache passes under memory pressure, and P3 evaluates kernel fusion, split-KV / persistent-kernel scheduling and tile auto-tuning. All candidates are merged into one plan.

## Contributors

Add the names and GitHub profiles of all four team members here.
