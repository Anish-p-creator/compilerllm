"""
Flask API backend for the LLMOpt unified optimization pipeline.

Endpoints:
    GET  /api/presets        → available models, GPUs, precisions
    POST /api/optimize       → run the full pipeline, return JSON report
    GET  /api/health         → health check
    GET  /                   → serve the frontend
"""
from __future__ import annotations

import json
import os
import traceback
from dataclasses import asdict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from pipeline import PipelineInput, run_pipeline

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
MODELS = [
    {"id": "llama-7b", "name": "LLaMA-7B", "params": "6.74B", "layers": 32, "kv_heads": 32, "gqa": False},
    {"id": "mistral-7b", "name": "Mistral-7B", "params": "7.24B", "layers": 32, "kv_heads": 8, "gqa": True},
]

GPUS = [
    {"id": "a100", "name": "NVIDIA A100 80GB SXM", "memory_gb": 80, "bandwidth_gbps": 2039, "fp16_tflops": 312},
    {"id": "h100", "name": "NVIDIA H100 80GB SXM5", "memory_gb": 80, "bandwidth_gbps": 3350, "fp16_tflops": 989.4},
    {"id": "t4", "name": "NVIDIA T4", "memory_gb": 16, "bandwidth_gbps": 320, "fp16_tflops": 65},
    {"id": "rtx4090", "name": "NVIDIA GeForce RTX 4090", "memory_gb": 24, "bandwidth_gbps": 1008, "fp16_tflops": 165.2},
]

PRECISIONS = [
    {"id": "fp32", "name": "FP32", "bytes": 4},
    {"id": "fp16", "name": "FP16", "bytes": 2},
    {"id": "bf16", "name": "BF16", "bytes": 2},
    {"id": "int8", "name": "INT8", "bytes": 1},
    {"id": "int4", "name": "INT4", "bytes": 0.5},
]

MODES = [
    {"id": "decode", "name": "Decode (autoregressive generation)"},
    {"id": "prefill", "name": "Prefill (prompt processing)"},
    {"id": "mixed", "name": "Mixed (prefill + decode)"},
]


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/presets")
def presets():
    return jsonify({
        "models": MODELS,
        "gpus": GPUS,
        "precisions": PRECISIONS,
        "modes": MODES,
    })


def _input_from(data: dict) -> PipelineInput:
    return PipelineInput(
        model=data.get("model", "llama-7b"),
        hardware=data.get("hardware", "a100"),
        batch_size=int(data.get("batch_size", 1)),
        sequence_length=int(data.get("sequence_length", 2048)),
        generation_length=int(data.get("generation_length", 256)),
        max_context=int(data.get("max_context", 0)),
        precision=data.get("precision", "fp16"),
        mode=data.get("mode", "decode"),
        pressure_threshold=float(data.get("pressure_threshold", 0.60)),
        block_size=int(data.get("block_size", 16)),
        chunk_size=int(data.get("chunk_size", 16)),
        system_prompt_tokens=int(data.get("system_prompt_tokens", 0)),
        fewshot_groups=int(data.get("fewshot_groups", 0)),
        fewshot_tokens=int(data.get("fewshot_tokens", 0)),
    )


@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        data = request.get_json(force=True)
        report = run_pipeline(_input_from(data))
        return jsonify(asdict(report))
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 400


def _match_gpu_preset(name: str):
    n = name.lower().replace(" ", "")
    for key, needle in (("h100", "h100"), ("a100", "a100"), ("rtx4090", "4090"), ("t4", "t4")):
        if needle in n:
            return key
    return None


@app.route("/api/detect")
def detect():
    """Read the GPU on *this* machine (pynvml → nvidia-smi) and map it to a preset."""
    gpus, source = [], None
    try:
        import pynvml
        pynvml.nvmlInit()
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            gpus.append({"name": name, "memory_gb": round(pynvml.nvmlDeviceGetMemoryInfo(h).total / 1e9, 1)})
        source = "pynvml"
    except Exception:
        if not gpus:
            import shutil
            import subprocess
            if shutil.which("nvidia-smi"):
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5).stdout
                for line in out.strip().splitlines():
                    name, mem = [s.strip() for s in line.split(",")]
                    gpus.append({"name": name, "memory_gb": round(float(mem) / 1024, 1)})
                source = "nvidia-smi"
    for g in gpus:
        g["preset"] = _match_gpu_preset(g["name"])
    return jsonify({
        "found": bool(gpus), "source": source, "gpus": gpus,
        "message": None if gpus else
        "No NVIDIA GPU visible on this machine (tried pynvml and nvidia-smi). "
        "Pick a target GPU manually — the pipeline is analytical, so it can plan for hardware you don't have.",
    })


SWEEPABLE = {"batch_size", "sequence_length", "generation_length", "system_prompt_tokens"}


def _metrics(m):
    if not m:
        return None
    return {k: m.get(k) for k in ("tpot_ms", "throughput_tok_s", "total_gib", "fits", "max_batch", "ttft_ms")}


@app.route("/api/sweep", methods=["POST"])
def sweep():
    """Run the pipeline across several values of one workload parameter."""
    try:
        data = request.get_json(force=True)
        param = data.get("param", "batch_size")
        if param not in SWEEPABLE:
            return jsonify({"error": f"param must be one of {sorted(SWEEPABLE)}"}), 400
        values = [int(v) for v in data.get("values", [])][:16]
        base = data.get("config", {})
        points = []
        for v in values:
            rep = asdict(run_pipeline(_input_from({**base, param: v})))
            points.append({
                "value": v,
                "baseline": _metrics(rep["p2_baseline_metrics"]),
                "optimized": _metrics(rep["p2_optimized_metrics"]),
                "selected": rep["summary"]["selected_optimizations"],
                "roofline_latency_ms": rep["p1_estimated_latency_ms"],
            })
        return jsonify({"param": param, "points": points})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 400


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get("PORT", 5000)), host="0.0.0.0")
