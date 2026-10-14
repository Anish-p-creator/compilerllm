// ============================================================
// LLMOpt Frontend — app.js
// ============================================================

const API_BASE = window.location.origin;

// Model / GPU metadata
const MODEL_INFO = {
    "llama-7b":   { layers: 32, kv_heads: 32, head_dim: 128, gqa: false },
    "mistral-7b": { layers: 32, kv_heads: 8,  head_dim: 128, gqa: true },
};
const HW_INFO = {
    "a100":    { mem: "80 GB",  bw: "2039 GB/s", tflops: "312 TFLOP/s" },
    "h100":    { mem: "80 GB",  bw: "3350 GB/s", tflops: "989 TFLOP/s" },
    "t4":      { mem: "16 GB",  bw: "320 GB/s",  tflops: "65 TFLOP/s" },
    "rtx4090": { mem: "24 GB",  bw: "1008 GB/s", tflops: "165 TFLOP/s" },
};

let currentReport = null;

// ---- Navigation ----
document.querySelectorAll(".nav-link").forEach(link => {
    link.addEventListener("click", (e) => {
        e.preventDefault();
        const section = link.dataset.section;
        switchSection(section);
    });
});

function switchSection(id) {
    document.querySelectorAll(".section").forEach(s => s.classList.remove("active"));
    document.querySelectorAll(".nav-link").forEach(l => l.classList.remove("active"));
    document.getElementById(id)?.classList.add("active");
    document.querySelector(`[data-section="${id}"]`)?.classList.add("active");
}

// ---- Model / Hardware Info Chips ----
document.getElementById("model-select").addEventListener("change", updateModelInfo);
document.getElementById("hardware-select").addEventListener("change", updateHWInfo);

function updateModelInfo() {
    const id = document.getElementById("model-select").value;
    const info = MODEL_INFO[id];
    if (!info) return;
    document.getElementById("chip-layers").textContent = info.layers;
    document.getElementById("chip-kv-heads").textContent = info.kv_heads;
    document.getElementById("chip-head-dim").textContent = info.head_dim;
    document.getElementById("chip-gqa").textContent = info.gqa ? "Yes" : "No";
}

function updateHWInfo() {
    const id = document.getElementById("hardware-select").value;
    const info = HW_INFO[id];
    if (!info) return;
    document.getElementById("chip-mem").textContent = info.mem;
    document.getElementById("chip-bw").textContent = info.bw;
    document.getElementById("chip-tflops").textContent = info.tflops;
}

// ---- Advanced Options Toggle ----
function toggleAdvanced() {
    const body = document.getElementById("advanced-body");
    const chevron = document.getElementById("adv-chevron");
    body.classList.toggle("open");
    chevron.classList.toggle("open");
}

// ---- Panel Toggle ----
function togglePanel(id) {
    const panel = document.querySelector(`#panel-${id} .panel-body`);
    const chevron = document.querySelector(`#panel-${id} .panel-chevron`);
    panel.classList.toggle("open");
    if (chevron) {
        chevron.style.transform = panel.classList.contains("open") ? "rotate(180deg)" : "";
    }
}

// ---- Toast ----
function showToast(msg) {
    const toast = document.getElementById("error-toast");
    document.getElementById("toast-text").textContent = msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 8000);
}
function closeToast() {
    document.getElementById("error-toast").classList.remove("show");
}

// ---- Run Optimization ----
async function runOptimization() {
    const btn = document.getElementById("run-btn");
    const status = document.getElementById("run-status");
    btn.disabled = true;
    status.textContent = "Connecting to backend...";

    // Switch to results tab
    switchSection("results");

    // Show loading
    document.getElementById("loading-container").style.display = "block";
    document.getElementById("results-content").style.display = "none";
    document.getElementById("results-subtitle").textContent = "Running optimization pipeline...";

    // Animate phases
    const phases = ["phase-1-step", "phase-2-step", "phase-3-step"];
    phases.forEach(p => { document.getElementById(p).className = "phase-step"; });

    let phaseIdx = 0;
    const phaseTimer = setInterval(() => {
        if (phaseIdx > 0) document.getElementById(phases[phaseIdx - 1]).className = "phase-step done";
        if (phaseIdx < phases.length) document.getElementById(phases[phaseIdx]).className = "phase-step active";
        phaseIdx++;
        if (phaseIdx > phases.length) clearInterval(phaseTimer);
    }, 600);

    // Collect form data
    const data = {
        model: document.getElementById("model-select").value,
        hardware: document.getElementById("hardware-select").value,
        batch_size: parseInt(document.getElementById("batch-input").value) || 1,
        sequence_length: parseInt(document.getElementById("seq-input").value) || 2048,
        generation_length: parseInt(document.getElementById("gen-input").value) || 256,
        max_context: parseInt(document.getElementById("ctx-input").value) || 0,
        precision: document.getElementById("precision-select").value,
        mode: document.getElementById("mode-select").value,
        pressure_threshold: parseFloat(document.getElementById("pressure-thresh").value) || 0.60,
        system_prompt_tokens: parseInt(document.getElementById("sys-tokens").value) || 0,
        fewshot_groups: parseInt(document.getElementById("fewshot-groups").value) || 0,
        fewshot_tokens: parseInt(document.getElementById("fewshot-tokens").value) || 0,
    };

    try {
        const resp = await fetch(`${API_BASE}/api/optimize`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data),
        });

        clearInterval(phaseTimer);
        phases.forEach(p => { document.getElementById(p).className = "phase-step done"; });

        if (!resp.ok) {
            const err = await resp.json();
            throw new Error(err.error || `Server error: ${resp.status}`);
        }

        const report = await resp.json();
        currentReport = report;

        // Small delay for UX
        await new Promise(r => setTimeout(r, 500));

        renderResults(report);
        document.getElementById("loading-container").style.display = "none";
        document.getElementById("results-content").style.display = "block";
        document.getElementById("results-subtitle").textContent =
            `${report.summary.model} on ${report.summary.hardware} — ${report.summary.mode} mode`;
        status.textContent = "✓ Pipeline completed successfully";
        status.style.color = "#34d399";
    } catch (e) {
        clearInterval(phaseTimer);
        document.getElementById("loading-container").style.display = "none";
        showToast(e.message);
        status.textContent = "✗ Error: " + e.message;
        status.style.color = "#fb7185";
        switchSection("configure");
    } finally {
        btn.disabled = false;
    }
}

// ---- Render Results ----
function renderResults(r) {
    // Summary cards
    document.getElementById("sum-memory").textContent = fmtGB(r.p1_total_memory_gb);
    document.getElementById("sum-memory-detail").textContent =
        `Weights: ${fmtGB(r.p1_weights_gb)} + KV: ${fmtGB(r.p1_kv_cache_gb)}`;

    document.getElementById("sum-latency").textContent = fmtMs(r.p1_estimated_latency_ms);
    document.getElementById("sum-latency-detail").textContent =
        `${r.p1_total_flops ? fmtFlops(r.p1_total_flops) : '—'} FLOPs, AI: ${r.p1_arithmetic_intensity}`;

    const selOpts = r.summary.selected_optimizations || [];
    document.getElementById("sum-opts").textContent = selOpts.length;
    document.getElementById("sum-opts-detail").textContent =
        selOpts.length > 0 ? selOpts.join(", ") : "No optimizations selected";

    document.getElementById("sum-fit").textContent = r.p1_fits_in_memory ? "YES" : "NO";
    document.getElementById("sum-fit").style.color = r.p1_fits_in_memory ? "#34d399" : "#fb7185";
    document.getElementById("sum-fit-detail").textContent =
        `${fmtGB(r.p1_total_memory_gb)} of ${HW_INFO[r.input_config.hardware]?.mem || '?'}`;

    // P1: Op Breakdown
    renderP1(r);

    // P2: KV-Cache Comparison
    renderP2(r);

    // P3: Fusion + Scheduling + Auto-tuner
    renderP3(r);

    // All Candidates
    renderCandidates(r);
}

function renderP1(r) {
    // Metrics
    const metricsDiv = document.getElementById("p1-metrics");
    metricsDiv.innerHTML = [
        metricHTML("Parameters", fmtB(r.p1_parameter_count)),
        metricHTML("Weights", fmtGB(r.p1_weights_gb)),
        metricHTML("KV Cache", fmtGB(r.p1_kv_cache_gb)),
        metricHTML("Total FLOPs", fmtFlops(r.p1_total_flops)),
        metricHTML("Arith. Intensity", `${r.p1_arithmetic_intensity} FLOP/B`),
        metricHTML("Est. Latency", fmtMs(r.p1_estimated_latency_ms)),
    ].join("");

    // Table
    const tbody = document.getElementById("p1-tbody");
    tbody.innerHTML = r.p1_op_breakdown.map(op => `
        <tr>
            <td style="font-family:var(--font-sans);font-weight:600;">${op.operation}</td>
            <td>${op.flops_fmt}</td>
            <td>${op.bytes_fmt}</td>
            <td>${op.arithmetic_intensity}</td>
            <td>${op.time_fmt}</td>
            <td>${op.pct_time}%</td>
            <td><span class="bound-badge ${op.bound === 'COMPUTE_BOUND' ? 'bound-compute' : 'bound-memory'}">${op.bound === 'COMPUTE_BOUND' ? 'Compute' : 'Memory'}</span></td>
        </tr>
    `).join("");
}

function renderP2(r) {
    const div = document.getElementById("p2-comparison");
    const base = r.p2_baseline_metrics;
    const opt = r.p2_optimized_metrics;

    if (!base || !Object.keys(base).length) {
        div.innerHTML = `<div style="color:var(--text-muted);padding:1rem;">P2 analysis not available for current configuration.</div>`;
        return;
    }

    const rows = [
        ["KV Cache (GiB)", fmtN(base.kv_gib, 2), fmtN(opt.kv_gib, 2), pctDelta(base.kv_gib, opt.kv_gib)],
        ["Peak GPU Memory (GiB)", fmtN(base.total_gib, 2), fmtN(opt.total_gib, 2), pctDelta(base.total_gib, opt.total_gib)],
        ["Fits in Memory", base.fits ? "Yes" : "NO (OOM)", opt.fits ? "Yes" : "NO (OOM)", ""],
        ["Max Concurrent Requests", base.max_batch, opt.max_batch, pctDelta(base.max_batch, opt.max_batch, true)],
        ["TPOT (ms)", fmtN(base.tpot_ms, 2), fmtN(opt.tpot_ms, 2), pctDelta(base.tpot_ms, opt.tpot_ms)],
        ["Throughput (tok/s)", Math.round(base.throughput_tok_s), Math.round(opt.throughput_tok_s), pctDelta(base.throughput_tok_s, opt.throughput_tok_s, true)],
        ["TTFT (ms)", Math.round(base.ttft_ms), Math.round(opt.ttft_ms), pctDelta(base.ttft_ms, opt.ttft_ms)],
    ];

    let html = `<div class="p2-header"><span>Metric</span><span>Baseline</span><span>Optimized</span><span>Change</span></div>`;
    html += rows.map(([label, bval, oval, delta]) => `
        <div class="p2-row">
            <span class="p2-label">${label}</span>
            <span class="p2-baseline">${bval}</span>
            <span class="p2-optimized">${oval}</span>
            <span class="p2-delta ${delta}">${delta}</span>
        </div>
    `).join("");

    if (r.p2_selected_passes && r.p2_selected_passes.length > 0) {
        html += `<div style="margin-top:1rem;padding:12px 16px;background:var(--accent-emerald-dim);border-radius:var(--radius-sm);font-size:0.88rem;">
            <strong style="color:var(--accent-emerald);">Selected:</strong> ${r.p2_selected_passes.join(" + ")}
        </div>`;
    }
    if (r.p2_plan_notes && r.p2_plan_notes.length > 0) {
        html += r.p2_plan_notes.map(n => `<div style="margin-top:0.5rem;padding:8px 16px;font-size:0.82rem;color:var(--text-muted);">ℹ ${n}</div>`).join("");
    }

    div.innerHTML = html;
}

function renderP3(r) {
    const div = document.getElementById("p3-content");
    let html = "";

    // Fusion Levels
    if (r.p3_fusion_levels && r.p3_fusion_levels.length > 0) {
        html += `<div class="p3-section"><h3><span class="p3-tag tag-fusion">Fusion</span> Kernel Fusion Analysis</h3>`;
        html += `<div class="table-wrapper"><table class="data-table"><thead><tr>
            <th>Level</th><th>Fused Ops</th><th>Traffic Saved/Layer</th>
            <th>Unfused → Fused (per layer)</th><th>Mem Saved</th><th>Δ ms/step</th>
        </tr></thead><tbody>`;
        r.p3_fusion_levels.forEach(lv => {
            const isBest = lv.is_best;
            html += `<tr style="${isBest ? 'background:var(--accent-indigo-dim);' : ''}">
                <td>L${lv.level}${isBest ? ' ★' : ''}</td>
                <td style="font-family:var(--font-sans);">${lv.label}</td>
                <td>${fmtBytes(lv.traffic_saved_bytes)}</td>
                <td>${lv.unfused_ms.toFixed(4)} → ${lv.fused_ms.toFixed(4)} ms</td>
                <td>${lv.mem_saved_gb.toFixed(3)} GB</td>
                <td class="${lv.delta_ms < 0 ? 'delta-good' : 'delta-bad'}">${lv.delta_ms > 0 ? '+' : ''}${lv.delta_ms.toFixed(3)}</td>
            </tr>`;
        });
        html += `</tbody></table></div></div>`;
    }

    // Scheduling
    if (r.p3_split_kv || r.p3_persistent) {
        html += `<div class="p3-section"><h3><span class="p3-tag tag-schedule">Schedule</span> Decode Scheduling</h3>`;
        if (r.p3_split_kv) {
            const sp = r.p3_split_kv;
            const label = sp.forced
                ? `Not needed: ${sp.units} blocks ≥ target ${sp.target}`
                : `${sp.units} blocks < target ${sp.target} → ${sp.splits} splits → ${sp.units * sp.splits} blocks`;
            html += `<div class="metric-item" style="margin-bottom:0.75rem;">
                <span class="metric-label">Split KV</span>
                <span class="metric-value" style="font-size:0.9rem;">${label}</span>
                <span style="font-size:0.8rem;color:${sp.delta_ms < 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${sp.delta_ms > 0 ? '+' : ''}${sp.delta_ms.toFixed(3)} ms/step</span>
            </div>`;
        }
        if (r.p3_persistent) {
            const pp = r.p3_persistent;
            html += `<div class="metric-item">
                <span class="metric-label">Persistent Kernel</span>
                <span class="metric-value" style="font-size:0.9rem;">Launch share: ${(pp.launch_share * 100).toFixed(1)}% (${pp.launches} launches)</span>
                <span style="font-size:0.8rem;color:${pp.delta_ms < 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)'};">${pp.delta_ms > 0 ? '+' : ''}${pp.delta_ms.toFixed(3)} ms/step</span>
            </div>`;
        }
        html += `</div>`;
    }

    // Auto-tuner
    if (r.p3_autotune_top && r.p3_autotune_top.length > 0) {
        html += `<div class="p3-section"><h3><span class="p3-tag tag-tune">Tune</span> Fused Attention Auto-Tuner</h3>`;
        if (r.p3_autotune_baseline) {
            html += `<div class="metric-item" style="margin-bottom:1rem;">
                <span class="metric-label">Baseline</span>
                <span class="metric-value" style="font-size:0.9rem;">${r.p3_autotune_baseline.config} → ${r.p3_autotune_baseline.time_us_per_layer} µs/layer (${r.p3_autotune_baseline.bound}-bound)</span>
            </div>`;
        }
        if (r.p3_autotune_chosen) {
            const ch = r.p3_autotune_chosen;
            html += `<div class="metric-item" style="margin-bottom:1rem;border-color:var(--accent-emerald);border-width:2px;">
                <span class="metric-label">Chosen Config</span>
                <span class="metric-value" style="font-size:0.9rem;">${ch.config} → ${ch.time_us_per_layer} µs/layer (${ch.bound}-bound, ${ch.smem_kb} KB smem, ${ch.blocks_per_sm} blk/SM)</span>
            </div>`;
        }
        html += `<div class="table-wrapper"><table class="data-table"><thead><tr>
            <th>#</th><th>Config</th><th>µs/layer</th><th>Bound</th><th>Smem</th><th>Blk/SM</th><th>vs Baseline</th>
        </tr></thead><tbody>`;
        r.p3_autotune_top.forEach(e => {
            html += `<tr style="${e.is_chosen ? 'background:var(--accent-emerald-dim);' : ''}">
                <td>${e.rank}${e.is_chosen ? ' ★' : ''}</td>
                <td>${e.config}</td>
                <td>${e.time_us_per_layer}</td>
                <td><span class="bound-badge ${e.bound === 'compute' ? 'bound-compute' : 'bound-memory'}">${e.bound}</span></td>
                <td>${e.smem_kb} KB</td>
                <td>${e.blocks_per_sm}</td>
                <td>${e.speedup_vs_baseline}x</td>
            </tr>`;
        });
        html += `</tbody></table></div></div>`;
    }

    if (!html) {
        html = `<div style="color:var(--text-muted);padding:1rem;">P3 analysis not available for current configuration.</div>`;
    }
    div.innerHTML = html;
}

function renderCandidates(r) {
    const tbody = document.getElementById("candidates-tbody");
    const cands = r.candidates || [];
    if (cands.length === 0) {
        tbody.innerHTML = `<tr><td colspan="7" style="color:var(--text-muted);text-align:center;">No candidates proposed</td></tr>`;
        return;
    }
    tbody.innerHTML = cands.map(c => {
        const statusClass = c.selected ? "status-selected" : (c.latency_delta_ms > 0 ? "status-rejected" : "status-proposed");
        const statusText = c.selected ? "Selected" : (c.latency_delta_ms > 0 ? "Rejected" : "Proposed");
        return `<tr>
            <td><span class="phase-badge ${c.phase.toLowerCase()}">${c.phase}</span></td>
            <td style="font-family:var(--font-sans);font-weight:600;">${c.name}</td>
            <td style="font-family:var(--font-sans);max-width:250px;overflow:hidden;text-overflow:ellipsis;" title="${escapeHtml(c.reason)}">${c.description}</td>
            <td class="${c.memory_delta_gb < 0 ? 'delta-good' : (c.memory_delta_gb > 0 ? 'delta-bad' : '')}">${c.memory_delta_gb !== 0 ? (c.memory_delta_gb > 0 ? '+' : '') + c.memory_delta_gb.toFixed(3) + ' GB' : '—'}</td>
            <td class="${c.latency_delta_ms < 0 ? 'delta-good' : (c.latency_delta_ms > 0 ? 'delta-bad' : '')}">${c.latency_delta_ms !== 0 ? (c.latency_delta_ms > 0 ? '+' : '') + c.latency_delta_ms.toFixed(3) + ' ms' : '—'}</td>
            <td><span class="risk-${c.accuracy_risk}">${c.accuracy_risk}</span></td>
            <td><span class="status-badge ${statusClass}">${statusText}</span></td>
        </tr>`;
    }).join("");
}

// ---- Formatting Helpers ----
function fmtGB(gb) {
    if (gb >= 1000) return (gb / 1000).toFixed(2) + " TB";
    if (gb >= 1) return gb.toFixed(2) + " GB";
    return (gb * 1000).toFixed(1) + " MB";
}
function fmtMs(ms) {
    if (ms >= 1000) return (ms / 1000).toFixed(3) + " s";
    if (ms >= 1) return ms.toFixed(2) + " ms";
    return (ms * 1000).toFixed(1) + " µs";
}
function fmtB(n) {
    if (n >= 1e12) return (n / 1e12).toFixed(2) + "T";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
    return n.toLocaleString();
}
function fmtFlops(n) {
    if (n >= 1e15) return (n / 1e15).toFixed(2) + " PFLOP";
    if (n >= 1e12) return (n / 1e12).toFixed(2) + " TFLOP";
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GFLOP";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MFLOP";
    return n + " FLOP";
}
function fmtBytes(n) {
    n = Math.abs(n);
    if (n >= 1e9) return (n / 1e9).toFixed(2) + " GB";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + " MB";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + " KB";
    return n + " B";
}
function fmtN(v, d) {
    if (v === null || v === undefined) return "—";
    return typeof v === "number" ? v.toFixed(d) : String(v);
}
function pctDelta(a, b, higherBetter = false) {
    if (!a || a === 0) return '<span class="delta-neutral">—</span>';
    const d = ((b - a) / a) * 100;
    const improved = higherBetter ? d > 0 : d < 0;
    const cls = Math.abs(d) < 0.1 ? "delta-neutral" : (improved ? "delta-good" : "delta-bad");
    const arrow = d > 0 ? "↑" : d < 0 ? "↓" : "=";
    return `<span class="${cls}">${arrow} ${Math.abs(d).toFixed(1)}%</span>`;
}
function metricHTML(label, value) {
    return `<div class="metric-item"><span class="metric-label">${label}</span><span class="metric-value">${value}</span></div>`;
}
function escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
