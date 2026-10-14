/* LLMOpt Studio — live front end for the unified optimization pipeline.
 * Every control change re-runs /api/optimize (debounced, previous request
 * aborted), then a background /api/sweep feeds the what-if chart. */
"use strict";

const API = window.location.origin;
const USABLE_FRACTION = 0.92; // matches P2 Options.usable_memory_fraction

const DEFAULTS = {
  model: "llama-7b", hardware: "a100", precision: "fp16", mode: "decode",
  batch_size: 1, sequence_length: 2048, generation_length: 256,
  system_prompt_tokens: 0, fewshot_groups: 0, fewshot_tokens: 0, pressure_threshold: 0.6,
};

const SCENARIOS = [
  { name: "Chatbot", set: { batch_size: 16, sequence_length: 1024, generation_length: 256, system_prompt_tokens: 512 } },
  { name: "Long-context RAG", set: { batch_size: 4, sequence_length: 16384, generation_length: 512, system_prompt_tokens: 1024 } },
  { name: "Offline batch", set: { batch_size: 128, sequence_length: 2048, generation_length: 512 } },
  { name: "Few-shot agent", set: { batch_size: 32, sequence_length: 2048, generation_length: 256, system_prompt_tokens: 256, fewshot_groups: 4, fewshot_tokens: 1024 } },
  { name: "Edge T4", set: { hardware: "t4", batch_size: 4, sequence_length: 2048, generation_length: 256 } },
  { name: "Single user", set: { batch_size: 1, sequence_length: 512, generation_length: 128 } },
];

const PASS_CATALOG = {
  P1: [{ name: "roofline", label: "roofline analysis" }],
  P2: [{ name: "int8_kv", label: "int8_kv" }, { name: "prefix_share", label: "prefix_share" }],
  P3: [
    { name: "fusion", label: "fusion" }, { name: "split_kv", label: "split_kv" },
    { name: "persistent_kernel", label: "persistent_kernel" }, { name: "autotune", label: "autotune" },
  ],
};

const NOT_PROPOSED_WHY = {
  int8_kv: (r) => `KV pressure ${pct(r.p2_kv_pressure)} is below the ${pct(r.input_config.pressure_threshold)} trigger, so the KV-cache pass did not fire.`,
  prefix_share: () => "No shared prefix in this workload. Add a shared system prompt or few-shot groups to give the radix trie something to dedupe.",
  fusion: () => "Fusion only applies to decode-phase attention in this model.",
  split_kv: () => "Not proposed for this phase.",
  persistent_kernel: () => "Not proposed for this phase.",
  autotune: () => "Not proposed for this phase.",
};

const HF_IDS = { "llama-7b": "meta-llama/Llama-2-7b-hf", "mistral-7b": "mistralai/Mistral-7B-v0.1" };

// ---------------------------------------------------------------------------
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const pct = (x) => `${Math.round((x || 0) * 100)}%`;
const fmtInt = (n) => (n >= 1024 && n % 1024 === 0 ? `${n / 1024}k` : n.toLocaleString());
const fmt = (n, d = 1) => (n == null || isNaN(n) ? "–" : Number(n).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d }));
const debounce = (fn, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; };

const state = { ...DEFAULTS, ...readHash() };
let presets = { gpus: [] };
let report = null;
let prevKpi = {};
let inflight = null;
let sweepInflight = null;
let sweepParam = "batch_size";
let candFilter = "all";
let activePass = null;
let engine = "vllm";
let detectedPreset = null;
const charts = {};

function readHash() {
  try {
    const o = Object.fromEntries(new URLSearchParams(location.hash.slice(1)));
    for (const k of Object.keys(o)) if (!(k in DEFAULTS)) delete o[k];
    for (const k of Object.keys(o)) if (typeof DEFAULTS[k] === "number") o[k] = Number(o[k]);
    return o;
  } catch { return {}; }
}
function writeHash() {
  const diff = Object.fromEntries(Object.entries(state).filter(([k, v]) => DEFAULTS[k] !== v));
  history.replaceState(null, "", "#" + new URLSearchParams(diff).toString());
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function initControls() {
  const sc = $("#scenarios");
  SCENARIOS.forEach((s) => {
    const b = document.createElement("button");
    b.className = "chip"; b.textContent = s.name;
    b.onclick = () => {
      Object.assign(state, { ...DEFAULTS, model: state.model, hardware: state.hardware, precision: state.precision }, s.set);
      $$(".chip", sc).forEach((c) => c.classList.toggle("on", c === b));
      syncControls(true); changed();
    };
    sc.appendChild(b);
  });

  $$(".seg[data-key]").forEach((seg) => {
    seg.addEventListener("click", (e) => {
      const b = e.target.closest("button"); if (!b) return;
      state[seg.dataset.key] = b.dataset.v;
      clearScenario(); syncControls(); changed();
    });
  });

  $$(".slider").forEach((sl) => {
    const inp = $("input", sl);
    inp.min = sl.dataset.min; inp.max = sl.dataset.max; inp.step = sl.dataset.step;
    inp.addEventListener("input", () => {
      const x = Number(inp.value);
      let v = sl.dataset.scale === "log2" ? Math.round(2 ** x) : x;
      if (sl.dataset.scale === "log2" && v > 256) v = Math.round(v / 64) * 64;
      state[sl.dataset.key] = v;
      paintSlider(sl); clearScenario(); changed();
    });
  });

  $("#sweep-param").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    sweepParam = b.dataset.v;
    $$("#sweep-param button").forEach((x) => x.classList.toggle("on", x === b));
    runSweep();
  });
  $("#cand-filter").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    candFilter = b.dataset.v;
    $$("#cand-filter button").forEach((x) => x.classList.toggle("on", x === b));
    if (report) renderCandidates(report);
  });
  $("#engine").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    engine = b.dataset.v;
    $$("#engine button").forEach((x) => x.classList.toggle("on", x === b));
    if (report) renderDeploy(report);
  });
  $("#detect-btn").onclick = detectGpu;
  $("#fleet-rps").oninput = () => report && renderFleet(report);
  $$(".copy").forEach((b) => (b.onclick = () => {
    navigator.clipboard?.writeText($("#" + b.dataset.copy).innerText).then(() => toast("Copied to clipboard"));
  }));
  $$(".tab[data-view]").forEach((t) => (t.onclick = () => showView(t.dataset.view)));
}

function clearScenario() { $$("#scenarios .chip").forEach((c) => c.classList.remove("on")); }

function showView(v) {
  $$(".tab[data-view]").forEach((t) => t.classList.toggle("active", t.dataset.view === v));
  $$(".view").forEach((s) => s.classList.toggle("active", s.id === "view-" + v));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function paintSlider(sl, flash) {
  const inp = $("input", sl), out = $("output", sl);
  const v = state[sl.dataset.key];
  out.textContent = sl.dataset.fmt === "pct" ? pct(v) : fmtInt(v);
  const p = (inp.value - inp.min) / (inp.max - inp.min) * 100;
  inp.style.background = `linear-gradient(90deg, var(--accent) 0%, var(--accent-2) ${p}%, rgba(255,255,255,.08) ${p}%)`;
  if (flash) { out.classList.add("flash"); setTimeout(() => out.classList.remove("flash"), 500); }
}

function syncControls(flash = false) {
  $$(".seg[data-key]").forEach((seg) => $$("button", seg).forEach((b) => b.classList.toggle("on", b.dataset.v === state[seg.dataset.key])));
  $$(".slider").forEach((sl) => {
    const v = state[sl.dataset.key];
    $("input", sl).value = sl.dataset.scale === "log2" ? Math.log2(Math.max(1, v)) : v;
    paintSlider(sl, flash);
  });
  $$(".gpu").forEach((g) => g.classList.toggle("on", g.dataset.v === state.hardware));
  $("#hf-id").textContent = HF_IDS[state.model];
}

function renderGpus() {
  const grid = $("#gpu-grid");
  const maxBw = Math.max(...presets.gpus.map((g) => g.bandwidth_gbps));
  grid.innerHTML = "";
  presets.gpus.forEach((g) => {
    const b = document.createElement("button");
    b.className = "gpu"; b.dataset.v = g.id;
    if (g.id === detectedPreset) b.classList.add("detected");
    b.innerHTML = `<b>${g.name.replace(/NVIDIA (GeForce )?/, "").replace(/ \d+GB.*/, "")}</b>
      <span>${g.memory_gb} GB · ${(g.bandwidth_gbps / 1000).toFixed(1)} TB/s</span>
      <div class="meter"><i style="width:${(g.bandwidth_gbps / maxBw) * 100}%"></i></div>`;
    b.title = `${g.name}\n${g.fp16_tflops} FP16 TFLOP/s`;
    b.onclick = () => { state.hardware = g.id; clearScenario(); syncControls(); changed(); };
    grid.appendChild(b);
  });
}

async function detectGpu() {
  const msg = $("#detect-msg");
  msg.textContent = "Querying this machine…";
  try {
    const d = await (await fetch(`${API}/api/detect`)).json();
    if (d.found) {
      const g = d.gpus[0];
      detectedPreset = g.preset;
      msg.textContent = `Found ${g.name} (${g.memory_gb} GB) via ${d.source}.` + (g.preset ? "" : " No matching preset; pick the closest one.");
      renderGpus();
      if (g.preset) { state.hardware = g.preset; syncControls(); changed(); }
    } else {
      msg.textContent = d.message;
    }
  } catch (e) { msg.textContent = "Detection failed: " + e.message; }
}

// ---------------------------------------------------------------------------
// Running the pipeline
// ---------------------------------------------------------------------------
const changed = debounce(() => { writeHash(); runOptimize(); }, 110);
const sweepSoon = debounce(() => runSweep(), 350);

function setPill(cls, text) {
  const p = $("#live-pill");
  p.className = "live-pill " + cls;
  $("#live-text").textContent = text;
}

async function runOptimize() {
  inflight?.abort();
  const ctrl = (inflight = new AbortController());
  setPill("busy", "compiling…");
  const t0 = performance.now();
  try {
    const res = await fetch(`${API}/api/optimize`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state), signal: ctrl.signal,
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    const ms = performance.now() - t0;
    report = data;
    setPill("ok", `live · ${ms.toFixed(0)} ms`);
    $("#run-ms").textContent = `round-trip ${ms.toFixed(0)} ms`;
    render(data);
    sweepSoon();
  } catch (e) {
    if (e.name === "AbortError") return;
    setPill("err", "error");
    toast("Pipeline error: " + e.message);
  }
}

async function runSweep() {
  sweepInflight?.abort();
  const ctrl = (sweepInflight = new AbortController());
  const values = {
    batch_size: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    sequence_length: [256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
    system_prompt_tokens: [0, 128, 256, 512, 1024, 2048, 4096],
  }[sweepParam];
  try {
    const res = await fetch(`${API}/api/sweep`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ param: sweepParam, values, config: state }), signal: ctrl.signal,
    });
    const d = await res.json();
    if (d.error) throw new Error(d.error);
    renderSweep(d);
  } catch (e) { if (e.name !== "AbortError") console.warn(e); }
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function render(r) {
  renderKpis(r);
  renderFlow(r);
  renderMemory(r);
  renderOps(r);
  renderRoofline(r);
  renderFusion(r);
  renderTune(r);
  renderSched(r);
  renderCandidates(r);
  renderTerm(r);
  renderDeploy(r);
}

function gpuPreset() { return presets.gpus.find((g) => g.id === state.hardware) || presets.gpus[0] || {}; }
function usableGib() { return (gpuPreset().memory_gb || 80) * USABLE_FRACTION; }

function animateNum(el, to, digits) {
  const from = parseFloat(el.dataset.v || "0");
  el.dataset.v = to;
  const t0 = performance.now(), dur = 650;
  const step = (t) => {
    const k = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    el.textContent = fmt(from + (to - from) * e, digits);
    if (k < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function setKpi(key, value, digits, delta, deltaClass) {
  const card = $(`.kpi[data-k=${key}]`);
  const num = $(".num", card);
  if (typeof value === "number") animateNum(num, value, digits); else num.textContent = value;
  const d = $(".kpi-d", card);
  d.textContent = delta || ""; d.className = "kpi-d " + (deltaClass || "");
  if (prevKpi[key] !== value) {
    card.classList.remove("pulse"); void card.offsetWidth; card.classList.add("pulse");
    setTimeout(() => card.classList.remove("pulse"), 60);
  }
  prevKpi[key] = value;
}

function changeText(base, opt, lowerIsBetter, unit = "%") {
  if (!base || base === opt) return ["no change vs baseline", ""];
  const ch = (opt - base) / base * 100;
  const good = lowerIsBetter ? ch < 0 : ch > 0;
  return [`${ch > 0 ? "▲" : "▼"} ${Math.abs(ch).toFixed(1)}${unit} vs baseline`, good ? "good" : "bad"];
}

function renderKpis(r) {
  const b = r.p2_baseline_metrics || {}, o = r.p2_optimized_metrics || {};
  const cap = usableGib();
  const [md, mc] = changeText(b.total_gib, o.total_gib, true);
  setKpi("mem", o.total_gib, 1, md, mc);
  const bar = $(".kpi[data-k=mem] .kpi-bar");
  $("i", bar).style.width = Math.min(100, (b.total_gib / cap) * 100) + "%";
  $("b", bar).style.width = Math.min(100, (o.total_gib / cap) * 100) + "%";
  // P2 metrics model memory traffic; P3 kernel passes report their own per-step savings.
  const p3Saved = r.candidates.filter((c) => c.phase === "P3" && r.summary.selected_optimizations.includes(c.name))
    .reduce((a, c) => a + Math.min(0, c.latency_delta_ms), 0);
  let [td, tc] = changeText(b.tpot_ms, o.tpot_ms, true);
  if (p3Saved < 0) { td = (tc ? td.replace(" vs baseline", "") + " · " : "") + `P3 kernels −${fmt(-p3Saved, 2)} ms/step`; tc = "good"; }
  setKpi("tpot", o.tpot_ms, 2, td, tc);
  const [pd, pc] = changeText(b.throughput_tok_s, o.throughput_tok_s, false);
  setKpi("tput", o.throughput_tok_s, 0, pd, pc);
  const bd = o.max_batch !== b.max_batch ? [`${o.max_batch > b.max_batch ? "▲" : "▼"} from ${b.max_batch}`, o.max_batch > b.max_batch ? "good" : "bad"] : ["same as baseline", ""];
  setKpi("batch", o.max_batch, 0, ...bd);
  const fitEl = $(".kpi[data-k=fit] .num");
  setKpi("fit", o.fits ? "YES" : "NO", 0, b.fits === o.fits ? `baseline: ${b.fits ? "yes" : "no"}` : "baseline did NOT fit", b.fits === o.fits ? "" : "good");
  fitEl.className = "num " + (o.fits ? "yes" : "no");
}

function renderFlow(r) {
  const s = r.summary, ic = r.input_config;
  const sel = new Set(s.selected_optimizations);
  const byName = Object.fromEntries(r.candidates.map((c) => [c.name, c]));
  $("#st-in").innerHTML = `<b>${s.model}</b> on <b>${s.hardware.replace(/NVIDIA (GeForce )?/, "").replace(/ \d+GB.*/, "")}</b><br>b=${ic.batch_size} · ${fmtInt(ic.sequence_length)}→${fmtInt(ic.generation_length)} tok`;
  $("#st-p1").innerHTML = `<b>${(r.p1_total_flops / 1e12).toFixed(1)}</b> TFLOP · AI <b>${r.p1_arithmetic_intensity}</b>`;
  $("#st-p2").innerHTML = `KV pressure <b style="color:${r.p2_kv_pressure > ic.pressure_threshold ? "var(--warn)" : "inherit"}">${pct(r.p2_kv_pressure)}</b> (trigger ${pct(ic.pressure_threshold)})`;
  $("#st-p3").innerHTML = r.p3_fusion_best_level ? `best fusion <b>L${r.p3_fusion_best_level}</b>${r.p3_autotune_chosen ? ` · tile <b>${r.p3_autotune_chosen.block_m}×${r.p3_autotune_chosen.block_n}</b>` : ""}` : "no kernel-level wins";
  $("#st-out").innerHTML = `<b>${sel.size}</b> pass${sel.size === 1 ? "" : "es"} on<br><b>−${fmt(s.total_memory_saved_gb, 1)}</b> GB · <b>−${fmt(s.total_latency_saved_ms, 2)}</b> ms`;

  for (const phase of ["P1", "P2", "P3"]) {
    const box = $(`.passes[data-phase=${phase}]`);
    box.innerHTML = "";
    PASS_CATALOG[phase].forEach((p) => {
      const c = byName[p.name];
      const b = document.createElement("button");
      const status = phase === "P1" ? "selected" : sel.has(p.name) ? "selected" : c ? "proposed" : "idle";
      b.className = `pass ${status}` + (activePass === p.name ? " active" : "");
      b.innerHTML = `<span>${p.label}</span><span class="st">${status === "selected" ? "✓" : status === "proposed" ? "·" : "–"}</span>`;
      b.onclick = () => { activePass = activePass === p.name ? null : p.name; renderFlow(r); };
      box.appendChild(b);
    });
  }
  renderPassDetail(r, byName, sel);

  // Pulse wave through the pipeline
  const stages = $$(".stage"), wires = $$(".wire");
  stages.forEach((st, i) => setTimeout(() => {
    st.classList.add("lit"); wires[i]?.classList.add("hot");
    setTimeout(() => { st.classList.remove("lit"); wires[i]?.classList.remove("hot"); }, 650);
  }, i * 130));
}

function renderPassDetail(r, byName, sel) {
  const box = $("#pass-detail");
  if (!activePass) { box.hidden = true; return; }
  box.hidden = false;
  if (activePass === "roofline") {
    const ops = r.p1_op_breakdown;
    const top = ops.reduce((a, b) => (a.pct_time > b.pct_time ? a : b), ops[0]);
    box.innerHTML = `<h4>roofline analysis · always runs</h4>
      <p>Builds the per-layer IR (${ops.map((o) => o.operation).join(" → ")}) and bounds each op by compute or memory bandwidth. ${top ? `${top.operation} dominates at ${top.pct_time}% of time and is ${top.bound.replace("_", "-").toLowerCase()}.` : ""}</p>
      <div class="pd-stats"><div><span>params </span><b>${(r.p1_parameter_count / 1e9).toFixed(2)}B</b></div><div><span>weights </span><b>${r.p1_weights_gb} GB</b></div><div><span>KV </span><b>${r.p1_kv_cache_gb} GB</b></div><div><span>est. latency </span><b>${fmt(r.p1_estimated_latency_ms, 1)} ms</b></div></div>`;
    return;
  }
  const c = byName[activePass];
  if (!c) {
    box.innerHTML = `<h4>${activePass} · not proposed</h4><p>${(NOT_PROPOSED_WHY[activePass] || (() => ""))(r)}</p>`;
    return;
  }
  const chosen = sel.has(c.name);
  box.innerHTML = `<h4>${c.name} · ${chosen ? '<span style="color:var(--accent-2)">selected</span>' : '<span style="color:var(--muted)">proposed, not selected</span>'}<span class="risk ${c.accuracy_risk}">${c.accuracy_risk} risk</span></h4>
    <p>${c.reason}</p>
    <div class="pd-stats"><div><span>memory Δ </span><b>${c.memory_delta_gb > 0 ? "+" : ""}${fmt(c.memory_delta_gb, 3)} GB</b></div><div><span>latency Δ </span><b>${c.latency_delta_ms > 0 ? "+" : ""}${fmt(c.latency_delta_ms, 3)} ms</b></div><div><span>phase </span><b>${c.phase} · ${c.pass_name}</b></div></div>`;
}

function renderMemory(r) {
  const b = r.p2_baseline_metrics, o = r.p2_optimized_metrics;
  const cap = usableGib();
  const scale = Math.max(cap, b.total_gib, o.total_gib) * 1.06;
  [["base", b], ["opt", o]].forEach(([k, m]) => {
    const row = $(`.mrow[data-r=${k}]`);
    $("i.w", row).style.width = (m.weights_gib / scale) * 100 + "%";
    $("i.kv", row).style.width = (m.kv_gib / scale) * 100 + "%";
    $(".cap", row).style.left = (cap / scale) * 100 + "%";
    $(".mval", row).textContent = `${fmt(m.total_gib, 1)} GiB`;
    row.classList.toggle("over", !m.fits);
  });
  $("#mem-sub").textContent = `usable ${fmt(cap, 1)} GiB of ${gpuPreset().memory_gb} GB`;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------
function chartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = "#8a93a8";
  Chart.defaults.font.family = "Inter, system-ui, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = "rgba(148,163,184,.1)";
  Chart.defaults.animation.duration = 650;
  Chart.defaults.animation.easing = "easeOutCubic";
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
  Chart.defaults.plugins.tooltip.backgroundColor = "#111521";
  Chart.defaults.plugins.tooltip.borderColor = "rgba(148,163,184,.25)";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
}

function upsert(id, config) {
  if (!window.Chart) return null;
  if (charts[id]) {
    const ch = charts[id];
    ch.data.labels = config.data.labels;
    config.data.datasets.forEach((ds, i) => {
      if (ch.data.datasets[i]) Object.assign(ch.data.datasets[i], ds);
      else ch.data.datasets.push(ds);
    });
    ch.data.datasets.length = config.data.datasets.length;
    if (config.options) Object.assign(ch.options, config.options);
    ch.update();
    return ch;
  }
  charts[id] = new Chart($("#" + id), config);
  return charts[id];
}

const OP_COLORS = ["#818cf8", "#22d3ee", "#a78bfa", "#34d399", "#f472b6", "#fbbf24", "#60a5fa"];

function renderOps(r) {
  const ops = r.p1_op_breakdown;
  upsert("ch-ops", {
    type: "doughnut",
    data: {
      labels: ops.map((o) => o.operation),
      datasets: [{
        data: ops.map((o) => o.pct_time),
        backgroundColor: ops.map((_, i) => OP_COLORS[i % OP_COLORS.length]),
        borderColor: "#0b0e16", borderWidth: 2, hoverOffset: 8,
      }],
    },
    options: {
      maintainAspectRatio: false, cutout: "62%",
      plugins: {
        legend: { position: "right" },
        tooltip: { callbacks: { label: (c) => { const o = ops[c.dataIndex]; return ` ${o.operation}: ${o.pct_time}% · ${o.time_fmt} · ${o.bound.replace("_", "-").toLowerCase()}`; } } },
      },
    },
  });
}

function renderRoofline(r) {
  const g = gpuPreset();
  const peak = g.fp16_tflops || 312, bw = g.bandwidth_gbps || 2039;
  const ridge = (peak * 1e12) / (bw * 1e9);
  const xs = [0.1, ridge, 1e5];
  const roof = xs.map((x) => ({ x, y: Math.min(peak, (bw * 1e9 * x) / 1e12) }));
  const ops = r.p1_op_breakdown.filter((o) => o.time_s > 0);
  const pts = ops.map((o) => ({ x: Math.max(0.1, o.arithmetic_intensity), y: o.flops / o.time_s / 1e12, op: o }));
  $("#roof-sub").textContent = `ridge ${ridge.toFixed(0)} FLOP/B · peak ${peak} TFLOP/s`;
  upsert("ch-roof", {
    type: "scatter",
    data: {
      datasets: [
        { label: "hardware roof", data: roof, showLine: true, borderColor: "#fbbf24", borderWidth: 2, pointRadius: 0, fill: false, tension: 0 },
        {
          label: "ops", data: pts, pointRadius: 7, pointHoverRadius: 10,
          backgroundColor: ops.map((_, i) => OP_COLORS[i % OP_COLORS.length]),
          borderColor: "#0b0e16", borderWidth: 2,
        },
      ],
    },
    options: {
      maintainAspectRatio: false,
      scales: {
        x: { type: "logarithmic", min: 0.1, max: 1e5, ticks: { callback: pow10Tick, autoSkip: false, maxRotation: 0 }, title: { display: true, text: "arithmetic intensity (FLOP/byte)" } },
        y: { type: "logarithmic", min: 0.01, max: 1e4, ticks: { callback: pow10Tick }, title: { display: true, text: "attainable TFLOP/s" } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (c) => c.raw.op ? ` ${c.raw.op.operation}: AI ${c.raw.op.arithmetic_intensity}, ${c.raw.y.toFixed(1)} TFLOP/s (${c.raw.op.bound.replace("_", "-").toLowerCase()})` : ` roof: ${c.raw.y.toFixed(1)} TFLOP/s` } },
      },
    },
  });
}

function pow10Tick(v) {
  const l = Math.log10(v);
  return Math.abs(l - Math.round(l)) < 1e-9 ? (v >= 1000 ? `${v / 1000}k` : String(v)) : "";
}

const markerPlugin = {
  id: "marker",
  afterDatasetsDraw(chart, _args, opts) {
    if (opts.index == null || opts.index < 0) return;
    const x = chart.scales.x.getPixelForValue(opts.index);
    const { top, bottom } = chart.chartArea;
    const ctx = chart.ctx;
    ctx.save();
    ctx.strokeStyle = "rgba(52,211,153,.7)"; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, bottom); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = "#34d399"; ctx.font = "600 10px JetBrains Mono";
    ctx.textAlign = "center"; ctx.fillText("you are here", x, top - 4);
    ctx.restore();
  },
};

function renderSweep(d) {
  const pts = d.points;
  const labels = pts.map((p) => fmtInt(p.value));
  const cur = state[d.param];
  let idx = pts.findIndex((p) => p.value >= cur);
  if (idx < 0) idx = pts.length - 1;
  const fitColor = (arr) => arr.map((p) => (p.optimized?.fits ? "#34d399" : "#f87171"));
  const ch = upsert("ch-sweep", {
    type: "line",
    data: {
      labels,
      datasets: [
        { label: "throughput · baseline", data: pts.map((p) => p.baseline?.throughput_tok_s), borderColor: "#6366f1", backgroundColor: "rgba(99,102,241,.08)", fill: true, tension: .35, pointRadius: 2, yAxisID: "y" },
        { label: "throughput · optimized", data: pts.map((p) => p.optimized?.throughput_tok_s), borderColor: "#34d399", backgroundColor: "rgba(52,211,153,.1)", fill: true, tension: .35, pointRadius: 4, pointBackgroundColor: fitColor(pts), yAxisID: "y" },
        { label: "TPOT · optimized", data: pts.map((p) => p.optimized?.tpot_ms), borderColor: "#fbbf24", borderDash: [5, 4], tension: .35, pointRadius: 0, yAxisID: "y1" },
      ],
    },
    options: {
      maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      layout: { padding: { top: 14 } },
      scales: {
        y: { type: "logarithmic", ticks: { callback: pow10Tick }, title: { display: true, text: "tok/s" } },
        y1: { type: "logarithmic", position: "right", ticks: { callback: pow10Tick }, grid: { drawOnChartArea: false }, title: { display: true, text: "ms / token" } },
        x: { title: { display: true, text: { batch_size: "batch size", sequence_length: "prompt tokens", system_prompt_tokens: "shared prefix tokens" }[d.param] } },
      },
      plugins: {
        marker: { index: idx },
        tooltip: { callbacks: { afterBody: (items) => { const p = pts[items[0].dataIndex]; return [`fits: ${p.optimized?.fits ? "yes" : "NO"}`, `passes: ${p.selected.join(", ") || "none"}`]; } } },
      },
    },
    plugins: [markerPlugin],
  });
  if (ch) { ch.options.plugins.marker = { index: idx }; ch.update("none"); }
}

function renderFusion(r) {
  const lv = r.p3_fusion_levels || [];
  upsert("ch-fusion", {
    type: "bar",
    data: {
      labels: lv.map((l) => `L${l.level}`),
      datasets: [{
        label: "Δ latency (ms)", data: lv.map((l) => l.delta_ms),
        backgroundColor: lv.map((l) => (l.is_best ? "#34d399" : l.delta_ms < 0 ? "rgba(129,140,248,.7)" : "rgba(248,113,113,.6)")),
        borderRadius: 6,
      }],
    },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { title: (c) => lv[c[0].dataIndex].label, afterLabel: (c) => { const l = lv[c.dataIndex]; return `launches saved/layer: ${l.launches_saved_per_layer}\nmemory saved: ${l.mem_saved_gb} GB${l.is_best ? "\n★ chosen" : ""}`; } } } },
      scales: { y: { title: { display: true, text: "ms (negative = faster)" } } },
    },
  });
}

function renderTune(r) {
  const top = (r.p3_autotune_top || []).slice(0, 8);
  const ch = r.p3_autotune_chosen, base = r.p3_autotune_baseline;
  $("#tune-sub").textContent = ch ? `chosen ${ch.config}` + (base ? ` · ${(base.time_us_per_layer / ch.time_us_per_layer).toFixed(2)}× vs default` : "") : "not run";
  upsert("ch-tune", {
    type: "bar",
    data: {
      labels: top.map((t) => t.config.replace(/ s=\d/, "")),
      datasets: [{
        label: "µs / layer", data: top.map((t) => t.time_us_per_layer),
        backgroundColor: top.map((t) => (t.is_chosen ? "#34d399" : "rgba(34,211,238,.55)")),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y", maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { callbacks: { afterLabel: (c) => { const t = top[c.dataIndex]; return `${t.bound}-bound · ${t.smem_kb} KB smem · ${t.blocks_per_sm} blocks/SM`; } } } },
      scales: { x: { title: { display: true, text: "µs per layer (lower is better)" } }, y: { ticks: { font: { family: "JetBrains Mono", size: 10 } } } },
    },
  });
}

function renderSched(r) {
  const sp = r.p3_split_kv, pp = r.p3_persistent, box = $("#sched");
  const sel = new Set(r.summary.selected_optimizations);
  let html = "";
  if (sp) {
    const nSm = Math.round(sp.target / 2);
    const used = Math.min(nSm, Math.round(Math.min(1, sp.units / sp.target) * nSm));
    const usedSplit = Math.min(nSm, Math.round(Math.min(1, (sp.units * sp.splits) / sp.target) * nSm));
    const on = sel.has("split_kv");
    const cells = Array.from({ length: nSm }, (_, i) => `<i class="${i < used ? "on" : on && i < usedSplit ? "on2" : ""}"></i>`).join("");
    html += `<div class="sch"><div class="sch-h">split_kv <span class="verdict ${on ? "yes" : "no"}">${on ? "applied" : "rejected"}</span></div>
      <p>${sp.units} work blocks for ${nSm} SMs → ${sp.splits} split${sp.splits > 1 ? "s" : ""} · Δ ${fmt(sp.delta_ms, 3)} ms</p>
      <div class="sm-grid" title="SM occupancy: indigo = unsplit, green = added by splitting">${cells}</div></div>`;
  }
  if (pp) {
    const on = sel.has("persistent_kernel");
    html += `<div class="sch"><div class="sch-h">persistent_kernel <span class="verdict ${on ? "yes" : "no"}">${on ? "applied" : "rejected"}</span></div>
      <p>${pp.launches} kernel launches = ${pct(pp.launch_share)} of the step · sync cost ${fmt(pp.sync_ms, 3)} ms · Δ ${fmt(pp.delta_ms, 3)} ms</p></div>`;
  }
  box.innerHTML = html || `<div class="empty">Scheduling passes only apply to decode.</div>`;
}

function renderCandidates(r) {
  const sel = new Set(r.summary.selected_optimizations);
  let list = [...r.candidates];
  if (candFilter === "P2" || candFilter === "P3") list = list.filter((c) => c.phase === candFilter);
  if (candFilter === "sel") list = list.filter((c) => sel.has(c.name));
  list.sort((a, b) => a.latency_delta_ms - b.latency_delta_ms);
  const cls = (v) => (v < 0 ? "g" : v > 0 ? "b" : "z");
  $("#cands").innerHTML = list.length ? list.map((c, i) => `
    <div class="cand ${sel.has(c.name) ? "sel" : ""}" style="animation-delay:${i * 40}ms">
      <div class="ph ${c.phase}">${c.phase}</div>
      <div><div class="nm">${sel.has(c.name) ? "✓ " : ""}${c.name}<span class="risk ${c.accuracy_risk}">${c.accuracy_risk}</span></div><div class="ds">${c.reason}</div></div>
      <div class="dl"><div class="${cls(c.latency_delta_ms)}">${c.latency_delta_ms > 0 ? "+" : ""}${fmt(c.latency_delta_ms, 3)} ms</div><div class="${cls(c.memory_delta_gb)}">${c.memory_delta_gb > 0 ? "+" : ""}${fmt(c.memory_delta_gb, 2)} GB</div></div>
    </div>`).join("") : `<div class="empty">No candidates match this filter.</div>`;
}

// ---------------------------------------------------------------------------
// Compiler log (typed out)
// ---------------------------------------------------------------------------
let termTimer = null;
function renderTerm(r) {
  const s = r.summary, ic = r.input_config, sel = new Set(s.selected_optimizations);
  const L = [];
  L.push(["", `$ llmopt --model ${ic.model} --gpu ${ic.hardware} --batch ${ic.batch_size} --prompt ${ic.sequence_length} --gen ${ic.generation_length} --${ic.precision} --${ic.mode}`]);
  L.push(["t-p1", `[P1] IR built: ${r.p1_op_breakdown.length} ops/layer, ${(r.p1_parameter_count / 1e9).toFixed(2)}B params`]);
  L.push(["t-p1", `[P1] ${(r.p1_total_flops / 1e12).toFixed(1)} TFLOP, AI ${r.p1_arithmetic_intensity} FLOP/B, roofline ${fmt(r.p1_estimated_latency_ms, 1)} ms`]);
  L.push([r.p1_fits_in_memory ? "t-ok" : "t-warn", `[P1] memory ${r.p1_total_memory_gb} GB → ${r.p1_fits_in_memory ? "fits" : "DOES NOT FIT"}`]);
  L.push([r.p2_kv_pressure > ic.pressure_threshold ? "t-warn" : "t-p2", `[P2] KV pressure ${pct(r.p2_kv_pressure)} ${r.p2_kv_pressure > ic.pressure_threshold ? ">" : "≤"} trigger ${pct(ic.pressure_threshold)}`]);
  r.candidates.filter((c) => c.phase === "P2").forEach((c) => L.push([sel.has(c.name) ? "t-ok" : "t-no", `[P2] ${sel.has(c.name) ? "✓" : "✗"} ${c.name}: ${fmt(c.memory_delta_gb, 2)} GB, ${fmt(c.latency_delta_ms, 2)} ms`]));
  (r.p2_plan_notes || []).forEach((n) => L.push(["t-warn", `[P2] note: ${n}`]));
  if (r.p3_fusion_levels?.length) L.push(["t-p3", `[P3] fusion: evaluated ${r.p3_fusion_levels.length} levels, best L${r.p3_fusion_best_level}`]);
  r.candidates.filter((c) => c.phase === "P3").forEach((c) => L.push([sel.has(c.name) ? "t-ok" : "t-no", `[P3] ${sel.has(c.name) ? "✓" : "✗"} ${c.name}: ${fmt(c.latency_delta_ms, 3)} ms`]));
  if (r.p3_autotune_chosen) L.push(["t-p3", `[P3] autotune: ${r.p3_autotune_top.length} configs ranked → ${r.p3_autotune_chosen.config}`]);
  L.push(["t-ok", `[plan] ${sel.size} passes: ${[...sel].join(", ") || "(none)"}`]);
  L.push(["t-ok", `[plan] saves ${fmt(s.total_memory_saved_gb, 2)} GB, ${fmt(s.total_latency_saved_ms, 2)} ms/step`]);

  clearInterval(termTimer);
  const term = $("#term");
  term.innerHTML = "";
  let i = 0;
  termTimer = setInterval(() => {
    $(".caret", term)?.remove();
    if (i >= L.length) { clearInterval(termTimer); term.insertAdjacentHTML("beforeend", '<span class="caret"></span>'); return; }
    const [c, t] = L[i++];
    const span = document.createElement("span");
    span.className = c; span.textContent = t + "\n";
    term.appendChild(span);
    term.insertAdjacentHTML("beforeend", '<span class="caret"></span>');
    term.scrollTop = term.scrollHeight;
  }, 45);
}

// ---------------------------------------------------------------------------
// Deploy view
// ---------------------------------------------------------------------------
function renderDeploy(r) {
  const sel = new Set(r.summary.selected_optimizations), ic = r.input_config, o = r.p2_optimized_metrics;
  const hf = HF_IDS[ic.model];
  const maxLen = ic.sequence_length + ic.generation_length;
  const maxSeqs = Math.max(1, Math.min(o.max_batch || ic.batch_size, 1024));
  const dtype = { fp32: "float32", fp16: "float16", bf16: "bfloat16", int8: "float16", int4: "float16" }[ic.precision];
  const lines = [], why = [];
  const flag = (f, reason) => { lines.push(f); if (reason) why.push(reason); };

  if (engine === "vllm") {
    flag(`vllm serve ${hf}`);
    flag(`  --dtype ${dtype}`);
    flag(`  --max-model-len ${maxLen}`, `<b>--max-model-len ${maxLen}</b>: prompt ${ic.sequence_length} + generation ${ic.generation_length}`);
    flag(`  --max-num-seqs ${maxSeqs}`, `<b>--max-num-seqs ${maxSeqs}</b>: P2's max concurrent batch that fits in memory`);
    flag(`  --gpu-memory-utilization ${USABLE_FRACTION}`, `<b>--gpu-memory-utilization</b>: matches the planner's usable-memory assumption`);
    if (ic.precision === "int8") flag(`  --quantization fp8`, `<b>--quantization fp8</b>: 8-bit weights, as you planned for`);
    if (ic.precision === "int4") flag(`  --quantization awq`, `<b>--quantization awq</b>: 4-bit weights (needs an AWQ checkpoint)`);
    if (sel.has("int8_kv")) flag(`  --kv-cache-dtype fp8`, `<b>--kv-cache-dtype fp8</b>: int8_kv was selected; vLLM's 8-bit KV cache gives the same ~2× KV saving`);
    if (sel.has("prefix_share")) flag(`  --enable-prefix-caching`, `<b>--enable-prefix-caching</b>: prefix_share found shared prompt prefixes to reuse`);
    if (sel.has("fusion")) why.push(`<b>fusion</b>: vLLM's FlashAttention backend already fuses QKV→softmax→output; nothing to add`);
    if (sel.has("split_kv")) why.push(`<b>split_kv</b>: done automatically by Flash-Decoding for small batches`);
    if (sel.has("persistent_kernel")) why.push(`<b>persistent_kernel</b>: CUDA graphs (default, don't pass --enforce-eager) remove most launch overhead`);
    lines.forEach((l, i) => { if (i < lines.length - 1) lines[i] = l + " \\"; });
  } else if (engine === "trt") {
    flag(`python convert_checkpoint.py --model_dir ./${hf.split("/")[1]} \\`);
    flag(`  --output_dir ./ckpt --dtype ${dtype}${sel.has("int8_kv") ? " --int8_kv_cache" : ""}${ic.precision === "int4" ? " --use_weight_only --weight_only_precision int4" : ic.precision === "int8" ? " --use_weight_only --weight_only_precision int8" : ""}`,
      sel.has("int8_kv") ? `<b>--int8_kv_cache</b>: int8_kv was selected` : null);
    flag(``);
    flag(`trtllm-build --checkpoint_dir ./ckpt --output_dir ./engine \\`);
    flag(`  --gemm_plugin ${dtype} \\`);
    flag(`  --max_batch_size ${maxSeqs} --max_input_len ${ic.sequence_length} --max_seq_len ${maxLen} \\`, `<b>--max_batch_size ${maxSeqs}</b>: P2's max concurrent batch`);
    flag(`  --context_fmha enable${sel.has("prefix_share") ? " --use_paged_context_fmha enable" : ""}`, sel.has("prefix_share") ? `<b>--use_paged_context_fmha</b>: required for KV-cache reuse (prefix_share). Also set <b>enable_kv_cache_reuse</b> at runtime.` : null);
    if (sel.has("fusion")) why.push(`<b>fusion</b>: covered by the fused MHA (context_fmha) plugin`);
    if (sel.has("split_kv")) why.push(`<b>split_kv</b>: TRT-LLM's multi-block decode mode does this`);
    if (r.p3_autotune_chosen) why.push(`<b>autotune</b>: trtllm-build runs its own tactic search; the chosen ${r.p3_autotune_chosen.config} tile is a good starting point for custom kernels`);
  } else {
    flag(`text-generation-launcher --model-id ${hf} \\`);
    flag(`  --dtype ${dtype} \\`);
    flag(`  --max-input-tokens ${ic.sequence_length} --max-total-tokens ${maxLen} \\`, `<b>--max-total-tokens ${maxLen}</b>: prompt + generation`);
    flag(`  --max-batch-size ${maxSeqs}${sel.has("int8_kv") ? " \\" : ""}`, `<b>--max-batch-size ${maxSeqs}</b>: P2's max concurrent batch`);
    if (sel.has("int8_kv")) flag(`  --kv-cache-dtype fp8_e5m2`, `<b>--kv-cache-dtype</b>: int8_kv was selected`);
    if (sel.has("prefix_share")) why.push(`<b>prefix_share</b>: TGI v3 caches prefixes automatically`);
    if (ic.precision === "int4" || ic.precision === "int8") why.push(`<b>--quantize</b>: add <code>--quantize awq</code> or <code>eetq</code> for ${ic.precision} weights`);
  }
  $("#cmd").textContent = lines.join("\n");
  $("#cmd-why").innerHTML = why.map((w) => `<li>${w}</li>`).join("") || "<li>Defaults are fine for this workload. No optimization changes the launch flags.</li>";

  $("#plan-chips").innerHTML =
    [...sel].map((s, i) => `<span class="plan-chip" style="animation-delay:${i * 60}ms">✓ ${s}</span>`).join("") +
    `<span class="plan-chip ctx">${r.summary.model} · ${r.summary.hardware.replace("NVIDIA ", "")} · ${ic.precision.toUpperCase()} · b=${ic.batch_size}</span>`;

  const b = r.p2_baseline_metrics;
  $("#cap-grid").innerHTML = [
    ["Time to first token", fmt(o.ttft_ms, 0), "ms", b.ttft_ms],
    ["Time per output token", fmt(o.tpot_ms, 2), "ms", b.tpot_ms],
    ["Aggregate throughput", fmt(o.throughput_tok_s, 0), "tok/s", b.throughput_tok_s],
    ["Max concurrent requests", o.max_batch, "", b.max_batch],
  ].map(([l, v, u, bv]) => `<div class="capc"><span>${l}</span><b>${v}</b> <small>${u}</small><br><small>baseline ${fmt(bv, l.startsWith("Time per") ? 2 : 0)}</small></div>`).join("");
  renderFleet(r);
}

function renderFleet(r) {
  const rps = Number($("#fleet-rps").value);
  const gen = Math.max(1, r.input_config.generation_length);
  $("#fleet-out").textContent = `${rps} req/s`;
  const need = rps * gen;
  const nOpt = Math.max(1, Math.ceil(need / Math.max(1, r.p2_optimized_metrics.throughput_tok_s)));
  const nBase = Math.max(1, Math.ceil(need / Math.max(1, r.p2_baseline_metrics.throughput_tok_s)));
  const shown = Math.min(nOpt, 60);
  $("#fleet-res").innerHTML = `${fmtInt(need)} tok/s needed → <b>${nOpt}</b> GPU${nOpt > 1 ? "s" : ""} optimized vs <b style="color:var(--muted)">${nBase}</b> baseline
    <div class="gpu-row">${Array.from({ length: shown }, (_, i) => `<i style="animation-delay:${i * 15}ms"></i>`).join("")}${nOpt > shown ? ` <span class="muted">+${nOpt - shown}</span>` : ""}</div>`;
}

// ---------------------------------------------------------------------------
// Misc
// ---------------------------------------------------------------------------
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg; t.hidden = false;
  clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 2600);
}

function backgroundField() {
  const cv = $("#bg-canvas"), ctx = cv.getContext("2d");
  if (matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  let W, H, P;
  const resize = () => {
    W = cv.width = innerWidth * devicePixelRatio; H = cv.height = innerHeight * devicePixelRatio;
    cv.style.width = innerWidth + "px"; cv.style.height = innerHeight + "px";
    P = Array.from({ length: Math.min(70, Math.round(innerWidth / 22)) }, () => ({
      x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .25, vy: (Math.random() - .5) * .25,
    }));
  };
  resize(); addEventListener("resize", resize);
  const D = 140 * devicePixelRatio;
  (function frame() {
    ctx.clearRect(0, 0, W, H);
    for (const p of P) {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0 || p.x > W) p.vx *= -1;
      if (p.y < 0 || p.y > H) p.vy *= -1;
    }
    for (let i = 0; i < P.length; i++) {
      for (let j = i + 1; j < P.length; j++) {
        const dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, d = Math.hypot(dx, dy);
        if (d < D) {
          ctx.strokeStyle = `rgba(129,140,248,${0.12 * (1 - d / D)})`;
          ctx.beginPath(); ctx.moveTo(P[i].x, P[i].y); ctx.lineTo(P[j].x, P[j].y); ctx.stroke();
        }
      }
      ctx.fillStyle = "rgba(129,140,248,.35)";
      ctx.fillRect(P[i].x, P[i].y, 2 * devicePixelRatio, 2 * devicePixelRatio);
    }
    requestAnimationFrame(frame);
  })();
}

async function init() {
  chartDefaults();
  backgroundField();
  initControls();
  try {
    presets = await (await fetch(`${API}/api/presets`)).json();
  } catch {
    setPill("err", "backend offline");
    toast("Can't reach the backend. Start it with: python integration/server.py");
    return;
  }
  renderGpus();
  syncControls();
  runOptimize();
  // writeHash uses replaceState (no event), so this only fires for pasted links / back-forward.
  addEventListener("hashchange", () => {
    Object.assign(state, DEFAULTS, readHash());
    clearScenario(); syncControls(true); runOptimize();
  });
}

init();
