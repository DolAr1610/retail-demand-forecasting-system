import { APP_CONFIG } from "./app.js";

const STATIC_METRICS = [
  {
    modelId: "persist",
    model: "Persistence baseline",
    rmse_log: 0.622656,
    r2: 0.6100,
    wape: 0.4277,
    note: "Naive reference: next value ≈ last observed value.",
  },
  {
    modelId: "roll7",
    model: "Rolling Mean (7 days)",
    rmse_log: 0.535552,
    r2: 0.7115,
    wape: 0.3529,
    note: "Smoothed baseline: stable and easy to interpret.",
  },
  {
    modelId: "gbt",
    model: "Gradient-Boosted Trees",
    rmse_log: 0.437693,
    r2: 0.8073,
    wape: 0.2555,
    note: "Strong non-linear model on structured features.",
  },
  {
    modelId: "rf",
    model: "Random Forest",
    rmse_log: 0.444813,
    r2: 0.8010,
    wape: 0.2625,
    note: "Robust ensemble model; stable across noisy segments.",
  },
];

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function renderTable(rows) {
  const tbody = document.querySelector("#metricsTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${r.model}</b></td>
      <td>${fmt(r.rmse_log, 6)}</td>
      <td>${fmt(r.r2, 4)}</td>
      <td>${fmt(r.wape, 4)}</td>
      <td class="small">${r.note || ""}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderMiniBars(containerId, rows, valueKey, label) {
  const el = document.getElementById(containerId);
  if (!el) return;

  const max = Math.max(...rows.map((r) => r[valueKey]));
  el.innerHTML = "";

  rows.forEach((r) => {
    const line = document.createElement("div");
    line.className = "barline";

    const name = document.createElement("div");
    name.className = "barname";
    name.textContent = r.model;

    const barWrap = document.createElement("div");
    barWrap.className = "barwrap";

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.width = `${(r[valueKey] / max) * 100}%`;

    const val = document.createElement("div");
    val.className = "barval";
    val.textContent = fmt(r[valueKey], 4);

    barWrap.appendChild(bar);
    line.appendChild(name);
    line.appendChild(barWrap);
    line.appendChild(val);

    el.appendChild(line);
  });

  const foot = document.createElement("div");
  foot.className = "small muted";
  foot.style.marginTop = "10px";
  foot.textContent = `${label} • dataset window: ${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel} • store: ${APP_CONFIG.subset.store}`;
  el.appendChild(foot);
}

async function tryFetchMetrics() {
  // Optional backend endpoint — if it exists, we use it.
  // Expected shape: [{model, rmse_log, r2, wape, note}, ...]
  const endpoints = ["/api/metrics", "/metrics", "/api/analytics/metrics"];
  for (const url of endpoints) {
    try {
      const res = await fetch(url, { cache: "no-store" });
      if (!res.ok) continue;
      const data = await res.json();
      if (Array.isArray(data) && data.length) return data;
    } catch (_) {
      // ignore
    }
  }
  return null;
}

async function refresh() {
  const live = await tryFetchMetrics();
  const rows = live || STATIC_METRICS;

  renderTable(rows);
  renderMiniBars("rmseChart", rows, "rmse_log", "RMSE (log1p)");
  renderMiniBars("wapeChart", rows, "wape", "WAPE (sales)");
}

document.getElementById("btnRefresh")?.addEventListener("click", refresh);
refresh();
