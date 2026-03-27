import { APP_CONFIG, loadAppConfig } from "./app.js";

function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function normalizeRows(metricsPayload) {
  const rows = metricsPayload?.metrics?.test_metrics || [];
  return rows.map((r) => ({
    model: r.model,
    rmse_log: r.RMSE,
    r2: r.R2,
    wape: r.WAPE,
    note: "",
  }));
}

function renderTable(rows) {
  const tbody = document.querySelector("#metricsTable tbody");
  if (!tbody) return;
  tbody.innerHTML = "";

  rows.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><b>${r.model}</b></td>
      <td>${fmt(r.rmse_log, 4)}</td>
      <td>${fmt(r.r2, 4)}</td>
      <td>${fmt(r.wape, 4)}</td>
      <td class="small">${r.note || ""}</td>
    `;
    tbody.appendChild(tr);
  });
}

function renderMiniBars(containerId, rows, valueKey, label) {
  const el = document.getElementById(containerId);
  if (!el || !rows.length) return;

  const max = Math.max(...rows.map((r) => Number(r[valueKey]) || 0), 1);
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
    bar.style.width = `${((Number(r[valueKey]) || 0) / max) * 100}%`;

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
  foot.textContent = `${label} • dataset window: ${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel} • default store: ${APP_CONFIG.subset.store}`;
  el.appendChild(foot);
}

async function refresh() {
  await loadAppConfig();
  const res = await fetch("/metrics/global", { cache: "no-store" });
  if (!res.ok) return;
  const payload = await res.json();
  const rows = normalizeRows(payload);
  renderTable(rows);
  renderMiniBars("rmseChart", rows, "rmse_log", "RMSE");
  renderMiniBars("wapeChart", rows, "wape", "WAPE");
}

document.getElementById("btnRefresh")?.addEventListener("click", refresh);
refresh();