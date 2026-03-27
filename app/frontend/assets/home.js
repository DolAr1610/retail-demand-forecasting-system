import { APP_CONFIG, loadAppConfig, fmt } from "./app.js";

function $(id) {
  return document.getElementById(id);
}

function firstMetricRow(rows) {
  if (!Array.isArray(rows) || !rows.length) return null;
  return rows[0];
}

function renderHomeMetrics() {
  const rows = APP_CONFIG.metrics || [];
  const best = firstMetricRow(rows);

  $("metricRmse").textContent = best ? fmt(best.RMSE ?? best.rmse_log, 4) : "—";
  $("metricR2").textContent = best ? fmt(best.R2 ?? best.r2, 4) : "—";
  $("metricWape").textContent = best ? fmt(best.WAPE ?? best.wape, 4) : "—";
  $("metricMae").textContent = best ? fmt(best.MAE ?? best.mae, 4) : "—";

  $("metricRmseNote").textContent = best ? `Best model: ${best.model}` : "No metrics loaded.";
  $("metricR2Note").textContent = best ? `Model quality on test split.` : "—";
  $("metricWapeNote").textContent = best ? `Weighted absolute percentage error.` : "—";
  $("metricMaeNote").textContent = best ? `Average absolute error in sales units.` : "—";
}

function renderModelList() {
  const list = $("modelList");
  if (!list) return;

  list.innerHTML = "";

  const models = APP_CONFIG.models || [];
  if (!models.length) {
    const li = document.createElement("li");
    li.textContent = "No models loaded.";
    list.appendChild(li);
    return;
  }

  models.forEach((m, idx) => {
    const li = document.createElement("li");
    li.innerHTML =
      idx === 0
        ? `<b>${m.name}</b> — available for prediction in the current artifact package.`
        : `<b>${m.name}</b> — available for comparison / selection.`;
    list.appendChild(li);
  });
}

async function initHome() {
  await loadAppConfig();
  renderHomeMetrics();
  renderModelList();
}

initHome();