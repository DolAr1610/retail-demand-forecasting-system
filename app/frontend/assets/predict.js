import { API, APP_CONFIG, fmtNum, setText, setSelectOptions } from "./app.js";

function isSelect(el) { return el && el.tagName === "SELECT"; }

async function apiGet(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`GET ${url} -> ${r.status}`);
  return r.json();
}
async function apiPost(url, body) {
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const txt = await r.text();
  let data = null;
  try { data = txt ? JSON.parse(txt) : null; } catch {}
  if (!r.ok) throw new Error((data && data.detail) ? data.detail : `POST ${url} -> ${r.status}: ${txt}`);
  return data;
}

function pick(el, fallback = "") {
  if (!el) return fallback;
  return (el.value ?? fallback).toString().trim();
}
function pickInt(el, fallback = 0) {
  const v = parseInt(pick(el, ""), 10);
  return Number.isFinite(v) ? v : fallback;
}

function parseRecentSales(text) {
  const s = (text ?? "").toString().trim();
  if (!s) return null;
  const arr = s.split(",").map(x => x.trim()).filter(Boolean).map(Number).filter(Number.isFinite);
  return arr.length ? arr : null;
}

function showError(msg) {
  const box = document.getElementById("errorBox");
  if (!box) { alert(msg); return; }
  box.textContent = msg;
  box.style.display = "block";
}
function clearError() {
  const box = document.getElementById("errorBox");
  if (!box) return;
  box.textContent = "";
  box.style.display = "none";
}

async function init() {
  clearError();

  const inpDate = document.getElementById("inpDate");
  const inpStore = document.getElementById("inpStore");
  const selModel = document.getElementById("selModel");

  const ctx = await apiGet(`${API}/predict/context`);

  setText("ctxRange", `${ctx.min_date} → ${ctx.max_date}`);

  if (inpDate) {
    inpDate.min = ctx.min_date;
    inpDate.max = ctx.max_date;
    inpDate.value = ctx.max_date;
  }

  if (inpStore) inpStore.value = (ctx.store_nbr ?? APP_CONFIG.store_nbr ?? "").toString();

  if (isSelect(selModel) && Array.isArray(ctx.models) && ctx.models.length) {
    setSelectOptions(selModel, ctx.models.map(m => ({ value: m.id, label: m.name })));
    selModel.value = "best_model";
  }
}

async function onSubmit(e) {
  e.preventDefault();
  clearError();

  const inpDate = document.getElementById("inpDate");
  const inpStore = document.getElementById("inpStore");
  const inpItem = document.getElementById("inpItem");
  const inpHorizon = document.getElementById("inpHorizon");
  const selModel = document.getElementById("selModel");
  const inpRecentSales = document.getElementById("inpRecentSales"); // додай textarea в html

  const asOf = pick(inpDate);
  const store = pickInt(inpStore, APP_CONFIG.store_nbr);
  const item = pickInt(inpItem, 0);
  const horizon = pickInt(inpHorizon, 1);
  const model_id = pick(selModel, "best_model");
  const recent_sales = parseRecentSales(pick(inpRecentSales, ""));

  if (!asOf) return showError("Вкажи дату (as_of_date).");
  if (!store) return showError("Вкажи store_nbr.");
  if (!item) return showError("Вкажи item_nbr (число).");

  const payload = { date: asOf, store_nbr: store, item_nbr: item, horizon, model_id };
  if (recent_sales) payload.recent_sales = recent_sales;

  try {
    const resp = await apiPost(`${API}/predict/point`, payload);

    setText("outAsOf", asOf);
    setText("outTarget", resp.target_date);
    setText("outModel", resp.model_id);
    setText("outPredLog", fmtNum(resp.pred_log, 6));
    setText("outPredSales", fmtNum(resp.pred_sales, 4));

    const lo = resp.band_low;
    const hi = resp.band_high;
    setText("outBand", (lo != null && hi != null) ? `${fmtNum(lo, 4)} .. ${fmtNum(hi, 4)}` : "—");
  } catch (err) {
    showError(err.message || String(err));
  }
}

document.addEventListener("DOMContentLoaded", () => {
  init().catch(err => showError(err.message || String(err)));
  const form = document.getElementById("frmPredict");
  if (form) form.addEventListener("submit", onSubmit);
});
