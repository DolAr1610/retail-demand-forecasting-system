import { APP_CONFIG } from "./app.js";

function $(id) {
  return document.getElementById(id);
}

function toISO(d) {
  const pad = (x) => String(x).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

function addDays(iso, days) {
  const d = new Date(iso + "T00:00:00");
  d.setDate(d.getDate() + Number(days));
  return toISO(d);
}

function seedValue(store, item, horizon) {
  // deterministic stub number for repeatable screenshots
  const s = Number(store || 1);
  const i = Number(item || 1);
  const h = Number(horizon || 1);
  const base = (s * 17 + i * 0.013 + h * 3.7) % 1;
  const pred = 15 + base * 40; // 15..55
  return Math.round(pred * 10) / 10;
}

function init() {
  const dateHint = $("dateHint");
  if (dateHint) {
    dateHint.textContent = `${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel}`;
  }

  const inpDate = $("inpDate");
  if (inpDate) {
    inpDate.min = APP_CONFIG.dataset.minDate;
    inpDate.max = APP_CONFIG.dataset.maxDate;
    inpDate.value = APP_CONFIG.dataset.maxDate;
  }

  const inpStore = $("inpStore");
  if (inpStore) inpStore.value = String(APP_CONFIG.subset.store);

  const inpModel = $("inpModel");
  if (inpModel) {
    inpModel.innerHTML = APP_CONFIG.models
      .map((m) => `<option value="${m.id}">${m.name}</option>`)
      .join("");
    inpModel.value = "gbt";
  }

  $("predictForm")?.addEventListener("submit", (e) => {
    e.preventDefault();

    const date = $("inpDate")?.value || APP_CONFIG.dataset.maxDate;
    const store = $("inpStore")?.value || APP_CONFIG.subset.store;
    const item = $("inpItem")?.value || 1;
    const horizon = $("inpHorizon")?.value || 1;
    const modelId = $("inpModel")?.value || "gbt";

    const modelName = APP_CONFIG.models.find((m) => m.id === modelId)?.name ?? modelId;

    const targetDate = addDays(date, horizon);
    const pred = seedValue(store, item, horizon);

    const bandLo = Math.max(0, Math.round((pred * 0.82) * 10) / 10);
    const bandHi = Math.round((pred * 1.18) * 10) / 10;

    $("outEmpty")?.classList.add("hidden");
    $("outCard")?.classList.remove("hidden");

    $("outModel").textContent = modelName;
    $("outTargetDate").textContent = targetDate;
    $("outPred").textContent = `${pred} units`;
    $("outBand").textContent = `${bandLo} – ${bandHi} units`;
  });
}

init();
