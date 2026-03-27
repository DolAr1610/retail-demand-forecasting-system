import { APP_CONFIG, loadAppConfig } from "./app.js";

function $(id) {
  return document.getElementById(id);
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return await res.json();
}

function fillSelect(selectEl, values, formatter = (v) => ({ value: v, label: String(v) })) {
  if (!selectEl) return;
  selectEl.innerHTML = "";

  values.forEach((v) => {
    const opt = document.createElement("option");
    const mapped = formatter(v);
    opt.value = mapped.value;
    opt.textContent = mapped.label;
    selectEl.appendChild(opt);
  });
}

let validItemsForStore = new Set();
let aliasToItemMap = new Map();
let itemToAliasMap = new Map();

function normalizeAlias(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, " ");
}

function resolveItemInput(rawValue) {
  const raw = String(rawValue || "").trim();
  if (!raw) return null;

  const normalized = normalizeAlias(raw);

  if (aliasToItemMap.has(normalized)) {
    return String(aliasToItemMap.get(normalized));
  }

  const bracketMatch = raw.match(/\((\d+)\)\s*$/);
  if (bracketMatch) {
    const candidate = bracketMatch[1];
    if (validItemsForStore.has(candidate)) {
      return candidate;
    }
  }

  if (/^\d+$/.test(raw) && validItemsForStore.has(raw)) {
    return raw;
  }

  return null;
}

async function loadStores() {
  const storeSelect = $("inpStore");
  if (!storeSelect) return;

  const stores = await fetchJson("/data/stores");
  fillSelect(storeSelect, stores, (v) => ({
    value: String(v),
    label: `Store ${v}`,
  }));

  const defaultStore = String(APP_CONFIG.subset.store || stores[0] || "");
  if (defaultStore && stores.map(String).includes(defaultStore)) {
    storeSelect.value = defaultStore;
  } else if (stores.length) {
    storeSelect.value = String(stores[0]);
  }
}

async function loadItemsForStore(storeId) {
  const itemHint = $("itemHint");
  const itemInput = $("inpItem");

  validItemsForStore = new Set();
  aliasToItemMap = new Map();
  itemToAliasMap = new Map();

  if (!storeId) {
    if (itemHint) itemHint.textContent = "Choose store first.";
    return;
  }

  const rows = await fetchJson(`/data/items-labeled?store=${encodeURIComponent(storeId)}`);

  rows.forEach((row) => {
    const itemNbr = String(row.item_nbr);
    const displayName = String(row.display_name || `Item ${itemNbr}`);

    validItemsForStore.add(itemNbr);
    aliasToItemMap.set(normalizeAlias(displayName), itemNbr);
    itemToAliasMap.set(itemNbr, displayName);
    aliasToItemMap.set(normalizeAlias(`${displayName} (${itemNbr})`), itemNbr);
  });

  if (itemHint) {
    itemHint.textContent =
      `Loaded ${rows.length} trained items for Store ${storeId}. ` +
      `You can enter Item 1 or numeric item_nbr.`;
  }

  if (itemInput && itemInput.value) {
    const resolved = resolveItemInput(itemInput.value);
    if (!resolved) {
      itemInput.value = "";
    }
  }
}

function loadModels() {
  const modelSelect = $("inpModel");
  if (!modelSelect) return;

  const models = APP_CONFIG.models?.length
    ? APP_CONFIG.models
    : [{ id: "model", name: "Model" }];

  fillSelect(modelSelect, models, (m) => ({
    value: m.id,
    label: m.name,
  }));
}

function renderPrediction(payload) {
  const row = payload.rows?.[0];

  $("outEmpty")?.classList.add("hidden");
  $("outCard")?.classList.remove("hidden");

  if (!row) {
    $("outModel").textContent = payload.model_name || "Model";
    $("outTargetDate").textContent = payload.date_from || "—";
    $("outItemLabel").textContent = "—";
    $("outPred").textContent = "No forecast";
    $("outActual").textContent = "—";
    $("outAbsError").textContent = "—";
    return;
  }

  const alias = itemToAliasMap.get(String(row.item_nbr));

  $("outModel").textContent = payload.model_name || "Model";
  $("outTargetDate").textContent = row.date;
  $("outItemLabel").textContent = alias
    ? `${alias} (${row.item_nbr})`
    : String(row.item_nbr);

  $("outPred").textContent = `${Number(row.pred).toFixed(2)} units`;

  if (row.actual !== null && row.actual !== undefined) {
    $("outActual").textContent = `${Number(row.actual).toFixed(2)} units`;
    $("outAbsError").textContent =
      row.abs_error !== null && row.abs_error !== undefined
        ? `${Number(row.abs_error).toFixed(2)}`
        : "—";
  } else {
    $("outActual").textContent = "—";
    $("outAbsError").textContent = "—";
  }
}

async function init() {
  await loadAppConfig();

  const dateHint = $("dateHint");
  if (dateHint) {
    dateHint.textContent = `${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel}`;
  }

  const dateInput = $("inpDate");
  if (dateInput) {
    dateInput.min = APP_CONFIG.dataset.minDate;
    dateInput.max = APP_CONFIG.dataset.maxDate;
    dateInput.value = APP_CONFIG.dataset.minDate;
  }

  await loadStores();
  loadModels();

  const storeSelect = $("inpStore");
  if (storeSelect) {
    await loadItemsForStore(storeSelect.value);

    storeSelect.addEventListener("change", async () => {
      await loadItemsForStore(storeSelect.value);
    });
  }

  $("predictForm")?.addEventListener("submit", async (e) => {
    e.preventDefault();

    const store = $("inpStore")?.value?.trim();
    const itemRaw = $("inpItem")?.value?.trim();
    const date = $("inpDate")?.value;
    const model = $("inpModel")?.value;

    if (!store || !itemRaw || !date) {
      alert("Choose store, enter item, choose date and model.");
      return;
    }

    const resolvedItem = resolveItemInput(itemRaw);

    if (!resolvedItem) {
      alert("This item is not available in the trained set for the selected store. Enter Item 1 style alias or valid numeric item_nbr.");
      return;
    }

    const qs = new URLSearchParams({
      store: String(store),
      item: String(resolvedItem),
      date_from: date,
      date_to: date,
      model: String(model),
    });

    try {
      const payload = await fetchJson(`/predict/timeseries?${qs.toString()}`);
      renderPrediction(payload);
    } catch (err) {
      alert(`Prediction failed: ${err.message}`);
    }
  });
}

init();