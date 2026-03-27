export const APP_CONFIG = {
  dataset: {
    minDate: "",
    maxDate: "",
    minDateLabel: "—",
    maxDateLabel: "—",
  },
  subset: {
    store: null,
    families: 0,
    items: 0,
  },
  models: [],
  metrics: [],
  modelInfo: {},
};

function formatDateLabel(isoDate) {
  if (!isoDate) return "—";
  try {
    return new Intl.DateTimeFormat("en-GB", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    }).format(new Date(`${isoDate}T00:00:00`));
  } catch {
    return isoDate;
  }
}

async function fetchJson(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`${url}: ${await res.text()}`);
  }
  return await res.json();
}

export async function loadAppConfig() {
  try {
    const [modelInfoPayload, metricsPayload, stores, families] = await Promise.all([
      fetchJson("/metrics/model_info"),
      fetchJson("/metrics/global"),
      fetchJson("/data/stores"),
      fetchJson("/data/families"),
    ]);

    const info = modelInfoPayload.metrics || {};
    const metrics = metricsPayload.metrics || {};

    APP_CONFIG.modelInfo = info;
    APP_CONFIG.metrics = metrics.test_metrics || metrics.leaderboard || [];

    APP_CONFIG.dataset.minDate = info.future_date_min || "";
    APP_CONFIG.dataset.maxDate = info.future_date_max || "";
    APP_CONFIG.dataset.minDateLabel = formatDateLabel(APP_CONFIG.dataset.minDate);
    APP_CONFIG.dataset.maxDateLabel = formatDateLabel(APP_CONFIG.dataset.maxDate);

    APP_CONFIG.subset.store =
      Array.isArray(info.selected_stores) && info.selected_stores.length
        ? info.selected_stores[0]
        : (stores?.[0] ?? null);

    APP_CONFIG.subset.families = Array.isArray(families) ? families.length : 0;
    APP_CONFIG.subset.items = Number(info.n_eligible_pairs || 0);

    const availableModels = info.available_serialized_models || [];
    const defaultModel = info.default_serialized_model_name || info.selected_fitted_model_name || null;

    if (availableModels.length) {
      APP_CONFIG.models = availableModels.map((m) => ({
        id: m,
        name: m,
      }));
    } else if (defaultModel) {
      APP_CONFIG.models = [{ id: defaultModel, name: defaultModel }];
    } else {
      APP_CONFIG.models = [];
    }

    if (
      APP_CONFIG.subset.store !== null &&
      Array.isArray(stores) &&
      stores.length &&
      !stores.includes(APP_CONFIG.subset.store)
    ) {
      APP_CONFIG.subset.store = stores[0];
    }
  } catch (err) {
    console.error("Failed to load app config:", err);
  }

  applyConfigPlaceholders();
}

export function setActiveNav() {
  const path = window.location.pathname.replace(/\/+$/, "");
  document.querySelectorAll(".nav-link").forEach((a) => a.classList.remove("active"));

  const routes = [
    { key: "/ui", href: "/ui/" },
    { key: "/ui/predict", href: "/ui/predict" },
    { key: "/ui/analytics", href: "/ui/analytics" },
    { key: "/ui/about", href: "/ui/about" },
    { key: "/ui/agent", href: "/ui/agent" },
  ];

  let hit = routes[0];
  for (const r of routes) {
    if (path === r.key) {
      hit = r;
      break;
    }
  }
  if (path === "/ui") hit = routes[0];

  const active = document.querySelector(`.nav-link[href="${hit.href}"], .nav-link[href="${hit.href}/"]`);
  if (active) active.classList.add("active");
}

export function fmt(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

export function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

export function applyConfigPlaceholders() {
  const windowText =
    APP_CONFIG.dataset.minDateLabel !== "—" && APP_CONFIG.dataset.maxDateLabel !== "—"
      ? `${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel}`
      : "—";

  setText("dsWindow", windowText);
  setText("subsetStore", APP_CONFIG.subset.store !== null ? String(APP_CONFIG.subset.store) : "—");
  setText("subsetFamilies", String(APP_CONFIG.subset.families || 0));
  setText("subsetItems", String(APP_CONFIG.subset.items || 0));
  setText("ctxRange", windowText);
  setText("ctxStore", APP_CONFIG.subset.store !== null ? String(APP_CONFIG.subset.store) : "—");
  setText("ctxFamilies", String(APP_CONFIG.subset.families || 0));
  setText("ctxItems", String(APP_CONFIG.subset.items || 0));
}

setActiveNav();