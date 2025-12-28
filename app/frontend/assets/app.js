export const APP_CONFIG = {
  dataset: {
    minDate: "2013-01-01",
    maxDate: "2017-08-15",
    minDateLabel: "1 Jan 2013",
    maxDateLabel: "15 Aug 2017",
  },
  subset: {
    store: 1,
    families: 5,
    items: 200,
  },
  models: [
    { id: "persist", name: "Persistence baseline" },
    { id: "roll7", name: "Rolling Mean (7 days)" },
    { id: "gbt", name: "Gradient-Boosted Trees" },
    { id: "rf", name: "Random Forest" },
  ],
};

// Highlight active nav item based on current path
export function setActiveNav() {
  const path = window.location.pathname.replace(/\/+$/, ""); // trim trailing slash

  document.querySelectorAll(".nav-link").forEach((a) => a.classList.remove("active"));

  const routes = [
    { key: "/ui", href: "/ui/" },
    { key: "/ui/predict", href: "/ui/predict" },
    { key: "/ui/analytics", href: "/ui/analytics" },
    { key: "/ui/about", href: "/ui/about" },
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

// Apply config placeholders where present
export function applyConfigPlaceholders() {
  setText("dsWindow", `${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel}`);
  setText("subsetStore", String(APP_CONFIG.subset.store));
  setText("subsetFamilies", String(APP_CONFIG.subset.families));
  setText("subsetItems", String(APP_CONFIG.subset.items));

  setText("ctxRange", `${APP_CONFIG.dataset.minDateLabel} → ${APP_CONFIG.dataset.maxDateLabel}`);
  setText("ctxStore", String(APP_CONFIG.subset.store));
  setText("ctxFamilies", String(APP_CONFIG.subset.families));
  setText("ctxItems", String(APP_CONFIG.subset.items));
}

// Boot
setActiveNav();
applyConfigPlaceholders();
