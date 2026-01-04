from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import joblib

from app.backend.models.schemas import PredictPointRequest


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def _log1p_nonneg(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    a = np.clip(a, 0.0, None)
    return np.log1p(a)


def _date_feats(d: date, start_date: Optional[date]) -> Dict[str, float]:
    ts = pd.Timestamp(d)
    dow = int(ts.dayofweek) + 1  # 1..7
    dom = int(ts.day)
    month = int(ts.month)
    weekofyear = int(ts.isocalendar().week)
    doy = int(ts.dayofyear)
    is_weekend = 1 if dow in (6, 7) else 0
    sin_doy = float(math.sin((2.0 * math.pi) * doy / 365.0))
    cos_doy = float(math.cos((2.0 * math.pi) * doy / 365.0))
    time_idx = int((ts.normalize() - pd.Timestamp(start_date)).days) if start_date else 0
    return {
        "dow": dow,
        "dom": dom,
        "month": month,
        "weekofyear": weekofyear,
        "doy": doy,
        "is_weekend": is_weekend,
        "sin_doy": sin_doy,
        "cos_doy": cos_doy,
        "time_idx": time_idx,
    }


@dataclass
class ModelRuntime:
    models_dir: Path
    meta: Dict[str, Any]
    feature_cols: Tuple[str, ...]
    _models_cache: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, models_dir: Path) -> "ModelRuntime":
        models_dir = Path(models_dir).expanduser().resolve()
        meta_path = models_dir / "model_meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing model_meta.json in {models_dir}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        feature_cols = tuple(meta.get("feature_cols") or [])
        if not feature_cols:
            raise ValueError("model_meta.json has no feature_cols")

        return cls(models_dir=models_dir, meta=meta, feature_cols=feature_cols)

    def list_model_ids(self):
        saved = self.meta.get("models_saved")
        if isinstance(saved, list) and saved:
            return [str(x) for x in saved]
        ids = ["best_model"]
        for p in self.models_dir.glob("*.joblib"):
            if p.name == "best_model.joblib":
                continue
            ids.append(p.stem)
        return sorted(set(ids))

    def load_model(self, model_id: str):
        model_id = str(model_id or "best_model")
        if model_id in self._models_cache:
            return self._models_cache[model_id]

        if model_id in ("best", "best_model"):
            path = self.models_dir / "best_model.joblib"
            cache_key = "best_model"
        else:
            path = self.models_dir / f"{model_id}.joblib"
            cache_key = model_id
            if not path.exists():
                path = self.models_dir / "best_model.joblib"
                cache_key = "best_model"

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        mdl = joblib.load(path)
        self._models_cache[cache_key] = mdl
        return mdl

    def _build_row(
        self,
        origin_date: date,
        target_date: date,
        recent_sales_log: np.ndarray,
        feature_overrides: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        feature_overrides = feature_overrides or {}

        start_date = None
        try:
            if self.meta.get("start_date"):
                start_date = pd.to_datetime(self.meta["start_date"]).date()
        except Exception:
            start_date = None

        row: Dict[str, float] = {c: 0.0 for c in self.feature_cols}

        # step horizon = 1
        if "horizon_cap" in row:
            row["horizon_cap"] = 1.0
        if "log1p_horizon_cap" in row:
            row["log1p_horizon_cap"] = float(math.log1p(1.0))

        o = _date_feats(origin_date, start_date)
        t = _date_feats(target_date, start_date)

        for k, v in o.items():
            if k in row:
                row[k] = float(v)
        for k, v in t.items():
            kk = f"{k}_t"
            if kk in row:
                row[kk] = float(v)

        # sales-derived features (log1p space)
        if recent_sales_log.size and "label_today" in row:
            row["label_today"] = float(recent_sales_log[-1])

        for lag in (1, 2, 3, 4, 5, 6, 7, 14, 28):
            key = f"lag{lag}_label"
            if key in row:
                val = recent_sales_log[-(lag + 1)] if recent_sales_log.size >= (lag + 1) else 0.0
                row[key] = float(val)

        for rw in (3, 7, 14, 28):
            key = f"roll{rw}_mean_label"
            if key in row:
                tail = recent_sales_log[-rw:] if recent_sales_log.size else np.array([], dtype=float)
                row[key] = float(np.mean(tail)) if tail.size else 0.0

        if "trend_7_28" in row:
            lag7 = recent_sales_log[-8] if recent_sales_log.size >= 8 else 0.0
            lag28 = recent_sales_log[-29] if recent_sales_log.size >= 29 else 0.0
            row["trend_7_28"] = float(lag7 - lag28)

        if "trend_1_7" in row:
            lag1 = recent_sales_log[-2] if recent_sales_log.size >= 2 else 0.0
            lag7 = recent_sales_log[-8] if recent_sales_log.size >= 8 else 0.0
            row["trend_1_7"] = float(lag1 - lag7)

        # manual overrides якщо захочеш (class/perishable/cluster/onpromotion_target тощо)
        for k, v in feature_overrides.items():
            if k in row:
                row[k] = _safe_float(v, row[k])

        return pd.DataFrame([row], columns=list(self.feature_cols))

    def predict_point(self, req: PredictPointRequest) -> Dict[str, Any]:
        warnings = []
        horizon = int(req.horizon or 1)
        if horizon < 1:
            horizon = 1

        recent_sales = req.recent_sales or []
        recent_sales_log = _log1p_nonneg(recent_sales)

        mdl = self.load_model(req.model_id)

        origin = req.origin_date
        cur_origin = origin
        cur_hist = recent_sales_log.copy()

        pred_log = 0.0
        for _ in range(horizon):
            target = cur_origin + timedelta(days=1)
            X = self._build_row(cur_origin, target, cur_hist, req.feature_overrides)
            pred_log = float(mdl.predict(X)[0])
            pred_log = float(max(pred_log, 0.0))  # log1p must be >= 0

            # recursive forward
            cur_hist = np.append(cur_hist, pred_log)
            if cur_hist.size > 60:
                cur_hist = cur_hist[-60:]
            cur_origin = target

        target_date = origin + timedelta(days=horizon)
        pred_sales = float(np.expm1(pred_log))

        band_low = max(0.0, pred_sales * 0.9)
        band_high = pred_sales * 1.1
        band = f"{band_low:.2f} – {band_high:.2f}"

        if not req.recent_sales:
            warnings.append("recent_sales не передано -> модель бачить нульову історію (прогноз буде умовний).")

        return {
            "target_date": target_date,
            "model_id": str(req.model_id),
            "pred_log": pred_log,
            "pred_sales": pred_sales,
            "band_low": float(band_low),
            "band_high": float(band_high),
            "band": band,
            "warnings": warnings,
        }


@lru_cache(maxsize=4)
def get_runtime(models_dir_str: str) -> ModelRuntime:
    return ModelRuntime.from_dir(Path(models_dir_str))
