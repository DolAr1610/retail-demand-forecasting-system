from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import joblib
import pandas as pd

from .artifact_store import ArtifactPaths


class ArtifactLoader:
    def __init__(self, paths: ArtifactPaths):
        self.paths = paths

        self._shared_bundle: Optional[Dict[str, Any]] = None
        self._model_payloads: Dict[str, Dict[str, Any]] = {}

        self._history_seed: Optional[pd.DataFrame] = None
        self._future_base: Optional[pd.DataFrame] = None
        self._valid_actuals: Optional[pd.DataFrame] = None
        self._catalog: Optional[pd.DataFrame] = None
        self._metrics: Optional[Dict[str, Any]] = None
        self._model_info: Optional[Dict[str, Any]] = None
        self._eligible_pairs: Optional[pd.DataFrame] = None
        self._available_dates: Optional[pd.DataFrame] = None
        self._item_aliases = None
        self._serialized_model_index: Optional[List[Dict[str, Any]]] = None

    def _root(self) -> Path:
        return Path(self.paths.root)

    def _parquet(self, rel: str) -> Path:
        return self._root() / rel

    def _json(self, rel: str) -> Dict[str, Any]:
        with open(self._root() / rel, "r", encoding="utf-8") as f:
            return json.load(f)

    # --------------------------------------------------
    # NEW MULTI-MODEL LOADERS
    # --------------------------------------------------
    def load_shared_bundle(self) -> Dict[str, Any]:
        if self._shared_bundle is None:
            p = self._root() / "bundle" / "shared_bundle.joblib"
            if not p.exists():
                raise FileNotFoundError(f"Shared bundle not found: {p}")
            self._shared_bundle = joblib.load(p)
        return self._shared_bundle

    def load_serialized_model_index(self) -> List[Dict[str, Any]]:
        if self._serialized_model_index is None:
            p = self._root() / "metadata" / "serialized_model_index.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self._serialized_model_index = json.load(f)
            else:
                self._serialized_model_index = []
        return self._serialized_model_index

    def load_available_model_names(self) -> List[str]:
        shared = self.load_shared_bundle()
        names = shared.get("available_model_names") or []
        return [str(x) for x in names]

    def load_default_model_name(self) -> Optional[str]:
        shared = self.load_shared_bundle()
        return shared.get("default_model_name")

    def load_model_payload(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        shared = self.load_shared_bundle()
        selected_model_name = model_name or shared.get("default_model_name")

        if not selected_model_name:
            raise ValueError("No model_name provided and no default_model_name found in shared bundle.")

        if selected_model_name in self._model_payloads:
            return self._model_payloads[selected_model_name]

        model_index = self.load_serialized_model_index()
        entry = next((x for x in model_index if x.get("model_name") == selected_model_name), None)

        if entry and entry.get("file"):
            model_path = self._root() / entry["file"]
        else:
            model_path = self._root() / "bundle" / "models" / f"{selected_model_name}.joblib"

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found for '{selected_model_name}': {model_path}")

        payload = joblib.load(model_path)
        self._model_payloads[selected_model_name] = payload
        return payload

    def load_bundle(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Backward-compatible combined bundle:
        shared preprocessing/config + selected model payload
        """
        shared = self.load_shared_bundle()
        model_payload = self.load_model_payload(model_name=model_name)

        combined = dict(shared)
        combined["model_name"] = model_payload["model_name"]
        combined["target_mode"] = model_payload["target_mode"]
        combined["model_family"] = model_payload.get("family")
        combined["model"] = model_payload["model"]
        return combined

    # --------------------------------------------------
    # DATA LOADERS
    # --------------------------------------------------
    def load_history_seed(self) -> pd.DataFrame:
        if self._history_seed is None:
            p = self._parquet("runtime/history_seed.parquet")
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df["store_nbr"] = pd.to_numeric(df["store_nbr"]).astype(int)
            df["item_nbr"] = pd.to_numeric(df["item_nbr"]).astype(int)
            self._history_seed = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)
        return self._history_seed.copy()

    def load_future_base(self) -> pd.DataFrame:
        if self._future_base is None:
            p = self._parquet("runtime/future_base_known.parquet")
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df["store_nbr"] = pd.to_numeric(df["store_nbr"]).astype(int)
            df["item_nbr"] = pd.to_numeric(df["item_nbr"]).astype(int)
            self._future_base = df.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)
        return self._future_base.copy()

    def load_valid_actuals(self) -> pd.DataFrame:
        if self._valid_actuals is None:
            p = self._parquet("runtime/valid_actuals.parquet")
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            df["store_nbr"] = pd.to_numeric(df["store_nbr"]).astype(int)
            df["item_nbr"] = pd.to_numeric(df["item_nbr"]).astype(int)
            self._valid_actuals = df.sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)
        return self._valid_actuals.copy()

    def load_catalog(self) -> pd.DataFrame:
        if self._catalog is None:
            base = self.load_future_base()
            cols = ["item_nbr", "family"]
            if not set(cols).issubset(base.columns):
                raise FileNotFoundError("family mapping not found in runtime/future_base_known.parquet")
            df = base[cols].dropna().drop_duplicates().copy()
            df["item_nbr"] = pd.to_numeric(df["item_nbr"]).astype(int)
            df["family"] = df["family"].astype(str)
            self._catalog = df.sort_values(["family", "item_nbr"]).reset_index(drop=True)
        return self._catalog.copy()

    def load_eligible_pairs(self) -> pd.DataFrame:
        if self._eligible_pairs is None:
            p = self._parquet("lookups/eligible_store_item_pairs.parquet")
            df = pd.read_parquet(p)
            df["store_nbr"] = pd.to_numeric(df["store_nbr"]).astype(int)
            df["item_nbr"] = pd.to_numeric(df["item_nbr"]).astype(int)
            self._eligible_pairs = df.drop_duplicates().sort_values(["store_nbr", "item_nbr"]).reset_index(drop=True)
        return self._eligible_pairs.copy()

    def load_available_dates(self) -> pd.DataFrame:
        if self._available_dates is None:
            p = self._parquet("lookups/available_dates.parquet")
            df = pd.read_parquet(p)
            df["date"] = pd.to_datetime(df["date"])
            self._available_dates = df.drop_duplicates().sort_values(["date"]).reset_index(drop=True)
        return self._available_dates.copy()

    def load_item_aliases(self):
        if self._item_aliases is None:
            p = self._root() / "lookups" / "item_aliases.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    self._item_aliases = json.load(f)
            else:
                pairs = self.load_eligible_pairs()
                items = sorted(pairs["item_nbr"].astype(int).dropna().unique().tolist())
                self._item_aliases = [
                    {"item_nbr": int(item_nbr), "display_name": f"Item {idx}"}
                    for idx, item_nbr in enumerate(items, start=1)
                ]
        return self._item_aliases

    def load_metrics(self) -> Dict[str, Any]:
        if self._metrics is None:
            self._metrics = self._json("metadata/metrics.json")
        return self._metrics

    def load_model_info(self) -> Dict[str, Any]:
        if self._model_info is None:
            self._model_info = self._json("metadata/model_info.json")
        return self._model_info

    def load_timeseries(
        self,
        store_nbr: int,
        item_nbr: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        from .predictor import PredictorService

        svc = PredictorService(self)
        return svc.timeseries(store_nbr, item_nbr, date_from, date_to, model_name=model_name)