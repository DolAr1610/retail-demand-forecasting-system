from __future__ import annotations

from typing import Optional, List

import numpy as np
import pandas as pd

from .loader import ArtifactLoader


class PredictorService:
    def __init__(self, loader: ArtifactLoader):
        self.loader = loader

    def families(self) -> List[str]:
        cat = self.loader.load_catalog()
        return sorted(cat["family"].dropna().unique().tolist())

    def items_by_family(self, family: str) -> List[int]:
        cat = self.loader.load_catalog()
        df = cat[cat["family"] == family]
        return df["item_nbr"].astype(int).dropna().unique().tolist()

    def _bundle(self, model_name: Optional[str] = None):
        return self.loader.load_bundle(model_name=model_name)

    def _final_clean_for_model(self, df_in: pd.DataFrame, categorical_cols, numeric_cols) -> pd.DataFrame:
        df = df_in.copy()
        for c in categorical_cols:
            df[c] = df[c].fillna("Unknown").astype(str)
        for c in numeric_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        return df

    def _build_recursive_feature_rows_for_date(
        self,
        pred_date: pd.Timestamp,
        future_base_df: pd.DataFrame,
        history_seed_df: pd.DataFrame,
        q_low: float,
        q_high: float,
        categorical_cols,
        numeric_cols,
        lags,
        rolling_windows,
        ewm_spans,
        flags,
    ) -> pd.DataFrame:
        base_today = future_base_df[future_base_df["date"] == pred_date].copy()
        if base_today.empty:
            return pd.DataFrame(columns=categorical_cols + numeric_cols)

        base_today = base_today.sort_values(["store_nbr", "item_nbr"]).reset_index(drop=True)
        hist = history_seed_df[history_seed_df["date"] < pred_date].copy()
        hist = hist.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

        use_promo_history = bool(flags.get("use_promo_history_features", False))
        use_returns = bool(flags.get("use_return_history_features", False))
        use_transactions = bool(flags.get("use_transactions_history_features", False))
        use_oil = bool(flags.get("use_oil_history_features", False))
        use_current_promo = bool(flags.get("use_current_day_promotion", False))

        rows = []
        for _, row in base_today.iterrows():
            store_id = row["store_nbr"]
            item_id = row["item_nbr"]
            s_hist = hist[(hist["store_nbr"] == store_id) & (hist["item_nbr"] == item_id)].copy()
            s_hist = s_hist.sort_values("date")

            sales_hist = s_hist["sales"].to_numpy(dtype=float)
            sales_hist_capped = np.clip(sales_hist, q_low, q_high) if len(sales_hist) > 0 else np.array([], dtype=float)
            promo_hist = s_hist["onpromotion"].to_numpy(dtype=float) if "onpromotion" in s_hist.columns else np.array([], dtype=float)

            feat = row.to_dict()
            age = len(s_hist)
            feat["series_age_days"] = age
            feat["has_history_1"] = int(age >= 1)
            feat["has_history_7"] = int(age >= 7)
            feat["has_history_14"] = int(age >= 14)
            feat["has_history_28"] = int(age >= 28)

            for lag in lags:
                lag_val = sales_hist[-lag] if age >= lag else np.nan
                feat[f"lag_{lag}"] = lag_val
                feat[f"log_lag_{lag}"] = np.log1p(max(lag_val, 0.0)) if pd.notna(lag_val) else np.nan

                if use_promo_history:
                    feat[f"promo_lag_{lag}"] = promo_hist[-lag] if len(promo_hist) >= lag else np.nan
                if use_returns:
                    feat[f"had_return_lag_{lag}"] = np.nan
                    feat[f"return_volume_lag_{lag}"] = np.nan
                if use_transactions:
                    feat[f"transactions_lag_{lag}"] = np.nan
                if use_oil:
                    feat[f"oil_lag_{lag}"] = np.nan

            for win in rolling_windows:
                if age >= win:
                    tail = sales_hist[-win:]
                    feat[f"rolling_mean_{win}"] = float(np.mean(tail))
                    feat[f"rolling_std_{win}"] = float(np.std(tail, ddof=1)) if win > 1 else 0.0
                    feat[f"rolling_min_{win}"] = float(np.min(tail))
                    feat[f"rolling_max_{win}"] = float(np.max(tail))
                else:
                    feat[f"rolling_mean_{win}"] = np.nan
                    feat[f"rolling_std_{win}"] = np.nan
                    feat[f"rolling_min_{win}"] = np.nan
                    feat[f"rolling_max_{win}"] = np.nan

                if use_promo_history:
                    feat[f"promo_rolling_mean_{win}"] = float(np.mean(promo_hist[-win:])) if len(promo_hist) >= win else np.nan
                if use_transactions:
                    feat[f"transactions_rolling_mean_{win}"] = np.nan
                if use_oil:
                    feat[f"oil_rolling_mean_{win}"] = np.nan

            for win in [7, 28]:
                feat[f"rolling_mean_capped_{win}"] = float(np.mean(sales_hist_capped[-win:])) if age >= win else np.nan

            for span in ewm_spans:
                feat[f"ewm_mean_{span}"] = float(pd.Series(sales_hist).ewm(span=span, adjust=False).mean().iloc[-1]) if age >= 1 else np.nan

            if use_current_promo:
                feat["promo_and_holiday"] = int((feat.get("onpromotion", 0) > 0) and (feat.get("is_holiday", 0) == 1))

            rows.append(feat)

        x_today = pd.DataFrame(rows)
        return self._final_clean_for_model(x_today, categorical_cols, numeric_cols)

    def _predict_rows_for_model(self, x_today_df: pd.DataFrame, model_name: Optional[str] = None):
        bundle = self._bundle(model_name=model_name)
        model = bundle["model"]
        preprocessor = bundle["preprocessor"]
        feature_cols = bundle["feature_cols"]
        target_mode = bundle.get("target_mode", "raw")

        X_today = preprocessor.transform(x_today_df[feature_cols]).astype(np.float32)
        yhat = model.predict(X_today)

        if target_mode == "log1p":
            yhat = np.expm1(yhat)

        return np.clip(np.asarray(yhat, dtype=np.float32), 0, None)

    def _recursive_predict_frame(
        self,
        future_frame: pd.DataFrame,
        history_seed_frame: pd.DataFrame,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        bundle = self._bundle(model_name=model_name)

        q_low = float(bundle["q_low"])
        q_high = float(bundle["q_high"])
        categorical_cols = bundle["categorical_feature_cols"]
        numeric_cols = bundle["numeric_feature_cols"]
        lags = bundle["lags"]
        rolling_windows = bundle["rolling_windows"]
        ewm_spans = bundle["ewm_spans"]
        flags = bundle.get("flags", {})

        future_frame = future_frame.copy().sort_values(["date", "store_nbr", "item_nbr"]).reset_index(drop=True)
        history_df = history_seed_frame.copy().sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

        pred_batches = []

        for pred_date in sorted(future_frame["date"].unique()):
            x_today = self._build_recursive_feature_rows_for_date(
                pred_date=pd.Timestamp(pred_date),
                future_base_df=future_frame,
                history_seed_df=history_df,
                q_low=q_low,
                q_high=q_high,
                categorical_cols=categorical_cols,
                numeric_cols=numeric_cols,
                lags=lags,
                rolling_windows=rolling_windows,
                ewm_spans=ewm_spans,
                flags=flags,
            )

            if x_today.empty:
                continue

            yhat = self._predict_rows_for_model(x_today, model_name=model_name)

            batch_pred = pd.DataFrame({
                "date": pd.to_datetime(x_today["date"]),
                "store_nbr": pd.to_numeric(x_today["store_nbr"]).astype(np.int64),
                "item_nbr": pd.to_numeric(x_today["item_nbr"]).astype(np.int64),
                "pred": yhat,
            })
            pred_batches.append(batch_pred)

            hist_add = pd.DataFrame({
                "date": pd.to_datetime(x_today["date"]),
                "store_nbr": pd.to_numeric(x_today["store_nbr"]).astype(np.int64),
                "item_nbr": pd.to_numeric(x_today["item_nbr"]).astype(np.int64),
                "onpromotion": pd.to_numeric(x_today["onpromotion"], errors="coerce").fillna(0).astype(np.int64),
                "sales": yhat,
            })
            hist_add["log_sales"] = np.log1p(hist_add["sales"])
            hist_add["sales_capped"] = np.clip(hist_add["sales"], q_low, q_high)

            history_df = pd.concat([
                history_df[["date", "store_nbr", "item_nbr", "onpromotion", "sales", "log_sales", "sales_capped"]],
                hist_add[["date", "store_nbr", "item_nbr", "onpromotion", "sales", "log_sales", "sales_capped"]],
            ], ignore_index=True).sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

        if pred_batches:
            return pd.concat(pred_batches, ignore_index=True)

        return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "pred"])

    def timeseries(
        self,
        store_nbr: int,
        item_nbr: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> pd.DataFrame:
        future_base = self.loader.load_future_base()
        history_seed = self.loader.load_history_seed()
        valid_actuals = self.loader.load_valid_actuals()
        eligible_pairs = self.loader.load_eligible_pairs()

        pair_exists = (
            (eligible_pairs["store_nbr"] == int(store_nbr)) &
            (eligible_pairs["item_nbr"] == int(item_nbr))
        ).any()

        if not pair_exists:
            return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "actual", "pred", "abs_error", "ape"])

        future_subset = future_base[
            (future_base["store_nbr"] == int(store_nbr)) &
            (future_base["item_nbr"] == int(item_nbr))
        ].copy()

        if date_from:
            future_subset = future_subset[future_subset["date"] >= pd.Timestamp(date_from)]
        if date_to:
            future_subset = future_subset[future_subset["date"] <= pd.Timestamp(date_to)]

        if future_subset.empty:
            return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "actual", "pred", "abs_error", "ape"])

        history_subset = history_seed[
            (history_seed["store_nbr"] == int(store_nbr)) &
            (history_seed["item_nbr"] == int(item_nbr))
        ].copy()

        pred_df = self._recursive_predict_frame(
            future_subset,
            history_subset,
            model_name=model_name,
        )

        if pred_df.empty:
            return pd.DataFrame(columns=["date", "store_nbr", "item_nbr", "actual", "pred", "abs_error", "ape"])

        actual_df = valid_actuals[
            (valid_actuals["store_nbr"] == int(store_nbr)) &
            (valid_actuals["item_nbr"] == int(item_nbr))
        ][["date", "store_nbr", "item_nbr", "sales_actual"]].copy()

        out = pred_df.merge(actual_df, on=["date", "store_nbr", "item_nbr"], how="left")
        out = out.rename(columns={"sales_actual": "actual"})
        out["abs_error"] = np.where(out["actual"].notna(), np.abs(out["actual"] - out["pred"]), np.nan)
        out["ape"] = np.where(
            out["actual"].notna() & (np.abs(out["actual"]) > 1e-12),
            out["abs_error"] / np.abs(out["actual"]),
            np.nan,
        )

        return out.sort_values(["date"]).reset_index(drop=True)