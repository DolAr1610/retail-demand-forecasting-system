from typing import Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from app.backend.models.schemas import PredictTimeseriesResponse, PredictRow
from app.backend.settings import Settings, get_settings
from app.backend.services.artifact_store import ArtifactPaths
from app.backend.services.loader import ArtifactLoader
from app.backend.services.predictor import PredictorService

router = APIRouter(prefix="/predict", tags=["predict"])


def _get_service(settings: Settings = Depends(get_settings)) -> PredictorService:
    paths = ArtifactPaths(root=settings.artifacts_dir)
    loader = ArtifactLoader(paths)
    return PredictorService(loader)


@router.get("/timeseries", response_model=PredictTimeseriesResponse)
def predict_timeseries(
    store: Optional[int] = Query(None, description="store_nbr"),
    item: int = Query(..., description="item_nbr"),
    date_from: str = Query(..., description="ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    model: Optional[str] = Query(None, description="selected model name"),
    settings: Settings = Depends(get_settings),
    svc: PredictorService = Depends(_get_service),
) -> PredictTimeseriesResponse:
    try:
        store_nbr = int(store) if store is not None else int(settings.default_store_nbr)

        from_dt = pd.Timestamp(date_from)
        to_dt = pd.Timestamp(date_to) if date_to else from_dt

        available_models = svc.loader.load_available_model_names()
        default_model_name = svc.loader.load_default_model_name()
        selected_model_name = model or default_model_name

        if available_models and selected_model_name not in available_models:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{selected_model_name}' is not available. Available models: {available_models}",
            )

        df = svc.timeseries(
            store_nbr,
            int(item),
            str(from_dt.date()),
            str(to_dt.date()),
            model_name=selected_model_name,
        )

        if df.empty:
            return PredictTimeseriesResponse(
                store_nbr=store_nbr,
                item_nbr=int(item),
                date_from=str(from_dt.date()),
                date_to=str(to_dt.date()),
                model_name=selected_model_name,
                rows=[],
                count=0,
            )

        rows = []
        for _, r in df.iterrows():
            rows.append(
                PredictRow(
                    date=str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
                    store_nbr=int(r["store_nbr"]),
                    item_nbr=int(r["item_nbr"]),
                    pred=float(r["pred"]),
                    actual=(float(r["actual"]) if pd.notna(r.get("actual")) else None),
                    abs_error=(float(r["abs_error"]) if pd.notna(r.get("abs_error")) else None),
                    ape=(float(r["ape"]) if pd.notna(r.get("ape")) else None),
                )
            )

        return PredictTimeseriesResponse(
            store_nbr=store_nbr,
            item_nbr=int(item),
            date_from=str(from_dt.date()),
            date_to=str(to_dt.date()),
            model_name=selected_model_name,
            rows=rows,
            count=len(rows),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))