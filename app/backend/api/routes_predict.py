from typing import Optional
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
    store: Optional[int] = Query(None, description="store_nbr (optional if single-store)"),
    item: int = Query(..., description="item_nbr"),
    date_from: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    date_to: Optional[str] = Query(None, description="ISO date YYYY-MM-DD"),
    settings: Settings = Depends(get_settings),
    svc: PredictorService = Depends(_get_service),
) -> PredictTimeseriesResponse:
    try:
        store_nbr = int(store) if store is not None else int(settings.default_store_nbr)

        df = svc.timeseries(store_nbr, int(item), date_from, date_to)
        if df.empty:
            return PredictTimeseriesResponse(
                store_nbr=store_nbr,
                item_nbr=int(item),
                date_from=date_from,
                date_to=date_to,
                rows=[],
                count=0,
            )

        rows = []
        for _, r in df.iterrows():
            rows.append(PredictRow(
                date=str(r["date"].date()) if hasattr(r["date"], "date") else str(r["date"]),
                store_nbr=int(r["store_nbr"]),
                item_nbr=int(r["item_nbr"]),
                actual=float(r["actual"]),
                pred=float(r["pred"]),
                abs_error=float(r["abs_error"]),
                ape=(float(r["ape"]) if r.get("ape") is not None and r["ape"] == r["ape"] else None),
            ))

        return PredictTimeseriesResponse(
            store_nbr=store_nbr,
            item_nbr=int(item),
            date_from=date_from,
            date_to=date_to,
            rows=rows,
            count=len(rows),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
