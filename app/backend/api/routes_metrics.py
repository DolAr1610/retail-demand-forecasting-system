from fastapi import APIRouter, Depends, HTTPException
from app.backend.models.schemas import GlobalMetricsResponse
from app.backend.settings import Settings, get_settings
from app.backend.services.artifact_store import ArtifactPaths
from app.backend.services.loader import ArtifactLoader
from app.backend.services.metrics import MetricsService

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _get_service(settings: Settings = Depends(get_settings)) -> MetricsService:
    paths = ArtifactPaths(root=settings.artifacts_dir)
    loader = ArtifactLoader(paths)
    return MetricsService(loader)


@router.get("/global", response_model=GlobalMetricsResponse)
def global_metrics(svc: MetricsService = Depends(_get_service)) -> GlobalMetricsResponse:
    try:
        return GlobalMetricsResponse(metrics=svc.global_metrics())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model_info", response_model=GlobalMetricsResponse)
def model_info(svc: MetricsService = Depends(_get_service)) -> GlobalMetricsResponse:
    try:
        return GlobalMetricsResponse(metrics=svc.model_info())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
