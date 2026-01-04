from __future__ import annotations

from pathlib import Path
from fastapi import APIRouter, HTTPException

from app.backend.settings import settings
from app.backend.models.schemas import PredictContextResponse, PredictPointRequest, PredictPointResponse
from app.backend.services.model_runtime import ModelRuntime

router = APIRouter(prefix="/predict")

_runtime: ModelRuntime | None = None

def runtime() -> ModelRuntime:
    global _runtime
    if _runtime is None:
        _runtime = ModelRuntime(models_dir=Path(settings.models_dir))
    return _runtime

@router.get("/context", response_model=PredictContextResponse)
def get_context():
    return runtime().get_context()

@router.post("/point", response_model=PredictPointResponse)
def predict_point(req: PredictPointRequest):
    try:
        return runtime().predict_point(req)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
