from __future__ import annotations

from fastapi import APIRouter, Depends

from app.backend.settings import Settings, get_settings
from app.backend.services.model_runtime import NoSparkRuntime

router = APIRouter(prefix="/data", tags=["data"])


@router.get("/catalog")
def get_catalog(settings: Settings = Depends(get_settings)):
    rt = NoSparkRuntime(settings.models_dir)
    return {
        "store_nbr": rt.subset_store,
        "families": rt.families,
        "items": rt.items_list,
        "start_date": str(rt.start_date),
        "end_date": str(rt.end_date),
    }


@router.get("/families")
def get_families(settings: Settings = Depends(get_settings)):
    rt = NoSparkRuntime(settings.models_dir)
    return {"families": rt.families}


@router.get("/items")
def get_items(settings: Settings = Depends(get_settings), family: str | None = None):
    rt = NoSparkRuntime(settings.models_dir)
    # якщо family нема/не використовується — просто всі items
    return {"items": rt.items_list}
