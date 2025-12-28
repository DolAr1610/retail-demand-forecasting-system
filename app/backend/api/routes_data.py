from typing import Optional, List
from fastapi import APIRouter, Depends, Query, HTTPException

from app.backend.settings import Settings, get_settings
from app.backend.services.artifact_store import ArtifactPaths
from app.backend.services.loader import ArtifactLoader

router = APIRouter(prefix="/data", tags=["data"])

def _get_loader(settings: Settings = Depends(get_settings)) -> ArtifactLoader:
    paths = ArtifactPaths(root=settings.artifacts_dir)
    return ArtifactLoader(paths)

@router.get("/families", response_model=List[str])
def list_families(loader: ArtifactLoader = Depends(_get_loader)) -> List[str]:
    try:
        cat = loader.load_catalog()
        return sorted(cat["family"].dropna().unique().tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/items", response_model=List[int])
def list_items(
    family: Optional[str] = Query(None, description="family name"),
    loader: ArtifactLoader = Depends(_get_loader),
) -> List[int]:
    try:
        cat = loader.load_catalog()
        if family:
            cat = cat[cat["family"] == family]
        return cat["item_nbr"].astype(int).dropna().unique().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
