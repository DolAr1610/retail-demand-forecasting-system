from typing import Optional, List

import pandas as pd
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

@router.get("/models", response_model=List[str])
def list_models(loader: ArtifactLoader = Depends(_get_loader)) -> List[str]:
    try:
        return loader.load_available_model_names()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/items", response_model=List[int])
def list_items(
    family: Optional[str] = Query(None, description="family name"),
    store: Optional[int] = Query(None, description="optional store_nbr filter"),
    loader: ArtifactLoader = Depends(_get_loader),
) -> List[int]:
    try:
        cat = loader.load_catalog()
        pairs = loader.load_eligible_pairs()

        if family:
            cat = cat[cat["family"] == family]

        if store is not None:
            pairs = pairs[pairs["store_nbr"] == int(store)]
            cat = cat[cat["item_nbr"].isin(pairs["item_nbr"].unique())]

        return cat["item_nbr"].astype(int).dropna().unique().tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/items-labeled")
def list_items_labeled(
    store: Optional[int] = Query(None, description="optional store_nbr filter"),
    family: Optional[str] = Query(None, description="optional family filter"),
    loader: ArtifactLoader = Depends(_get_loader),
):
    try:
        aliases = loader.load_item_aliases()
        alias_map = {int(x["item_nbr"]): x["display_name"] for x in aliases}

        cat = loader.load_catalog()
        pairs = loader.load_eligible_pairs()

        if store is not None:
            pairs = pairs[pairs["store_nbr"] == int(store)]

        valid_items = set(pairs["item_nbr"].astype(int).tolist())
        cat = cat[cat["item_nbr"].astype(int).isin(valid_items)].copy()

        if family:
            cat = cat[cat["family"] == family].copy()

        cat["item_nbr"] = pd.to_numeric(cat["item_nbr"], errors="coerce")
        cat = cat.dropna(subset=["item_nbr"]).copy()
        cat["item_nbr"] = cat["item_nbr"].astype(int)

        rows = []
        for _, r in cat.sort_values(["item_nbr"]).iterrows():
            item_nbr = int(r["item_nbr"])
            rows.append({
                "item_nbr": item_nbr,
                "display_name": alias_map.get(item_nbr, f"Item {item_nbr}"),
                "family": str(r["family"]) if pd.notna(r.get("family")) else None,
            })

        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stores", response_model=List[int])
def list_stores(loader: ArtifactLoader = Depends(_get_loader)) -> List[int]:
    try:
        pairs = loader.load_eligible_pairs()
        return sorted(pairs["store_nbr"].astype(int).dropna().unique().tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))