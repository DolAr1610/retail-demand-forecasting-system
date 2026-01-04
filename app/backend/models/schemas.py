from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, AliasChoices


class ModelInfo(BaseModel):
    id: str
    name: str


class PredictContextResponse(BaseModel):
    min_date: str
    max_date: str
    store_nbr: Optional[int] = None
    models: list[ModelInfo] = Field(default_factory=list)
    items: list[int] = Field(default_factory=list)


class PredictPointRequest(BaseModel):
    origin_date: str = Field(validation_alias=AliasChoices("origin_date", "date"))
    store_nbr: int
    item_nbr: int
    horizon: int = 1
    model_id: str = "best_model"
    recent_sales: Optional[list[float]] = None


class PredictPointResponse(BaseModel):
    origin_date: str
    target_date: str
    store_nbr: int
    item_nbr: int
    horizon: int
    model_id: str
    pred_log: float
    pred_sales: float
    band_low: Optional[float] = None
    band_high: Optional[float] = None
    warnings: list[str] = Field(default_factory=list)
