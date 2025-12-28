from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class DataStatusResponse(BaseModel):
    artifacts_dir: str
    has_predictions: bool
    has_metrics: bool
    has_model_info: bool
    has_lookups: bool

    stores_count: Optional[int] = None
    items_count: Optional[int] = None

    date_min: Optional[str] = None  # ISO date
    date_max: Optional[str] = None  # ISO date

    notes: List[str] = Field(default_factory=list)


class GlobalMetricsResponse(BaseModel):
    metrics: Dict[str, Any] = Field(default_factory=dict)


class ListResponse(BaseModel):
    values: List[int]


class PredictRow(BaseModel):
    date: str
    store_nbr: int
    item_nbr: int
    actual: float
    pred: float
    abs_error: float
    ape: Optional[float] = None


class PredictTimeseriesResponse(BaseModel):
    store_nbr: int
    item_nbr: int
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    rows: List[PredictRow]
    count: int
