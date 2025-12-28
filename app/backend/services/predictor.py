from typing import Optional, List
import pandas as pd
from .loader import ArtifactLoader

class PredictorService:
    """
    MVP predictor: pulls backtest predictions from parquet artifacts.
    (No online inference / no Spark here.)
    """

    def __init__(self, loader: ArtifactLoader):
        self.loader = loader

    def timeseries(
        self,
        store_nbr: int,
        item_nbr: int,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.loader.load_timeseries(store_nbr, item_nbr, date_from, date_to)

    def families(self) -> List[str]:
        cat = self.loader.load_catalog()  
        return sorted(cat["family"].dropna().unique().tolist())

    def items_by_family(self, family: str) -> List[int]:
        cat = self.loader.load_catalog()
        df = cat[cat["family"] == family]
        return df["item_nbr"].astype(int).dropna().unique().tolist()
