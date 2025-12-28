from typing import Optional
import pandas as pd
from pathlib import Path

from .artifact_store import ArtifactPaths

class ArtifactLoader:
    def __init__(self, paths: ArtifactPaths):
        self.paths = paths

    def load_timeseries(self, store_nbr: int, item_nbr: int,
                        date_from: Optional[str] = None,
                        date_to: Optional[str] = None) -> pd.DataFrame:
        # як у тебе вже зроблено
        ...

    def load_catalog(self) -> pd.DataFrame:
        """
        Must return columns: item_nbr (int), family (str)
        Put this file into artifacts, e.g. artifacts/catalog.parquet
        """
        p = Path(self.paths.root) / "catalog.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            df["item_nbr"] = df["item_nbr"].astype(int)
            df["family"] = df["family"].astype(str)
            return df[["item_nbr", "family"]].drop_duplicates().sort_values(["family","item_nbr"])

        raise FileNotFoundError(f"Catalog not found: {p}. Create artifacts/catalog.parquet with item_nbr,family.")
