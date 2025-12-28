import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ArtifactPaths:
    root: str

    @property
    def predictions_dir(self) -> str:
        return os.path.join(self.root, "valid_predictions")

    @property
    def lookups_dir(self) -> str:
        return os.path.join(self.root, "lookups")

    @property
    def metrics_path(self) -> str:
        return os.path.join(self.root, "metrics.json")

    @property
    def model_info_path(self) -> str:
        return os.path.join(self.root, "model_info.json")

    def store_partition_path(self, store_nbr: int) -> str:
        # Spark partition folder format: store_nbr=1
        return os.path.join(self.predictions_dir, f"store_nbr={int(store_nbr)}")

    def stores_lookup_path(self) -> str:
        return os.path.join(self.lookups_dir, "stores_list")

    def items_lookup_path(self) -> str:
        return os.path.join(self.lookups_dir, "items_list")

    def exists_dir(self, path: str) -> bool:
        return os.path.isdir(path)

    def exists_file(self, path: str) -> bool:
        return os.path.isfile(path)

    def find_any_parquet(self, path: str) -> Optional[str]:
        if not os.path.isdir(path):
            return None
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".parquet"):
                    return os.path.join(root, f)
        return None
