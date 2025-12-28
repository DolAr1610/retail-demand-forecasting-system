from typing import Dict
from .loader import ArtifactLoader


class MetricsService:
    def __init__(self, loader: ArtifactLoader):
        self.loader = loader

    def global_metrics(self) -> Dict:
        return self.loader.load_metrics()

    def model_info(self) -> Dict:
        return self.loader.load_model_info()
