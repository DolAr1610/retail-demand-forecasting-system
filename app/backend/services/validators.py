from typing import List
from .artifact_store import ArtifactPaths


def validate_artifacts_layout(p: ArtifactPaths) -> List[str]:
    notes: List[str] = []

    if not p.exists_dir(p.root):
        notes.append(f"Artifacts root dir not found: {p.root}")
        return notes

    if not p.exists_dir(p.predictions_dir):
        notes.append("Missing predictions dir: valid_predictions/")

    if not p.exists_dir(p.lookups_dir):
        notes.append("Missing lookups dir: lookups/")

    if not p.exists_file(p.metrics_path):
        notes.append("Missing metrics.json")

    if not p.exists_file(p.model_info_path):
        notes.append("Missing model_info.json")

    if p.find_any_parquet(p.predictions_dir) is None:
        notes.append("No .parquet files found under valid_predictions/ (is it empty?)")

    if p.exists_dir(p.lookups_dir):
        if p.find_any_parquet(p.stores_lookup_path()) is None:
            notes.append("No parquet found under lookups/stores_list (expected parquet dataset)")
        if p.find_any_parquet(p.items_lookup_path()) is None:
            notes.append("No parquet found under lookups/items_list (expected parquet dataset)")

    return notes
