"""Model wrappers and artifact naming conventions."""

from __future__ import annotations

from pathlib import Path


LABEL_MODEL_NAME = "label_model.joblib"
CWE_MODEL_NAME = "cwe_model.joblib"
CWE_MAPPING_NAME = "cwe_mapping.json"
FEATURE_COLUMNS_NAME = "feature_columns.json"
TRAIN_CACHE_NAME = "train_features.joblib"
TEST_CACHE_NAME = "test_features.joblib"


def ensure_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
