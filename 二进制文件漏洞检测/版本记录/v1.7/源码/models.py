"""Model bundle naming conventions for the competition workspace."""

from __future__ import annotations

from pathlib import Path


MODEL_VERSION = "v1.7"

LEGACY_LABEL_MODEL_NAME = "label_model.joblib"
LEGACY_CWE_MODEL_NAME = "cwe_model.joblib"
LEGACY_CWE_MAPPING_NAME = "cwe_mapping.json"
LEGACY_FEATURE_COLUMNS_NAME = "feature_columns.json"
LEGACY_TRAIN_CACHE_NAME = "train_features.joblib"
LEGACY_TEST_CACHE_NAME = "test_features.joblib"

LABEL_MODEL_NAME = f"label_model_{MODEL_VERSION}.joblib"
CWE_MODEL_NAME = f"cwe_model_{MODEL_VERSION}.joblib"
CWE_MAPPING_NAME = f"cwe_mapping_{MODEL_VERSION}.json"
FEATURE_COLUMNS_NAME = f"feature_columns_{MODEL_VERSION}.json"
TRAIN_CACHE_NAME = f"train_features_{MODEL_VERSION}.joblib"
TEST_CACHE_NAME = f"test_features_{MODEL_VERSION}.joblib"
TRAIN_BYTE_CACHE_NAME = f"train_bytes_{MODEL_VERSION}.joblib"
TEST_BYTE_CACHE_NAME = f"test_bytes_{MODEL_VERSION}.joblib"
# Universal caches: test data never changes, use version-independent names
UNIVERSAL_TEST_CACHE_NAME = "test_features.joblib"
UNIVERSAL_TEST_BYTE_CACHE_NAME = "test_bytes.joblib"
UNIVERSAL_TRAIN_CACHE_NAME = "train_features_base.joblib"
UNIVERSAL_TRAIN_BYTE_CACHE_NAME = "train_bytes_base.joblib"
TABULAR_BUNDLE_NAME = f"tabular_bundle_{MODEL_VERSION}.joblib"
NEURAL_BUNDLE_NAME = f"neural_bundle_{MODEL_VERSION}.pt"
FUSION_CONFIG_NAME = f"fusion_config_{MODEL_VERSION}.json"
SUBMISSION_NAME = f"submission_{MODEL_VERSION}.csv"
PSEUDO_TRAIN_CSV = "pseudo_train_v1.7.csv"

# Per-seed artifact names
def seed_label_model(seed: int) -> str:
    return f"label_model_{MODEL_VERSION}_seed{seed}.joblib"

def seed_cwe_model(seed: int) -> str:
    return f"cwe_model_{MODEL_VERSION}_seed{seed}.joblib"

def seed_neural_bundle(seed: int) -> str:
    return f"neural_bundle_{MODEL_VERSION}_seed{seed}.pt"

def seed_fusion_mlp(seed: int) -> str:
    return f"fusion_mlp_{MODEL_VERSION}_seed{seed}.pt"

def seed_tabular_bundle(seed: int) -> str:
    return f"tabular_bundle_{MODEL_VERSION}_seed{seed}.joblib"


def ensure_model_dir(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
