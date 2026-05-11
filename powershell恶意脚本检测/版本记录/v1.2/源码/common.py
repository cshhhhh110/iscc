from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data_train.csv"
TEST_PATH = PROJECT_ROOT / "data_test.csv"
MODEL_DIR = PROJECT_ROOT / "模型"
RESULT_DIR = PROJECT_ROOT / "提交结果"
PROJECT_LOG_PATH = PROJECT_ROOT / "ACTION_LOG.md"
TOTAL_LOG_PATH = PROJECT_ROOT.parent / "ACTION_LOG.md"

LABELS = [0, 1, 2]
TARGET_COLUMN = "label"
ID_COLUMN = "name"
FEATURE_COUNT = 15
ARTIFACT_VERSION = "v1.2"


def append_log(message: str) -> None:
    PROJECT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with PROJECT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {timestamp} {message}\n")


def append_total_log(message: str) -> None:
    try:
        TOTAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with TOTAL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"- {timestamp} {message}\n")
    except OSError:
        pass


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"training data missing columns: {sorted(missing)}")
    return df


def load_test(path: Path = TEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"test data missing column: {ID_COLUMN}")
    return df


def feature_columns(train_df: pd.DataFrame) -> list[str]:
    cols = [c for c in train_df.columns if c not in {ID_COLUMN, TARGET_COLUMN}]
    if len(cols) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} feature columns, got {len(cols)}: {cols}")
    return cols


def validate_features(df: pd.DataFrame, columns: Iterable[str]) -> None:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"data missing feature columns: {missing}")
    if df[columns].isna().any().any():
        raise ValueError("feature matrix contains missing values")


def build_extra_trees_model(random_state: int = 2026, n_estimators: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=None,
        max_features=None,
        min_samples_leaf=1,
        n_jobs=1,
    )


def build_hgb_model(random_state: int = 2026, max_iter: int = 350) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.03,
        categorical_features=list(range(FEATURE_COUNT)),
    )


def align_proba(model, proba: np.ndarray, label_order: Iterable[int] = LABELS) -> np.ndarray:
    label_order = [int(x) for x in label_order]
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    aligned = np.zeros((proba.shape[0], len(label_order)), dtype=np.float32)
    for out_idx, label in enumerate(label_order):
        aligned[:, out_idx] = proba[:, class_to_index[label]]
    return aligned


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if abs(temperature - 1.0) <= 1e-12:
        return proba.astype(np.float32, copy=False)
    scaled = np.power(np.clip(proba, 1e-12, 1.0), 1.0 / temperature).astype(np.float32, copy=False)
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled_sum = np.clip(scaled_sum, 1e-12, None)
    return scaled / scaled_sum


def apply_class_bias(proba: np.ndarray, class_bias: Iterable[float]) -> np.ndarray:
    bias = np.asarray(list(class_bias), dtype=np.float32)
    if bias.ndim != 1:
        raise ValueError(f"class_bias must be 1D, got shape {bias.shape}")
    if not np.isfinite(bias).all() or (bias <= 0).any():
        raise ValueError(f"class_bias must contain positive finite values, got {bias.tolist()}")
    if proba.shape[1] != len(bias):
        raise ValueError(f"class_bias length {len(bias)} does not match probability columns {proba.shape[1]}")
    scaled = proba * bias[None, :]
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled_sum = np.clip(scaled_sum, 1e-12, None)
    return scaled / scaled_sum


def _predict_tree_bundle_proba(bundle: Mapping[str, object], X: pd.DataFrame | np.ndarray) -> np.ndarray:
    model_type = bundle.get("model_type", "tree_blend")
    if model_type not in {"tree_blend", "blend"}:
        raise ValueError(f"unknown tree bundle type: {model_type}")
    models = bundle["models"]
    weights = bundle["blend_weights"]
    if len(models) != len(weights):
        raise ValueError("tree blend weights do not match model count")
    labels = bundle.get("labels", LABELS)
    proba = np.zeros((len(X), len(labels)), dtype=np.float32)
    for weight, model in zip(weights, models):
        proba += float(weight) * align_proba(model, model.predict_proba(X), labels)
    return proba


def _predict_torch_ensemble_proba(bundle: Mapping[str, object], X: np.ndarray, device=None) -> np.ndarray:
    from tabular_nn import predict_torch_ensemble_proba

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.int64, copy=True)
    else:
        X = np.asarray(X, dtype=np.int64)
    fold_bundles = bundle.get("fold_models", [])
    return predict_torch_ensemble_proba(fold_bundles, X, device=device)


def predict_bundle_proba(
    bundle: Mapping[str, object],
    X: pd.DataFrame | np.ndarray,
    device=None,
) -> np.ndarray:
    if "selected_model_bundle" in bundle:
        bundle = bundle["selected_model_bundle"]

    model_type = bundle.get("model_type")
    if model_type in {"tree_blend", "blend"}:
        proba = _predict_tree_bundle_proba(bundle, X)
    elif model_type == "torch_ensemble":
        proba = _predict_torch_ensemble_proba(bundle, X, device=device)
    elif model_type == "fusion":
        components = bundle.get("components", {})
        weights = bundle.get("component_weights", {})
        if not components or not weights:
            raise ValueError("fusion bundle missing components or weights")
        labels = bundle.get("labels", LABELS)
        proba = np.zeros((len(X), len(labels)), dtype=np.float32)
        for name, component in components.items():
            weight = float(weights[name])
            proba += weight * predict_bundle_proba(component, X, device=device)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    proba = apply_temperature(proba, float(bundle.get("temperature", 1.0)))
    proba = apply_class_bias(proba, bundle.get("class_bias", [1.0] * proba.shape[1]))
    return proba


def predict_from_bundle(
    bundle: Mapping[str, object],
    X: pd.DataFrame | np.ndarray,
    device=None,
) -> np.ndarray:
    labels = np.asarray(bundle.get("labels", LABELS), dtype=int)
    proba = predict_bundle_proba(bundle, X, device=device)
    return labels[np.argmax(proba, axis=1)]


def dump_joblib_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    joblib.dump(payload, tmp_path, compress=3)
    tmp_path.replace(path)


def write_csv_atomic(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_csv(tmp_path, **kwargs)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
