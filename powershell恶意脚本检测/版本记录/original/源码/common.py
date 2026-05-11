from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data_train.csv"
TEST_PATH = PROJECT_ROOT / "data_test.csv"
MODEL_DIR = PROJECT_ROOT / "模型"
RESULT_DIR = PROJECT_ROOT / "提交结果"
LOG_PATH = PROJECT_ROOT / "ACTION_LOG.md"

LABELS = [0, 1, 2]
TARGET_COLUMN = "label"
ID_COLUMN = "name"
FEATURE_COUNT = 15


def append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {timestamp} {message}\n")


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


def apply_class_bias(proba: np.ndarray, class_bias: Iterable[float]) -> np.ndarray:
    bias = np.asarray(list(class_bias), dtype=np.float32)
    if bias.shape != (proba.shape[1],):
        raise ValueError(f"class_bias length {len(bias)} does not match probability columns {proba.shape[1]}")
    if not np.isfinite(bias).all() or (bias <= 0).any():
        raise ValueError(f"class_bias must contain positive finite values, got {bias.tolist()}")
    return proba * bias[None, :]


def predict_from_bundle(bundle: dict, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    label_order = np.asarray(bundle.get("labels", LABELS), dtype=int)
    model_type = bundle.get("model_type", "single")

    if model_type == "single":
        model = bundle["model"]
        proba = align_proba(model, model.predict_proba(X), label_order)
    elif model_type == "blend":
        models = bundle["models"]
        weights = bundle["blend_weights"]
        proba = np.zeros((len(X), len(label_order)), dtype=np.float32)
        for weight, model in zip(weights, models):
            proba += float(weight) * align_proba(model, model.predict_proba(X), label_order)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    proba = apply_class_bias(proba, bundle.get("class_bias", [1.0] * len(label_order)))
    return label_order[np.argmax(proba, axis=1)]


def dump_joblib_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    joblib.dump(payload, tmp_path)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
