"""Training entrypoint for the ISCC binary vulnerability task."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_name, "1")

import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import (
    CWE_MAPPING_NAME,
    CWE_MODEL_NAME,
    FEATURE_COLUMNS_NAME,
    LABEL_MODEL_NAME,
    TRAIN_CACHE_NAME,
    ensure_model_dir,
)
from utils import write_json


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
BINARIES_DIR = ROOT / "binaries"
MODEL_DIR = ROOT / "模型"


def _rows_to_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feature_columns = get_feature_columns()
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    y_label = np.zeros(len(rows), dtype=np.int32)
    cwe_ids: List[str] = []
    binary_ids: List[str] = []

    for index, row in enumerate(tqdm(rows, desc="Extracting train features", total=len(rows))):
        binary_ids.append(row["binary_id"])
        y_label[index] = int(row["label"])
        cwe_ids.append(row["cwe_id"])
        feats = extract_features(binary_path(BINARIES_DIR, row["binary_id"]))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)

    return matrix, y_label, cwe_ids, binary_ids


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_threshold = 0.50
    best_score = -1.0
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _train_binary_model(X: np.ndarray, y: np.ndarray) -> Tuple[HistGradientBoostingClassifier, float]:
    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    params = dict(
        random_state=42,
        learning_rate=0.06,
        max_iter=250,
        max_depth=8,
        min_samples_leaf=20,
        verbose=1,
    )
    probe = HistGradientBoostingClassifier(**params)
    probe.fit(X_tr, y_tr)
    proba = probe.predict_proba(X_val)[:, 1]
    threshold = _best_threshold(y_val, proba)

    final_model = HistGradientBoostingClassifier(**params)
    final_model.fit(X, y)
    return final_model, threshold


def _train_cwe_model(X: np.ndarray, cwe_ids: List[str]) -> Tuple[RandomForestClassifier, List[str]]:
    classes = sorted(set(cwe_ids))
    mapping = {name: index for index, name in enumerate(classes)}
    y = np.asarray([mapping[cwe] for cwe in cwe_ids], dtype=np.int32)
    class_counts = np.bincount(y, minlength=len(classes))
    min_class_count = int(class_counts.min()) if len(class_counts) else 0
    if min_class_count < 2:
        print(
            "warning: at least one CWE class has fewer than 2 samples; "
            "CWE model will use fixed balanced class weights."
        )
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(classes)),
        y=y,
    )
    class_weight = {index: float(weight) for index, weight in enumerate(class_weights)}
    total_trees = 300
    batch_size = 25
    model = RandomForestClassifier(
        n_estimators=batch_size,
        random_state=42,
        n_jobs=1,
        class_weight=class_weight,
        min_samples_leaf=1,
        warm_start=True,
        verbose=0,
    )
    fitted_trees = 0
    targets = list(range(batch_size, total_trees + 1, batch_size))
    if targets[-1] != total_trees:
        targets.append(total_trees)
    with tqdm(total=total_trees, desc="Training CWE forest", unit="tree") as tree_bar:
        for target_trees in targets:
            model.n_estimators = target_trees
            model.fit(X, y)
            tree_bar.update(target_trees - fitted_trees)
            fitted_trees = target_trees
            tree_bar.set_postfix_str(f"{fitted_trees}/{total_trees}")
    return model, classes


def main() -> None:
    ensure_model_dir(MODEL_DIR)
    rows = read_csv_rows(TRAIN_CSV)
    cache_path = MODEL_DIR / TRAIN_CACHE_NAME

    with tqdm(total=4, desc="Training pipeline", unit="stage") as pipeline:
        pipeline.set_postfix_str("load cache")
        if cache_path.exists():
            cache = joblib.load(cache_path)
            X = cache["X"]
            y_label = cache["y_label"]
            cwe_ids = cache["cwe_ids"]
            feature_columns = cache["feature_columns"]
        else:
            X, y_label, cwe_ids, _ = _rows_to_matrix(rows)
            feature_columns = get_feature_columns()
            joblib.dump(
                {
                    "X": X,
                    "y_label": y_label,
                    "cwe_ids": cwe_ids,
                    "feature_columns": feature_columns,
                },
                cache_path,
            )
        pipeline.update(1)

        pipeline.set_postfix_str("train label")
        label_model, threshold = _train_binary_model(X, y_label)
        pipeline.update(1)

        pipeline.set_postfix_str("train cwe")
        positive_mask = y_label == 1
        positive_cwe_ids = [cwe_ids[i] for i in range(len(cwe_ids)) if positive_mask[i]]
        cwe_model, cwe_classes = _train_cwe_model(X[positive_mask], positive_cwe_ids)
        pipeline.update(1)

        pipeline.set_postfix_str("save artifacts")
        joblib.dump(
            {
                "model": label_model,
                "threshold": threshold,
                "feature_columns": feature_columns,
            },
            MODEL_DIR / LABEL_MODEL_NAME,
        )
        joblib.dump(
            {
                "model": cwe_model,
                "feature_columns": feature_columns,
                "classes": cwe_classes,
            },
            MODEL_DIR / CWE_MODEL_NAME,
        )
        write_json(MODEL_DIR / FEATURE_COLUMNS_NAME, feature_columns)
        write_json(
            MODEL_DIR / CWE_MAPPING_NAME,
            {
                "classes": cwe_classes,
                "class_to_index": {name: index for index, name in enumerate(cwe_classes)},
            },
        )
        pipeline.update(1)

    print(f"saved models to {MODEL_DIR}")
    print(f"binary threshold: {threshold:.2f}")


if __name__ == "__main__":
    main()
