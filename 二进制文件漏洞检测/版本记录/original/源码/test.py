"""Prediction entrypoint for the ISCC binary vulnerability task."""

from __future__ import annotations

import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    os.environ.setdefault(_name, "1")

import csv
from pathlib import Path

import joblib
import numpy as np
from tqdm import tqdm

from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import CWE_MODEL_NAME, LABEL_MODEL_NAME, TEST_CACHE_NAME, ensure_model_dir


ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "test.csv"
BINARIES_DIR = ROOT / "binaries"
MODEL_DIR = ROOT / "模型"
OUTPUT_DIR = ROOT / "提交结果"
OUTPUT_CSV = OUTPUT_DIR / "submission_final.csv"


def _rows_to_matrix(rows):
    feature_columns = get_feature_columns()
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    binary_ids = []
    for index, row in enumerate(tqdm(rows, desc="Extracting test features", total=len(rows))):
        binary_ids.append(row["binary_id"])
        feats = extract_features(binary_path(BINARIES_DIR, row["binary_id"]))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)
    return matrix, binary_ids


def main() -> None:
    ensure_model_dir(MODEL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    label_bundle = joblib.load(MODEL_DIR / LABEL_MODEL_NAME)
    cwe_bundle = joblib.load(MODEL_DIR / CWE_MODEL_NAME)

    rows = read_csv_rows(TEST_CSV)
    cache_path = MODEL_DIR / TEST_CACHE_NAME
    if cache_path.exists():
        cache = joblib.load(cache_path)
        X = cache["X"]
        binary_ids = cache["binary_ids"]
    else:
        X, binary_ids = _rows_to_matrix(rows)
        joblib.dump({"X": X, "binary_ids": binary_ids}, cache_path)

    label_model = label_bundle["model"]
    threshold = float(label_bundle["threshold"])
    cwe_model = cwe_bundle["model"]
    cwe_classes = cwe_bundle["classes"]

    probabilities = label_model.predict_proba(X)[:, 1]
    label_pred = (probabilities >= threshold).astype(int)
    cwe_pred = [""] * len(rows)
    positive_index = np.where(label_pred == 1)[0]
    if len(positive_index) > 0:
        cwe_index_pred = cwe_model.predict(X[positive_index]).astype(int)
        for offset, class_index in zip(positive_index, cwe_index_pred):
            cwe_pred[offset] = cwe_classes[int(class_index)]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["binary_id", "label", "cwe_id"])
        for binary_id, label, cwe_id in zip(binary_ids, label_pred.tolist(), cwe_pred):
            writer.writerow([binary_id, int(label), cwe_id])

    print(f"wrote {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
