"""Prediction entrypoint for the ISCC binary vulnerability v1.4 pipeline."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from tqdm import tqdm

from byte_features import rows_to_byte_matrix
from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import (
    FUSION_CONFIG_NAME,
    LABEL_MODEL_NAME,
    CWE_MODEL_NAME,
    LEGACY_CWE_MODEL_NAME,
    LEGACY_LABEL_MODEL_NAME,
    LEGACY_TEST_CACHE_NAME,
    MODEL_VERSION,
    NEURAL_BUNDLE_NAME,
    SUBMISSION_NAME,
    TABULAR_BUNDLE_NAME,
    TEST_BYTE_CACHE_NAME,
    TEST_CACHE_NAME,
    ensure_model_dir,
)
from nn_models import (
    ByteMetaMultiTaskNet,
    FusionMLP,
    TabularNormalizer,
    apply_tabular_normalizer,
    predict_fusion_mlp,
    predict_multitask,
)
from utils import read_json

ROOT = Path(__file__).resolve().parents[1]
TEST_CSV = ROOT / "test.csv"
BINARIES_DIR = ROOT / "binaries"
MODEL_DIR = ROOT / "模型"
OUTPUT_DIR = ROOT / "提交结果"
OUTPUT_CSV = OUTPUT_DIR / SUBMISSION_NAME
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rows_to_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, List[str]]:
    feature_columns = get_feature_columns()
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    binary_ids: List[str] = []
    for index, row in enumerate(tqdm(rows, desc="Extracting test tabular features", total=len(rows))):
        binary_id = row["binary_id"]
        binary_ids.append(binary_id)
        feats = extract_features(binary_path(BINARIES_DIR, binary_id))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)
    return matrix, binary_ids


def _load_or_build_tabular_cache(rows: List[Dict[str, str]]) -> Dict[str, object]:
    versioned_cache = MODEL_DIR / TEST_CACHE_NAME
    legacy_cache = MODEL_DIR / LEGACY_TEST_CACHE_NAME
    if versioned_cache.exists():
        return joblib.load(versioned_cache)
    if legacy_cache.exists():
        cache = joblib.load(legacy_cache)
        joblib.dump(cache, versioned_cache)
        return cache

    X, binary_ids = _rows_to_matrix(rows)
    cache = {"X": X, "binary_ids": binary_ids, "feature_columns": get_feature_columns()}
    joblib.dump(cache, versioned_cache)
    return cache


def _load_or_build_byte_cache(rows: List[Dict[str, str]], byte_length: int) -> Dict[str, object]:
    cache_path = MODEL_DIR / TEST_BYTE_CACHE_NAME
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_length = int(cache.get("byte_length", 0))
        cached_matrix = cache.get("X_byte")
        if cached_length == byte_length and getattr(cached_matrix, "shape", (0, 0))[1] == byte_length:
            return cache
        print("warning: test byte cache length differs from current config; rebuilding byte cache.")

    X_byte, binary_ids = rows_to_byte_matrix(rows, BINARIES_DIR, byte_length=byte_length, desc="Extracting test byte windows")
    cache = {
        "X_byte": X_byte,
        "binary_ids": binary_ids,
        "byte_length": byte_length,
    }
    joblib.dump(cache, cache_path)
    return cache


def _torch_load(path: Path, device: torch.device) -> Dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _aligned_positive_probability(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        return np.zeros(X.shape[0], dtype=np.float32)
    if 1 in classes:
        positive_index = int(np.where(classes == 1)[0][0])
    else:
        positive_index = min(1, proba.shape[1] - 1)
    return np.asarray(proba[:, positive_index], dtype=np.float32)


def _aligned_cwe_probability(model, X: np.ndarray, num_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=np.float32)
    aligned = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    model_classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    for source_index, class_index in enumerate(model_classes):
        class_int = int(class_index)
        if 0 <= class_int < num_classes:
            aligned[:, class_int] = raw[:, source_index]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


def _load_neural_model() -> Tuple[ByteMetaMultiTaskNet, TabularNormalizer, List[str], int]:
    bundle_path = MODEL_DIR / NEURAL_BUNDLE_NAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing {MODEL_VERSION} neural model: {bundle_path}")
    bundle = _torch_load(bundle_path, DEVICE)
    model = ByteMetaMultiTaskNet(**bundle["model_config"]).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    normalizer = TabularNormalizer(
        mean=bundle["normalizer"]["mean"].cpu().numpy().astype(np.float32),
        std=bundle["normalizer"]["std"].cpu().numpy().astype(np.float32),
    )
    return model, normalizer, list(bundle["cwe_classes"]), int(bundle["byte_length"])


def _load_tabular_models() -> Tuple[object, object]:
    bundle_path = MODEL_DIR / TABULAR_BUNDLE_NAME
    if bundle_path.exists():
        bundle = joblib.load(bundle_path)
        label_file = bundle.get("label_model_file", LABEL_MODEL_NAME)
        cwe_file = bundle.get("cwe_model_file", CWE_MODEL_NAME)
    else:
        label_file = LABEL_MODEL_NAME if (MODEL_DIR / LABEL_MODEL_NAME).exists() else LEGACY_LABEL_MODEL_NAME
        cwe_file = CWE_MODEL_NAME if (MODEL_DIR / CWE_MODEL_NAME).exists() else LEGACY_CWE_MODEL_NAME
    label_bundle = joblib.load(MODEL_DIR / label_file)
    cwe_bundle = joblib.load(MODEL_DIR / cwe_file)
    return label_bundle["model"], cwe_bundle["model"]


def _load_fusion_mlp(num_cwe_classes: int) -> FusionMLP:
    fusion_path = MODEL_DIR / "fusion_mlp_v1.4.pt"
    if not fusion_path.exists():
        raise FileNotFoundError(f"missing fusion MLP model: {fusion_path}")
    bundle = _torch_load(fusion_path, DEVICE)
    model = FusionMLP(num_cwe_classes=num_cwe_classes).to(DEVICE)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def _report_submission_delta(output_csv: Path) -> None:
    previous_candidates = [OUTPUT_DIR / "submission_v1.3.csv", OUTPUT_DIR / "submission_v1.2.csv", OUTPUT_DIR / "submission_v1.1.csv"]
    previous_csv = next((path for path in previous_candidates if path.exists()), None)
    if previous_csv is None or not output_csv.exists():
        return

    with previous_csv.open("r", encoding="utf-8", newline="") as f:
        prev_rows = list(csv.DictReader(f))
    with output_csv.open("r", encoding="utf-8", newline="") as f:
        curr_rows = list(csv.DictReader(f))

    if len(prev_rows) != len(curr_rows):
        print(f"note: delta skipped because row counts differ ({len(prev_rows)} vs {len(curr_rows)}).")
        return

    prev_by_id = {row["binary_id"]: row for row in prev_rows}
    label_diff = 0
    cwe_diff = 0
    both_positive_cwe_diff = 0
    for row in curr_rows:
        prev = prev_by_id.get(row["binary_id"])
        if prev is None:
            continue
        if prev["label"] != row["label"]:
            label_diff += 1
        if prev["cwe_id"] != row["cwe_id"]:
            cwe_diff += 1
        if prev["label"] == "1" and row["label"] == "1" and prev["cwe_id"] != row["cwe_id"]:
            both_positive_cwe_diff += 1

    print(
        f"delta vs {previous_csv.name}: "
        f"label_diff={label_diff}, cwe_diff={cwe_diff}, "
        f"both_positive_cwe_diff={both_positive_cwe_diff}"
    )


def main() -> None:
    ensure_model_dir(MODEL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fusion_config = read_json(MODEL_DIR / FUSION_CONFIG_NAME)
    neural_model, normalizer, cwe_classes, byte_length = _load_neural_model()
    label_model, cwe_model = _load_tabular_models()

    rows = read_csv_rows(TEST_CSV)
    tabular_cache = _load_or_build_tabular_cache(rows)
    X = np.asarray(tabular_cache["X"], dtype=np.float32)
    binary_ids = list(tabular_cache["binary_ids"])

    byte_cache = _load_or_build_byte_cache(rows, byte_length)
    X_byte = np.asarray(byte_cache["X_byte"], dtype=np.uint8)
    X_neural = apply_tabular_normalizer(X, normalizer)

    tree_label_probs = _aligned_positive_probability(label_model, X)
    tree_cwe_probs = _aligned_cwe_probability(cwe_model, X, len(cwe_classes))
    neural_label_probs, neural_cwe_probs = predict_multitask(
        neural_model,
        X_byte,
        X_neural,
        batch_size=int(fusion_config.get("batch_size", 64)),
        device=DEVICE,
        desc=f"Predict neural {MODEL_VERSION}",
    )

    fusion_mode = fusion_config.get("fusion_mode", "scalar")
    if fusion_mode == "mlp":
        fusion_mlp = _load_fusion_mlp(len(cwe_classes))
        label_probs, cwe_probs = predict_fusion_mlp(
            fusion_mlp,
            tree_label_probs,
            tree_cwe_probs,
            neural_label_probs,
            neural_cwe_probs,
            batch_size=int(fusion_config.get("batch_size", 64)),
            device=DEVICE,
        )
    else:
        label_probs = (
            float(fusion_config["neural_label_weight"]) * neural_label_probs
            + float(fusion_config["tree_label_weight"]) * tree_label_probs
        )
        cwe_probs = (
            float(fusion_config["neural_cwe_weight"]) * neural_cwe_probs
            + float(fusion_config["tree_cwe_weight"]) * tree_cwe_probs
        )

    label_pred = (label_probs >= float(fusion_config["fusion_threshold"])).astype(int)
    cwe_pred = [""] * len(rows)
    positive_index = np.where(label_pred == 1)[0]
    for index in positive_index:
        cwe_pred[index] = cwe_classes[int(cwe_probs[index].argmax())]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["binary_id", "label", "cwe_id"])
        for binary_id, label, cwe_id in zip(binary_ids, label_pred.tolist(), cwe_pred):
            writer.writerow([binary_id, int(label), cwe_id])

    print(f"wrote {OUTPUT_CSV}")
    _report_submission_delta(OUTPUT_CSV)


if __name__ == "__main__":
    main()
