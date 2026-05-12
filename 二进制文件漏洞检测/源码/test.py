"""Prediction entrypoint for v2.6: weighted ensemble + scalar fusion + TTA."""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch

# Deterministic inference for reproducibility
os.environ["PYTHONHASHSEED"] = "42"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from utils import tqdm

from byte_features import DEFAULT_BYTE_LENGTH, rows_to_byte_matrix, rows_to_byte_matrix_tta
from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import (
    CWE_MAPPING_NAME, FEATURE_COLUMNS_NAME, FUSION_CONFIG_NAME,
    MODEL_VERSION, SUBMISSION_NAME, TABULAR_BUNDLE_NAME,
    UNIVERSAL_TEST_CACHE_NAME, UNIVERSAL_TEST_BYTE_CACHE_NAME,
    ensure_model_dir, seed_label_model, seed_cwe_model,
    seed_neural_bundle, seed_fusion_mlp,
)
from nn_models import (
    ByteMetaMultiTaskNet, TabularNormalizer,
    apply_tabular_normalizer, predict_multitask,
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
        bid = row["binary_id"]
        binary_ids.append(bid)
        feats = extract_features(binary_path(BINARIES_DIR, bid))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)
    return matrix, binary_ids


def _load_or_build_tabular_cache(rows: List[Dict[str, str]]) -> Dict[str, object]:
    """Load test tabular features from universal cache (test data never changes)."""
    cache_path = MODEL_DIR / UNIVERSAL_TEST_CACHE_NAME
    current_cols = get_feature_columns()
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_cols = list(cache.get("feature_columns", []))
        if len(cached_cols) == len(current_cols):
            return cache
        print(f"Test feature columns changed ({len(cached_cols)} -> {len(current_cols)}), rebuilding cache...")
    X, binary_ids = _rows_to_matrix(rows)
    cache = {"X": X, "binary_ids": binary_ids, "feature_columns": current_cols}
    joblib.dump(cache, cache_path)
    return cache


def _load_or_build_byte_cache(rows: List[Dict[str, str]], byte_length: int) -> Dict[str, object]:
    """Load test byte features from universal cache (test data never changes)."""
    cache_path = MODEL_DIR / UNIVERSAL_TEST_BYTE_CACHE_NAME
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_length = int(cache.get("byte_length", 0))
        cached_matrix = cache.get("X_byte")
        if cached_length == byte_length and getattr(cached_matrix, "shape", (0, 0))[1] == byte_length:
            return cache
        print("warning: test byte cache mismatch; rebuilding.")
    X_byte, binary_ids = rows_to_byte_matrix(rows, BINARIES_DIR, byte_length=byte_length, desc="Extracting test byte windows")
    cache = {"X_byte": X_byte, "binary_ids": binary_ids, "byte_length": byte_length}
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
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = min(1, proba.shape[1] - 1)
    return np.asarray(proba[:, pos_idx], dtype=np.float32)


def _aligned_cwe_probability(model, X: np.ndarray, num_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=np.float32)
    aligned = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    model_classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    for si, ci in enumerate(model_classes):
        ci_int = int(ci)
        if 0 <= ci_int < num_classes:
            aligned[:, ci_int] = raw[:, si]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


def _predict_neural_tta(
    model: ByteMetaMultiTaskNet,
    byte_matrices: List[np.ndarray],
    tabular_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Average neural predictions across TTA byte windows."""
    all_label = []
    all_cwe = []
    for w_idx, byte_mat in enumerate(byte_matrices):
        lp, cp = predict_multitask(model, byte_mat, tabular_matrix,
                                   batch_size=batch_size, device=device,
                                   desc=f"{desc} [window {w_idx+1}/{len(byte_matrices)}]")
        all_label.append(lp)
        all_cwe.append(cp)
    return (np.mean(all_label, axis=0).astype(np.float32),
            np.mean(all_cwe, axis=0).astype(np.float32))


def main():
    ensure_model_dir(MODEL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fusion_config = read_json(MODEL_DIR / FUSION_CONFIG_NAME)
    fusion_mode = fusion_config.get("fusion_mode", "scalar")
    seeds = fusion_config.get("seeds", [42, 123, 456])
    byte_length = int(fusion_config.get("byte_length", 8192))
    batch_size = int(fusion_config.get("batch_size", 64))
    use_tta = int(fusion_config.get("tta_windows", 0)) > 1
    tta_windows = int(fusion_config.get("tta_windows", 3))

    print(f"v2.6 prediction: fusion_mode={fusion_mode}, seeds={seeds}, tta={use_tta}({tta_windows} windows)")
    print(f"device={DEVICE}, byte_length={byte_length}, batch_size={batch_size}")

    rows = read_csv_rows(TEST_CSV)
    cwe_mapping = read_json(MODEL_DIR / CWE_MAPPING_NAME)
    cwe_classes = list(cwe_mapping["classes"])
    print(f"Test samples: {len(rows)}, CWE classes: {len(cwe_classes)}")

    # Tabular cache
    tabular_cache = _load_or_build_tabular_cache(rows)
    X = np.asarray(tabular_cache["X"], dtype=np.float32)
    binary_ids = list(tabular_cache["binary_ids"])

    # Byte cache (single window for tabular, TTA windows for neural)
    byte_cache = _load_or_build_byte_cache(rows, byte_length)
    X_byte_base = np.asarray(byte_cache["X_byte"], dtype=np.uint8)

    if use_tta:
        byte_tta = rows_to_byte_matrix_tta(rows, BINARIES_DIR, byte_length=byte_length,
                                            num_windows=tta_windows, desc="TTA byte windows")
        byte_windows = byte_tta  # List of matrices
    else:
        byte_windows = [X_byte_base]

    # Load per-seed models and accumulate predictions
    all_tree_label = []
    all_tree_cwe = []
    all_neural_label = []
    all_neural_cwe = []

    for seed in seeds:
        print(f"\n--- Predicting with seed={seed} ---")
        # Tabular
        label_bundle = joblib.load(MODEL_DIR / seed_label_model(seed))
        cwe_bundle = joblib.load(MODEL_DIR / seed_cwe_model(seed))
        label_model = label_bundle["model"]
        cwe_model = cwe_bundle["model"]

        tree_lp = _aligned_positive_probability(label_model, X)
        tree_cp = _aligned_cwe_probability(cwe_model, X, len(cwe_classes))
        all_tree_label.append(tree_lp)
        all_tree_cwe.append(tree_cp)

        # Neural
        neural_bundle = _torch_load(MODEL_DIR / seed_neural_bundle(seed), DEVICE)
        neural_model = ByteMetaMultiTaskNet(**neural_bundle["model_config"]).to(DEVICE)
        neural_model.load_state_dict(neural_bundle["state_dict"])
        neural_model.eval()
        normalizer = TabularNormalizer(
            mean=neural_bundle["normalizer"]["mean"].cpu().numpy().astype(np.float32),
            std=neural_bundle["normalizer"]["std"].cpu().numpy().astype(np.float32),
        )
        X_neural = apply_tabular_normalizer(X, normalizer)

        if use_tta:
            neural_lp, neural_cp = _predict_neural_tta(
                neural_model, byte_windows, X_neural,
                batch_size=batch_size, device=DEVICE, desc=f"Neural TTA s={seed}")
        else:
            neural_lp, neural_cp = predict_multitask(
                neural_model, X_byte_base, X_neural,
                batch_size=batch_size, device=DEVICE, desc=f"Neural s={seed}")
        all_neural_label.append(neural_lp)
        all_neural_cwe.append(neural_cp)

    # Weighted average across seeds (by scalar_cwe performance)
    seed_weights_dict = fusion_config.get("seed_weights", {})
    if seed_weights_dict:
        sw = np.array([seed_weights_dict.get(str(s), 1.0/len(seeds)) for s in seeds])
        sw = sw / sw.sum()  # normalize
        print(f"\nWeighted ensemble: {' '.join(f's{s}={w:.3f}' for s, w in zip(seeds, sw))}")
    else:
        sw = np.ones(len(seeds)) / len(seeds)

    avg_tree_label = np.average(all_tree_label, axis=0, weights=sw).astype(np.float32)
    avg_tree_cwe = np.average(all_tree_cwe, axis=0, weights=sw).astype(np.float32)
    avg_neural_label = np.average(all_neural_label, axis=0, weights=sw).astype(np.float32)
    avg_neural_cwe = np.average(all_neural_cwe, axis=0, weights=sw).astype(np.float32)

    print(f"\nUsing scalar fusion (weighted ensemble, {len(seeds)} seeds)")
    label_probs = (float(fusion_config["scalar_neural_label_weight"]) * avg_neural_label +
                   float(fusion_config["scalar_tree_label_weight"]) * avg_tree_label)

    global_nw_cwe = float(fusion_config["scalar_neural_cwe_weight"])
    cwe_probs = (global_nw_cwe * avg_neural_cwe +
                 float(fusion_config["scalar_tree_cwe_weight"]) * avg_tree_cwe)

    fusion_threshold = float(fusion_config.get("fusion_threshold", 0.5))

    # Generate submission
    label_pred = (label_probs >= fusion_threshold).astype(int)
    cwe_pred = [""] * len(rows)
    for idx in np.where(label_pred == 1)[0]:
        cwe_pred[idx] = cwe_classes[int(cwe_probs[idx].argmax())]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["binary_id", "label", "cwe_id"])
        for bid, lbl, cwe_id in zip(binary_ids, label_pred.tolist(), cwe_pred):
            writer.writerow([bid, int(lbl), cwe_id])

    print(f"\nWrote {OUTPUT_CSV}")
    n_pos = int(label_pred.sum())
    print(f"  label=1: {n_pos} ({100*n_pos/len(rows):.1f}%)")
    print(f"  label=0: {len(rows) - n_pos} ({100*(len(rows)-n_pos)/len(rows):.1f}%)")

    # Delta vs v1.4
    v14_csv = OUTPUT_DIR / "submission_v1.4.csv"
    if v14_csv.exists():
        with v14_csv.open("r", encoding="utf-8", newline="") as f:
            v14_rows = list(csv.DictReader(f))
        with OUTPUT_CSV.open("r", encoding="utf-8", newline="") as f:
            curr_rows = list(csv.DictReader(f))
        if len(v14_rows) == len(curr_rows):
            v14_by_id = {r["binary_id"]: r for r in v14_rows}
            label_diff = sum(1 for r in curr_rows if r["label"] != v14_by_id[r["binary_id"]]["label"])
            cwe_diff = sum(1 for r in curr_rows if r["cwe_id"] != v14_by_id[r["binary_id"]]["cwe_id"])
            both_pos_diff = sum(1 for r in curr_rows
                                if r["label"] == "1" and v14_by_id[r["binary_id"]]["label"] == "1"
                                and r["cwe_id"] != v14_by_id[r["binary_id"]]["cwe_id"])
            print(f"  delta vs v1.4: label_diff={label_diff}, cwe_diff={cwe_diff}, both_positive_cwe_diff={both_pos_diff}")


if __name__ == "__main__":
    main()
