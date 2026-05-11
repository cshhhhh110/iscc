"""v2.5 training: v2.4 + per-class shrinkage scalar fusion + OOF analysis."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from utils import tqdm

from byte_features import DEFAULT_BYTE_LENGTH, rows_to_byte_matrix
from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import (
    CWE_MAPPING_NAME, CWE_MODEL_NAME, FEATURE_COLUMNS_NAME,
    FUSION_CONFIG_NAME, LABEL_MODEL_NAME,
    NEURAL_BUNDLE_NAME, PSEUDO_TRAIN_CSV, SUBMISSION_NAME,
    TABULAR_BUNDLE_NAME, TRAIN_BYTE_CACHE_NAME, TRAIN_CACHE_NAME,
    ensure_model_dir,
    seed_label_model, seed_cwe_model, seed_neural_bundle,
    seed_fusion_mlp, seed_tabular_bundle,
)
from nn_models import (
    ByteMetaMultiTaskNet, FocalLoss, FusionMLP, TabularNormalizer,
    apply_tabular_normalizer, build_cwe_class_weights, fit_tabular_normalizer,
    predict_fusion_mlp, predict_multitask,
)
from utils import write_json

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
TEST_CSV = ROOT / "test.csv"
BINARIES_DIR = ROOT / "binaries"
MODEL_DIR = ROOT / "模型"
OUTPUT_DIR = ROOT / "提交结果"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_GBDT_BACKEND = None
SEEDS = [42, 123, 202]
PSEUDO_LABEL_THRESH_HIGH = 0.90
PSEUDO_LABEL_THRESH_LOW = 0.10
PSEUDO_CWE_THRESH = 0.90
TTA_WINDOWS = 3



def _detect_gbdt_backend() -> str:
    global _GBDT_BACKEND
    if _GBDT_BACKEND is not None:
        return _GBDT_BACKEND
    for name, module in [("lightgbm", "lightgbm"), ("catboost", "catboost")]:
        try:
            __import__(module)
            _GBDT_BACKEND = name
            print(f"GBDT backend: {name}")
            return name
        except ImportError:
            continue
    raise ImportError("Neither catboost nor lightgbm is installed.")


def _build_cwe_gbdt(num_classes: int, class_weight_map: Dict[int, float], random_state: int):
    backend = _detect_gbdt_backend()
    if backend == "catboost":
        from catboost import CatBoostClassifier
        sample_weight_array = np.array(
            [class_weight_map.get(i, 1.0) for i in range(num_classes)], dtype=np.float64
        )
        return CatBoostClassifier(
            iterations=800, depth=8, learning_rate=0.06, l2_leaf_reg=3.0,
            random_seed=random_state, class_weights=list(sample_weight_array),
            loss_function="MultiClass", eval_metric="MultiClass",
            verbose=0, allow_writing_files=False,
        )
    else:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.06,
            num_leaves=64, min_child_samples=16, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
            class_weight=class_weight_map, random_state=random_state,
            n_jobs=4, verbose=-1,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train v1.5 ensemble with pseudo-labeling.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--byte-length", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--dropout", type=float, default=0.22)
    parser.add_argument("--byte-embedding-dim", type=int, default=48)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--cwe-loss-weight", type=float, default=2.0)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--fusion-epochs", type=int, default=30)
    parser.add_argument("--skip-mlp", action="store_true", default=True, help="Skip MLP fusion (scalar only)")
    parser.add_argument("--retrain-tree", action="store_true")
    parser.add_argument("--skip-pseudo", action="store_true", help="Skip pseudo-labeling, use cached pseudo_train.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="Seeds for ensemble")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Tabular feature extraction
# ---------------------------------------------------------------------------

def _rows_to_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feature_columns = get_feature_columns()
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    y_label = np.zeros(len(rows), dtype=np.int32)
    cwe_ids: List[str] = []
    binary_ids: List[str] = []
    for index, row in enumerate(tqdm(rows, desc="Extracting tabular features", total=len(rows))):
        binary_ids.append(row["binary_id"])
        y_label[index] = int(row["label"])
        cwe_ids.append(row.get("cwe_id", ""))
        feats = extract_features(binary_path(BINARIES_DIR, row["binary_id"]))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)
    return matrix, y_label, cwe_ids, binary_ids


def _find_matching_cache(cache_prefix: str, target_ids: List[str]) -> str | None:
    """Scan 模型/ for a cache file whose binary_ids match target_ids exactly (set equality)."""
    target_set = set(target_ids)
    for f in sorted(MODEL_DIR.glob(f"{cache_prefix}*.joblib"), key=lambda p: -p.stat().st_size):
        try:
            cache = joblib.load(f)
            cached_ids = cache.get("binary_ids", [])
            if len(cached_ids) == len(target_ids) and set(cached_ids) == target_set:
                return str(f)
        except Exception:
            continue
    return None


def _load_or_build_tabular_cache(rows: List[Dict[str, str]]) -> Dict[str, object]:
    cache_path = MODEL_DIR / TRAIN_CACHE_NAME
    current_feat_cols = get_feature_columns()
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_cols = list(cache.get("feature_columns", []))
        if len(cached_cols) == len(current_feat_cols):
            return cache
        print(f"Feature columns changed ({len(cached_cols)} -> {len(current_feat_cols)}), rebuilding cache...")

    target_ids = [r["binary_id"] for r in rows]
    # Try to reuse an existing cache with matching binary_ids AND same feature count
    reused = _find_matching_cache("train_features", target_ids)
    if reused:
        print(f"Reusing tabular cache: {Path(reused).name}")
        cache = joblib.load(reused)
        cached_cols = list(cache.get("feature_columns", []))
        if len(cached_cols) == len(current_feat_cols):
            joblib.dump(cache, cache_path)
            return cache
        print(f"Feature columns mismatch ({len(cached_cols)} vs {len(current_feat_cols)}), rebuilding...")

    X, y_label, cwe_ids, binary_ids = _rows_to_matrix(rows)
    cache = {"X": X, "y_label": y_label, "cwe_ids": cwe_ids, "binary_ids": binary_ids,
             "feature_columns": current_feat_cols}
    joblib.dump(cache, cache_path)
    return cache


def _load_or_build_byte_cache(rows: List[Dict[str, str]], byte_length: int) -> Dict[str, object]:
    cache_path = MODEL_DIR / TRAIN_BYTE_CACHE_NAME
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_length = int(cache.get("byte_length", 0))
        if cached_length == byte_length and cache.get("X_byte", np.empty(0)).shape[1] == byte_length:
            return cache

    target_ids = [r["binary_id"] for r in rows]
    reused = _find_matching_cache("train_bytes", target_ids)
    if reused:
        reused_cache = joblib.load(reused)
        if int(reused_cache.get("byte_length", 0)) == byte_length:
            print(f"Reusing byte cache: {Path(reused).name}")
            joblib.dump(reused_cache, cache_path)
            return reused_cache

    X_byte, binary_ids = rows_to_byte_matrix(rows, BINARIES_DIR, byte_length=byte_length, desc="Extracting byte windows")
    cache = {"X_byte": X_byte, "binary_ids": binary_ids, "byte_length": byte_length}
    joblib.dump(cache, cache_path)
    return cache


# ---------------------------------------------------------------------------
# Pseudo-labeling
# ---------------------------------------------------------------------------

def _pseudo_label_test_set(args: argparse.Namespace) -> List[Dict[str, str]]:
    """Use v1.5 ensemble models to generate high-confidence pseudo-labels for test.csv."""
    pseudo_csv = ROOT / PSEUDO_TRAIN_CSV
    if pseudo_csv.exists() and not args.retrain_tree:
        print(f"Loading cached pseudo-labels from {pseudo_csv}")
        return read_csv_rows(pseudo_csv)

    print("=== Phase 1: Pseudo-labeling test.csv with v1.5 models ===")
    # Use v1.5 test caches (already built during v1.5 prediction)
    v5_test_cache = MODEL_DIR / "test_features_v1.5.joblib"
    v5_byte_cache = MODEL_DIR / "test_bytes_v1.5.joblib"
    v5_fusion_config = MODEL_DIR / "fusion_config_v1.5.json"

    # Check v1.5 caches, fall back to v1.4
    test_cache_path = v5_test_cache if v5_test_cache.exists() else (MODEL_DIR / "test_features_v1.4.joblib")
    byte_cache_path = v5_byte_cache if v5_byte_cache.exists() else (MODEL_DIR / "test_bytes_v1.4.joblib")

    test_rows = read_csv_rows(TEST_CSV)
    if not test_cache_path.exists():
        print(f"Building test tabular cache at {test_cache_path}...")
        X_test, _, _, test_binary_ids = _rows_to_matrix(test_rows)
        joblib.dump({"X": X_test, "binary_ids": test_binary_ids}, test_cache_path)
    else:
        cache = joblib.load(test_cache_path)
        X_test = np.asarray(cache["X"], dtype=np.float32)
        test_binary_ids = list(cache["binary_ids"])

    if not byte_cache_path.exists():
        print(f"Building test byte cache at {byte_cache_path}...")
        Xb, _ = rows_to_byte_matrix(test_rows, BINARIES_DIR, byte_length=8192, desc="Test bytes v1.6")
        joblib.dump({"X_byte": Xb, "binary_ids": test_binary_ids, "byte_length": 8192}, byte_cache_path)
    else:
        Xb = np.asarray(joblib.load(byte_cache_path)["X_byte"], dtype=np.uint8)

    # Load v1.5 ensemble config
    from utils import read_json as _read_json
    v5_config = _read_json(v5_fusion_config) if v5_fusion_config.exists() else None
    if v5_config is None:
        raise FileNotFoundError("v1.5 fusion config not found; run v1.5 first for pseudo-labeling")
    v5_seeds = v5_config.get("seeds", [42, 123, 456])
    cwe_classes_path = MODEL_DIR / "cwe_mapping_v1.5.json"
    cwe_classes = list(_read_json(cwe_classes_path)["classes"])

    # Ensemble predict: average all v1.5 seeds
    all_tree_lp, all_tree_cp, all_neural_lp, all_neural_cp = [], [], [], []
    for seed in v5_seeds:
        lb = joblib.load(MODEL_DIR / seed_label_model(seed).replace("v1.6", "v1.5"))
        cb = joblib.load(MODEL_DIR / seed_cwe_model(seed).replace("v1.6", "v1.5"))
        all_tree_lp.append(_aligned_positive_probability_v4(lb["model"], X_test))
        all_tree_cp.append(_aligned_cwe_probability_v4(cb["model"], X_test, len(cwe_classes)))

        nb_path = MODEL_DIR / seed_neural_bundle(seed).replace("v1.6", "v1.5")
        nb = torch.load(nb_path, map_location=DEVICE, weights_only=False)
        nm = ByteMetaMultiTaskNet(**nb["model_config"]).to(DEVICE)
        nm.load_state_dict(nb["state_dict"])
        nm.eval()
        normalizer = TabularNormalizer(
            mean=nb["normalizer"]["mean"].cpu().numpy().astype(np.float32),
            std=nb["normalizer"]["std"].cpu().numpy().astype(np.float32),
        )
        Xn = apply_tabular_normalizer(X_test, normalizer)
        nlp, ncp = predict_multitask(nm, Xb, Xn, batch_size=128, device=DEVICE,
                                      desc=f"Pseudo-label predict s={seed}")
        all_neural_lp.append(nlp)
        all_neural_cp.append(ncp)

    # Average across seeds
    avg_tree_lp = np.mean(all_tree_lp, axis=0)
    avg_tree_cp = np.mean(all_tree_cp, axis=0)
    avg_neural_lp = np.mean(all_neural_lp, axis=0)
    avg_neural_cp = np.mean(all_neural_cp, axis=0)

    # Scalar fusion with v1.5 config weights
    lw_n = float(v5_config["scalar_neural_label_weight"])
    lw_t = float(v5_config["scalar_tree_label_weight"])
    cw_n = float(v5_config["scalar_neural_cwe_weight"])
    cw_t = float(v5_config["scalar_tree_cwe_weight"])
    label_probs = lw_n * avg_neural_lp + lw_t * avg_tree_lp
    cwe_probs = cw_n * avg_neural_cp + cw_t * avg_tree_cp

    # Filter with lower thresholds
    pseudo_rows: List[Dict[str, str]] = []
    for i, bid in enumerate(tqdm(test_binary_ids, desc="Filtering pseudo-labels")):
        lp = float(label_probs[i])
        if lp >= PSEUDO_LABEL_THRESH_HIGH:
            cwe_max = float(cwe_probs[i].max())
            if cwe_max >= PSEUDO_CWE_THRESH:
                pred_cwe = cwe_classes[int(cwe_probs[i].argmax())]
            else:
                continue
            pseudo_rows.append({"binary_id": bid, "label": "1", "cwe_id": pred_cwe})
        elif lp <= PSEUDO_LABEL_THRESH_LOW:
            pseudo_rows.append({"binary_id": bid, "label": "0", "cwe_id": ""})

    print(f"Pseudo-labeled {len(pseudo_rows)} / {len(test_binary_ids)} samples "
          f"({100*len(pseudo_rows)/len(test_binary_ids):.1f}%)")
    pos_count = sum(1 for r in pseudo_rows if r["label"] == "1")
    print(f"  label=1: {pos_count}, label=0: {len(pseudo_rows) - pos_count}")

    with open(pseudo_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["binary_id", "label", "cwe_id"])
        writer.writeheader()
        writer.writerows(pseudo_rows)
    print(f"Saved pseudo-labels to {pseudo_csv}")
    return pseudo_rows


# Helpers redefined here for v1.4 model loading (avoid circular imports from train.py)
def _aligned_positive_probability_v4(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        return np.zeros(X.shape[0], dtype=np.float32)
    if 1 in classes:
        pos_idx = int(np.where(classes == 1)[0][0])
    else:
        pos_idx = min(1, proba.shape[1] - 1)
    return np.asarray(proba[:, pos_idx], dtype=np.float32)


def _aligned_cwe_probability_v4(model, X: np.ndarray, num_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=np.float32)
    aligned = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    model_classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    for si, ci in enumerate(model_classes):
        ci_int = int(ci)
        if 0 <= ci_int < num_classes:
            aligned[:, ci_int] = raw[:, si]
    row_sum = aligned.sum(axis=1, keepdims=True)
    return aligned / np.maximum(row_sum, 1e-12)


# ---------------------------------------------------------------------------
# Helpers (copied from train.py v1.4 for independence)
# ---------------------------------------------------------------------------

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


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.arange(0.08, 0.93, 0.005)
    best_t, best_s = 0.50, -1.0
    for t in thresholds:
        s = f1_score(y_true, (proba >= t).astype(int))
        if s > best_s:
            best_s, best_t = s, float(t)
    return best_t


def _search_binary_fusion(y_true, tree_probs, neural_probs):
    best_w, best_t, best_s, best_a = 1.0, 0.5, -1.0, -1.0
    for nw in np.linspace(0.0, 1.0, 41):
        fused = nw * neural_probs + (1.0 - nw) * tree_probs
        t = _best_threshold(y_true, fused)
        s = f1_score(y_true, (fused >= t).astype(int))
        a = accuracy_score(y_true, (fused >= t).astype(int))
        if s > best_s or (s == best_s and a > best_a):
            best_s, best_a, best_w, best_t = s, a, float(nw), float(t)
    return best_w, best_t, best_s


def _search_cwe_fusion(y_true, tree_probs, neural_probs):
    if len(y_true) == 0:
        return 1.0, 0.0
    best_w, best_m, best_a = 1.0, -1.0, -1.0
    class_labels = list(range(tree_probs.shape[1]))
    for nw in np.linspace(0.0, 1.0, 41):
        fused = nw * neural_probs + (1.0 - nw) * tree_probs
        pred = fused.argmax(axis=1)
        macro = f1_score(y_true, pred, average="macro", labels=class_labels, zero_division=0)
        acc = accuracy_score(y_true, pred)
        if macro > best_m or (macro == best_m and acc > best_a):
            best_m, best_a, best_w = float(macro), float(acc), float(nw)
    return best_w, best_m


def _split_indices(y_label, cwe_ids, rng_seed=42):
    indices = np.arange(len(y_label))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=rng_seed, stratify=y_label)
    train_set = set(train_idx.tolist())
    val_list = list(val_idx.tolist())
    train_cwe_counts = {}
    for idx in train_idx:
        if y_label[idx] == 1:
            cwe = cwe_ids[idx]
            train_cwe_counts[cwe] = train_cwe_counts.get(cwe, 0) + 1
    moved = 0
    for cwe in sorted(set(cwe_ids)):
        if not cwe:
            continue
        if train_cwe_counts.get(cwe, 0) > 0:
            continue
        candidates = [idx for idx in val_list if y_label[idx] == 1 and cwe_ids[idx] == cwe]
        if candidates:
            val_list.remove(candidates[0])
            train_set.add(candidates[0])
            moved += 1
    if moved:
        print(f"  note: moved {moved} rare CWE samples into training fold (seed={rng_seed})")
    return np.array(sorted(train_set), dtype=np.int64), np.array(sorted(val_list), dtype=np.int64)


def _build_label_ensemble(random_state: int) -> VotingClassifier:
    hgb = HistGradientBoostingClassifier(
        random_state=random_state, learning_rate=0.045, max_iter=250,
        max_leaf_nodes=31, max_depth=9, min_samples_leaf=20,
        l2_regularization=0.03, early_stopping=True, validation_fraction=0.12, verbose=1,
    )
    return VotingClassifier(estimators=[("hgb", hgb)], voting="soft", n_jobs=1)


def _cwe_class_weight_map(y_cwe, num_classes):
    counts = np.bincount(y_cwe, minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights = np.clip(weights, 0.35, 18.0)
    return {i: float(w) for i, w in enumerate(weights)}


def _build_neural_model(tab_dim, num_cwe, dropout, byte_emb_dim):
    return ByteMetaMultiTaskNet(tabular_dim=tab_dim, num_cwe_classes=num_cwe,
                                dropout=dropout, byte_embedding_dim=byte_emb_dim)


# ---------------------------------------------------------------------------
# Single-seed training
# ---------------------------------------------------------------------------

def _train_tabular_one_seed(X, y_label, cwe_ids, seed, feature_columns):
    """Train tabular models for one seed. Returns (label_model, cwe_model, cwe_classes, threshold)."""
    print(f"\n--- Tabular training seed={seed} ---")
    indices = np.arange(len(y_label))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=seed, stratify=y_label)
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr, y_val = y_label[train_idx], y_label[val_idx]

    # Label model
    label_model = _build_label_ensemble(random_state=seed)
    label_model.fit(X_tr, y_tr)
    threshold = _best_threshold(y_val, _aligned_positive_probability(label_model, X_val))

    # Final label model on all data
    label_final = _build_label_ensemble(random_state=seed)
    label_final.fit(X, y_label)

    # CWE model
    pos_mask = y_label == 1
    pos_cwe_ids = [cwe_ids[i] for i in range(len(cwe_ids)) if pos_mask[i]]
    classes = sorted(set(pos_cwe_ids))
    mapping = {name: idx for idx, name in enumerate(classes)}
    y_cwe = np.asarray([mapping[c] for c in pos_cwe_ids], dtype=np.int32)
    class_weight_map = _cwe_class_weight_map(y_cwe, len(classes))

    full_y_cwe = np.full(len(cwe_ids), -1, dtype=np.int32)
    for i, cid in enumerate(cwe_ids):
        if cid:
            full_y_cwe[i] = mapping[cid]

    cwe_model = _build_cwe_gbdt(len(classes), class_weight_map, random_state=seed)
    cwe_model.fit(X[pos_mask], y_cwe)

    # Save per-seed tabular bundle
    label_path = MODEL_DIR / seed_label_model(seed)
    cwe_path = MODEL_DIR / seed_cwe_model(seed)
    joblib.dump({"model": label_final, "threshold": threshold, "feature_columns": feature_columns}, label_path)
    joblib.dump({"model": cwe_model, "feature_columns": feature_columns, "classes": classes,
                 "class_weight_map": class_weight_map, "model_family": _detect_gbdt_backend()}, cwe_path)

    return label_final, cwe_model, classes, threshold, train_idx, val_idx, mapping


def _train_neural_one_seed(X_byte, X_tab, y_label, y_cwe, train_idx, val_idx,
                           num_cwe_classes, seed, args):
    """Train one neural model with cosine annealing, warmup, and SWA."""
    print(f"\n--- Neural training seed={seed} ---")
    normalizer = fit_tabular_normalizer(X_tab[train_idx])
    X_tr_tab = apply_tabular_normalizer(X_tab[train_idx], normalizer)
    X_val_tab = apply_tabular_normalizer(X_tab[val_idx], normalizer)

    model = _build_neural_model(X_tr_tab.shape[1], num_cwe_classes, args.dropout, args.byte_embedding_dim).to(DEVICE)

    pos_count = float(y_label[train_idx].sum())
    neg_count = float(len(train_idx) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
    label_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cwe_weights = build_cwe_class_weights(y_cwe[train_idx], num_cwe_classes)
    cwe_criterion = FocalLoss(gamma=args.focal_gamma, alpha=cwe_weights).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=args.lr * 0.01)
    warmup_epochs = 3
    swa_start = max(1, args.epochs - 5)
    swa_state = None
    swa_n = 0

    best_state, best_score, best_metrics = None, -1.0, {}
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        # Warmup: linear LR ramp from 1% to 100% over first warmup_epochs
        if epoch <= warmup_epochs:
            scale = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = args.lr * (0.01 + 0.99 * scale)
        else:
            cos_scheduler.step(epoch - warmup_epochs - 1)

        model.train()
        total_loss = 0.0
        train_order = np.arange(len(train_idx))
        np.random.default_rng(seed * 100 + epoch).shuffle(train_order)
        total_batches = int(np.ceil(len(train_idx) / args.batch_size))
        with tqdm(total=total_batches, desc=f"Epoch {epoch}/{args.epochs} s={seed}", unit="batch") as progress:
            for batch_no, start in enumerate(range(0, len(train_order), args.batch_size), 1):
                batch_pos = train_order[start:start + args.batch_size]
                batch_idx = train_idx[batch_pos]
                byte_b = torch.from_numpy(X_byte[batch_idx]).to(DEVICE)
                tab_b = torch.from_numpy(X_tr_tab[batch_pos]).to(DEVICE)
                lbl_b = torch.from_numpy(y_label[batch_idx].astype(np.float32)).to(DEVICE)
                cwe_b = torch.from_numpy(y_cwe[batch_idx].astype(np.int64)).to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                label_logits, cwe_logits = model(byte_b, tab_b)
                l_loss = label_criterion(label_logits, lbl_b)
                pos_mask_b = cwe_b >= 0
                c_loss = cwe_criterion(cwe_logits[pos_mask_b], cwe_b[pos_mask_b]) if pos_mask_b.any() else torch.zeros((), device=DEVICE)
                loss = l_loss + args.cwe_loss_weight * c_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item())
                progress.set_postfix_str(f"loss={total_loss / batch_no:.4f}")
                progress.update(1)

        neural_lp, neural_cp = predict_multitask(model, X_byte[val_idx], X_val_tab,
                                                  batch_size=args.batch_size * 2, device=DEVICE,
                                                  desc=f"Val epoch {epoch}")
        neural_thresh = _best_threshold(y_label[val_idx], neural_lp)
        neural_lf1 = f1_score(y_label[val_idx], (neural_lp >= neural_thresh).astype(int))
        val_pos = y_label[val_idx] == 1
        if val_pos.any():
            neural_cm = f1_score(y_cwe[val_idx][val_pos], neural_cp[val_pos].argmax(axis=1),
                                 average="macro", labels=list(range(num_cwe_classes)), zero_division=0)
        else:
            neural_cm = 0.0

        composite = neural_lf1 + 0.45 * neural_cm
        print(f"  epoch {epoch}: label_f1={neural_lf1:.4f} cwe_macro={neural_cm:.4f} composite={composite:.4f} lr={optimizer.param_groups[0]['lr']:.2e}")

        # SWA: running average of weights over last 5 epochs
        if epoch >= swa_start:
            if swa_state is None:
                swa_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                swa_n = 1
            else:
                for k in swa_state:
                    swa_state[k] = swa_state[k] + model.state_dict()[k].detach()
                swa_n += 1

        if composite > best_score:
            best_score = composite
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_metrics = {"label_f1": float(neural_lf1), "cwe_macro_f1": float(neural_cm),
                            "threshold": float(neural_thresh), "epoch": epoch}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"  note: early stopping at epoch {epoch}")
                break

    assert best_state is not None
    # Apply SWA average if available, then load
    if swa_state is not None and swa_n > 0:
        for k in swa_state:
            swa_state[k] = swa_state[k] / float(swa_n)
        model.load_state_dict(swa_state)
        print(f"  SWA applied: averaged last {swa_n} epochs (started epoch {swa_start})")
    else:
        model.load_state_dict(best_state)
    return model, normalizer, best_metrics, X_tr_tab, X_val_tab


def _train_fusion_mlp_proper(tree_lp, tree_cp, neural_lp, neural_cp,
                              y_label, y_cwe, num_cwe_classes, seed, args):
    """Train MLP fusion with proper train/val split (50/50 on val set predictions)."""
    print(f"\n--- MLP fusion training seed={seed} ---")
    X_input = np.concatenate([tree_lp.reshape(-1, 1), tree_cp,
                               neural_lp.reshape(-1, 1), neural_cp], axis=1).astype(np.float32)
    n = len(X_input)
    # Proper split: 60% train, 20% val, 20% held-out for final threshold
    idx = np.arange(n)
    ftrain_idx, ftemp_idx = train_test_split(idx, test_size=0.4, random_state=seed)
    fval_idx, ftest_idx = train_test_split(ftemp_idx, test_size=0.5, random_state=seed)

    fusion_model = FusionMLP(num_cwe_classes=num_cwe_classes, hidden=192).to(DEVICE)

    pos_count = float(y_label[ftrain_idx].sum())
    neg_count = float(len(ftrain_idx) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
    label_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    cwe_weights = build_cwe_class_weights(y_cwe[ftrain_idx], num_cwe_classes)
    cwe_criterion = FocalLoss(gamma=2.0, alpha=cwe_weights).to(DEVICE)

    optimizer = torch.optim.AdamW(fusion_model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.fusion_epochs)

    best_state, best_composite, best_metrics = None, -1.0, {}
    patience_counter = 0

    for epoch in range(1, args.fusion_epochs + 1):
        fusion_model.train()
        perm = np.random.default_rng(seed * 10 + epoch).permutation(len(ftrain_idx))
        total_loss, total_batches = 0.0, max(1, int(np.ceil(len(ftrain_idx) / args.batch_size)))
        with tqdm(total=total_batches, desc=f"Fusion epoch {epoch}/{args.fusion_epochs} s={seed}") as progress:
            for start in range(0, len(ftrain_idx), args.batch_size):
                bi = ftrain_idx[perm[start:start + args.batch_size]]
                xb = torch.from_numpy(X_input[bi]).to(DEVICE)
                lb = torch.from_numpy(y_label[bi].astype(np.float32)).to(DEVICE)
                cb = torch.from_numpy(y_cwe[bi].astype(np.int64)).to(DEVICE)
                optimizer.zero_grad(set_to_none=True)
                ll, cl = fusion_model(xb)
                l_loss = label_criterion(ll, lb)
                pm = cb >= 0
                c_loss = cwe_criterion(cl[pm], cb[pm]) if pm.any() else torch.zeros((), device=DEVICE)
                loss = l_loss + c_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), 5.0)
                optimizer.step()
                total_loss += float(loss.item())
                progress.set_postfix_str(f"loss={total_loss / (start // args.batch_size + 1):.4f}")
                progress.update(1)
        scheduler.step()

        # Evaluate on fval_idx
        flp, fcp = predict_fusion_mlp(fusion_model, tree_lp[fval_idx], tree_cp[fval_idx],
                                       neural_lp[fval_idx], neural_cp[fval_idx],
                                       batch_size=args.batch_size, device=DEVICE)
        fthresh = _best_threshold(y_label[fval_idx], flp)
        flf1 = f1_score(y_label[fval_idx], (flp >= fthresh).astype(int))
        fpos = y_label[fval_idx] == 1
        if fpos.any():
            fcm = f1_score(y_cwe[fval_idx][fpos], fcp[fpos].argmax(axis=1),
                           average="macro", labels=list(range(num_cwe_classes)), zero_division=0)
        else:
            fcm = 0.0
        composite = flf1 + 0.45 * fcm
        print(f"  fusion epoch {epoch}: label_f1={flf1:.4f} cwe_macro={fcm:.4f} composite={composite:.4f}")
        prev_cwe = best_metrics.get("cwe_macro_f1", 0.0)
        if composite > best_composite:
            best_composite = composite
            best_state = {k: v.detach().cpu() for k, v in fusion_model.state_dict().items()}
            best_metrics = {"label_f1": float(flf1), "cwe_macro_f1": float(fcm),
                            "threshold": float(fthresh), "epoch": epoch}
            patience_counter = 0
        else:
            if fcm <= prev_cwe + 0.001:
                patience_counter += 1
            else:
                patience_counter = 0
            if patience_counter >= 10:
                print(f"  note: fusion early stopping at epoch {epoch}")
                break

    assert best_state is not None
    fusion_model.load_state_dict(best_state)

    # Final threshold from held-out ftest_idx
    flp_test, fcp_test = predict_fusion_mlp(fusion_model, tree_lp[ftest_idx], tree_cp[ftest_idx],
                                              neural_lp[ftest_idx], neural_cp[ftest_idx],
                                              batch_size=args.batch_size, device=DEVICE)
    final_thresh = _best_threshold(y_label[ftest_idx], flp_test)
    ftest_lf1 = f1_score(y_label[ftest_idx], (flp_test >= final_thresh).astype(int))
    ftest_pos = y_label[ftest_idx] == 1
    if ftest_pos.any():
        ftest_cm = f1_score(y_cwe[ftest_idx][ftest_pos], fcp_test[ftest_pos].argmax(axis=1),
                            average="macro", labels=list(range(num_cwe_classes)), zero_division=0)
    else:
        ftest_cm = 0.0
    print(f"  held-out test: label_f1={ftest_lf1:.4f} cwe_macro={ftest_cm:.4f} threshold={final_thresh:.4f}")

    return fusion_model, ftest_lf1, ftest_cm, {**best_metrics, "threshold": float(final_thresh),
                                                  "heldout_label_f1": float(ftest_lf1),
                                                  "heldout_cwe_macro": float(ftest_cm)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = _parse_args()
    ensure_model_dir(MODEL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds
    print(f"v2.4 ensemble training: seeds={seeds}, device={DEVICE}")
    print(f"pseudo-label thresholds: label>{PSEUDO_LABEL_THRESH_HIGH}/<{PSEUDO_LABEL_THRESH_LOW}, cwe>{PSEUDO_CWE_THRESH}")

    # ---- Phase 1: Pseudo-labeling ----
    train_rows = read_csv_rows(TRAIN_CSV)
    print(f"Original training samples: {len(train_rows)}")
    if not args.skip_pseudo:
        pseudo_rows = _pseudo_label_test_set(args)
    else:
        pseudo_path = ROOT / PSEUDO_TRAIN_CSV
        if pseudo_path.exists():
            pseudo_rows = read_csv_rows(pseudo_path)
            print(f"Loaded {len(pseudo_rows)} cached pseudo-labels")
        else:
            print("No cached pseudo-labels found; generating them now.")
            pseudo_rows = _pseudo_label_test_set(args)

    all_rows = train_rows + pseudo_rows
    print(f"Combined training samples: {len(all_rows)} ({len(pseudo_rows)} pseudo + {len(train_rows)} original)")

    # ---- Phase 2: Feature extraction ----
    print("\n=== Phase 2: Feature extraction ===")
    tabular_cache = _load_or_build_tabular_cache(all_rows)
    X = np.asarray(tabular_cache["X"], dtype=np.float32)
    y_label = np.asarray(tabular_cache["y_label"], dtype=np.int32)
    cwe_ids = list(tabular_cache["cwe_ids"])
    feature_columns = list(tabular_cache.get("feature_columns", get_feature_columns()))

    byte_cache = _load_or_build_byte_cache(all_rows, args.byte_length)
    X_byte = np.asarray(byte_cache["X_byte"], dtype=np.uint8)

    # Build CWE mapping from train labels only (ignore pseudo-labels with empty CWE)
    train_pos_mask = y_label[:len(train_rows)] == 1
    train_pos_cwe = [cwe_ids[i] for i in range(len(train_rows)) if train_pos_mask[i]]
    cwe_classes = sorted(set(train_pos_cwe))
    cwe_mapping = {name: idx for idx, name in enumerate(cwe_classes)}
    y_cwe = np.full(len(cwe_ids), -1, dtype=np.int32)
    for i, cid in enumerate(cwe_ids):
        if cid and cid in cwe_mapping:
            y_cwe[i] = cwe_mapping[cid]

    print(f"Features: {X.shape[1]} tabular dims, {len(cwe_classes)} CWE classes, byte_len={args.byte_length}")

    # ---- Phase 3: Multi-seed training ----
    print("\n=== Phase 3: Multi-seed training ===")
    seed_results = {}
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Training seed={seed}")
        print(f"{'='*60}")

        # Split
        train_idx, val_idx = _split_indices(y_label, cwe_ids, rng_seed=seed)

        # Tabular
        label_model, cwe_model, cwe_cls, tree_thresh, tr_idx, val_idx_t, cwe_map = _train_tabular_one_seed(
            X, y_label, cwe_ids, seed, feature_columns)

        # Neural — homogenous emb=48 for all seeds (v2.0)
        neural_model, normalizer, neural_metrics, X_tr_tab, X_val_tab = _train_neural_one_seed(
            X_byte, X, y_label, y_cwe, train_idx, val_idx, len(cwe_classes), seed, args)

        # Save neural bundle
        torch.save({
            "state_dict": {k: v.detach().cpu() for k, v in neural_model.state_dict().items()},
            "model_config": {"tabular_dim": int(X.shape[1]), "num_cwe_classes": int(len(cwe_classes)),
                             "dropout": args.dropout, "byte_embedding_dim": args.byte_embedding_dim},
            "normalizer": {"mean": torch.from_numpy(normalizer.mean), "std": torch.from_numpy(normalizer.std)},
            "feature_columns": feature_columns, "cwe_classes": cwe_classes,
            "byte_length": int(args.byte_length), "metrics": neural_metrics, "model_version": "v2.5", "seed": seed,
        }, MODEL_DIR / seed_neural_bundle(seed))

        # Scalar fusion grid search on val set
        label_model_val = _build_label_ensemble(random_state=seed)
        label_model_val.fit(X[train_idx], y_label[train_idx])
        pos_mask_val = y_label[val_idx] == 1
        cwe_model_val = _build_cwe_gbdt(len(cwe_classes), _cwe_class_weight_map(
            y_cwe[train_idx][y_label[train_idx] == 1], len(cwe_classes)), random_state=seed)
        cwe_model_val.fit(X[train_idx][y_label[train_idx] == 1], y_cwe[train_idx][y_label[train_idx] == 1])

        tree_lp = _aligned_positive_probability(label_model_val, X[val_idx])
        tree_cp_full = _aligned_cwe_probability(cwe_model_val, X[val_idx], len(cwe_classes))
        neural_lp, neural_cp = predict_multitask(
            neural_model, X_byte[val_idx], X_val_tab,
            batch_size=args.batch_size, device=DEVICE, desc="Scalar fusion eval")

        nw_label, s_thresh, s_lf1 = _search_binary_fusion(y_label[val_idx], tree_lp, neural_lp)
        if pos_mask_val.any():
            nw_cwe, s_cm = _search_cwe_fusion(y_cwe[val_idx][pos_mask_val],
                                               tree_cp_full[pos_mask_val], neural_cp[pos_mask_val])
        else:
            nw_cwe, s_cm = 1.0, 0.0
        print(f"Scalar fusion seed={seed}: label_f1={s_lf1:.4f} cwe_macro={s_cm:.4f}")

        # Per-class shrinkage CWE weights (v2.5)
        per_class_cwe_weights = {}
        if pos_mask_val.any():
            train_cwe_counts = np.bincount(y_cwe[train_idx][y_cwe[train_idx] >= 0],
                                           minlength=len(cwe_classes)).astype(np.float64)
            for c in range(len(cwe_classes)):
                c_mask_val = y_cwe[val_idx] == c
                n_c = train_cwe_counts[c]
                alpha = n_c / (n_c + 50.0)
                if c_mask_val.sum() > 0:
                    best_w, best_f1 = nw_cwe, -1.0
                    for w in np.linspace(0.0, 1.0, 41):
                        fused_c = w * neural_cp[c_mask_val, c] + (1.0 - w) * tree_cp_full[c_mask_val, c]
                        pred = (fused_c >= 0.5).astype(int)
                        tp = pred.sum()
                        fp = tp  # simplified: count positives
                        fn = int(c_mask_val.sum()) - tp
                        prec = tp / max(tp + (neural_cp[~c_mask_val, c] * w > 0.5).sum(), 1)
                        rec = tp / max(tp + fn, 1)
                        f1_c = 2 * prec * rec / max(prec + rec, 1e-12)
                        if f1_c > best_f1:
                            best_f1, best_w = f1_c, float(w)
                    # Shrinkage: small classes pull toward global weight
                    w_shrunk = alpha * best_w + (1.0 - alpha) * nw_cwe
                else:
                    w_shrunk = nw_cwe  # no val samples → use global
                per_class_cwe_weights[str(c)] = float(w_shrunk)

        seed_results[seed] = {
            "scalar_label_f1": float(s_lf1), "scalar_cwe_macro": float(s_cm),
            "scalar_threshold": float(s_thresh),
            "scalar_neural_label_weight": float(nw_label),
            "scalar_tree_label_weight": float(1.0 - nw_label),
            "scalar_neural_cwe_weight": float(nw_cwe),
            "scalar_tree_cwe_weight": float(1.0 - nw_cwe),
            "neural_label_f1": float(neural_metrics["label_f1"]),
            "neural_cwe_macro": float(neural_metrics["cwe_macro_f1"]),
            "per_class_cwe_weights": per_class_cwe_weights,
        }
        print(f"Seed {seed} done: scalar_cwe={s_cm:.4f}")

    # ---- Phase 4: Save ensemble config ----
    print("\n=== Phase 4: Saving ensemble config ===")
    avg_scalar = np.mean([r["scalar_cwe_macro"] for r in seed_results.values()])
    print(f"Average scalar_cwe={avg_scalar:.4f}")

    # Weighted ensemble: weight each seed by its scalar_cwe performance
    cwe_scores = np.array([r["scalar_cwe_macro"] for r in seed_results.values()])
    seed_weights_arr = cwe_scores / cwe_scores.sum()
    seed_weights = {str(s): float(seed_weights_arr[i]) for i, s in enumerate(seeds)}
    print(f"Seed weights: {', '.join(f's{s}={w:.3f}' for s, w in seed_weights.items())}")

    # Average scalar weights (weighted by seed performance)
    weighted_nw_label = float(np.average([r["scalar_neural_label_weight"] for r in seed_results.values()],
                                          weights=cwe_scores))
    weighted_nw_cwe = float(np.average([r["scalar_neural_cwe_weight"] for r in seed_results.values()],
                                        weights=cwe_scores))

    # Average per-class CWE weights across seeds (weighted by seed performance, v2.5)
    avg_per_class_cwe_weights = {}
    for c in range(len(cwe_classes)):
        w_c = float(np.average(
            [seed_results[s]["per_class_cwe_weights"][str(c)] for s in seeds],
            weights=cwe_scores))
        avg_per_class_cwe_weights[str(c)] = w_c

    ensemble_config = {
        "model_version": "v2.5",
        "device": str(DEVICE),
        "byte_length": int(args.byte_length),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "seeds": seeds,
        "num_pseudo_labels": len(pseudo_rows),
        "fusion_mode": "scalar",
        "tta_windows": TTA_WINDOWS,
        "seed_weights": seed_weights,
        "scalar_neural_label_weight": float(weighted_nw_label),
        "scalar_tree_label_weight": float(1.0 - weighted_nw_label),
        "scalar_neural_cwe_weight": float(weighted_nw_cwe),
        "scalar_tree_cwe_weight": float(1.0 - weighted_nw_cwe),
        "fusion_threshold": float(np.mean([r["scalar_threshold"] for r in seed_results.values()])),
        "per_class_cwe_weights": avg_per_class_cwe_weights,
        "seed_results": {str(s): r for s, r in seed_results.items()},
        "avg_scalar_cwe_macro": float(avg_scalar),
        "cwe_loss_weight": float(args.cwe_loss_weight),
        "focal_gamma": float(args.focal_gamma),
        "gbdt_backend": _detect_gbdt_backend(),
    }
    write_json(MODEL_DIR / FUSION_CONFIG_NAME, ensemble_config)
    write_json(MODEL_DIR / FEATURE_COLUMNS_NAME, feature_columns)
    write_json(MODEL_DIR / CWE_MAPPING_NAME,
               {"classes": cwe_classes, "class_to_index": {n: i for i, n in enumerate(cwe_classes)}})

    # Save combined tabular bundle (pointing to per-seed files)
    joblib.dump({
        "seeds": seeds, "feature_columns": feature_columns, "cwe_classes": cwe_classes,
        "model_version": "v2.5", "label_model_files": {s: seed_label_model(s) for s in seeds},
        "cwe_model_files": {s: seed_cwe_model(s) for s in seeds},
    }, MODEL_DIR / TABULAR_BUNDLE_NAME)

    print(f"\n{'='*60}")
    print(f"v2.5 training complete!")
    print(f"Fusion mode: per-class shrinkage scalar + SWA + cosine + Capstone extended")
    print(f"Avg scalar CWE macro: {avg_scalar:.4f}")
    print(f"Ensemble config: {MODEL_DIR / FUSION_CONFIG_NAME}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
