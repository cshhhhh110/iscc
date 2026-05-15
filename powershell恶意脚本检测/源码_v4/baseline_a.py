"""
v4.0 Baseline A — Pure LGBM, no lookup, no pseudo-label.
Evaluated on fixed Canary-G and Canary-S splits.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
TEST_PATH = PKG_ROOT / "data_test.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
N_FOLDS = 5
RANDOM_SEED = 2026

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=-1,
)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("name", "label")]


def make_key(row) -> Tuple[int, ...]:
    return tuple(int(v) for v in row)


def compute_entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def main():
    # ---- Load splits ----
    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        splits = json.load(f)
    canary_g_idx = set(splits["canary_g_indices"])
    canary_s_idx = set(splits["canary_s_indices"])
    print(f"Canary-G: {len(canary_g_idx)} samples, Canary-S: {len(canary_s_idx)} samples")

    train = pd.read_csv(TRAIN_PATH)
    features = feature_columns(train)
    y = train["label"].astype(int).to_numpy()

    # ---- Train/Canary split ----
    train_mask = np.ones(len(train), dtype=bool)
    train_mask[list(canary_g_idx)] = False  # exclude Canary-G from training
    train_idx = np.where(train_mask)[0]

    X_all = train[features].to_numpy(dtype=np.float32)
    X_train_full = X_all[train_idx]
    y_train_full = y[train_idx]

    # Key groups for StratifiedGroupKFold (on training portion only)
    key_arr_all = np.array([make_key(row) for row in train[features].itertuples(index=False, name=None)])
    key_arr_train = key_arr_all[train_idx]

    group_ids: dict = {}
    groups_train = []
    for k in key_arr_train:
        kt = tuple(k)
        if kt not in group_ids:
            group_ids[kt] = len(group_ids)
        groups_train.append(group_ids[kt])
    groups_train = np.array(groups_train, dtype=int)

    # ---- Train LGBM via StratifiedGroupKFold ----
    print(f"Training on {len(train_idx)} samples ({len(group_ids)} unique keys)...")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits_list = list(cv.split(X_train_full, y_train_full, groups_train))

    oof = np.zeros((len(X_all), 3), dtype=np.float32)
    models = []
    for fold, (tr_idx, va_idx) in enumerate(splits_list):
        fold_seed = RANDOM_SEED + fold * 100
        m = LGBMClassifier(random_state=fold_seed, **LGBM_PARAMS)
        m.fit(X_train_full[tr_idx], y_train_full[tr_idx])
        # Map back to original indices
        orig_va = train_idx[va_idx]
        oof[orig_va] = m.predict_proba(X_all[orig_va])
        models.append(m)

    # ---- Evaluate: Canary-G ----
    cg_pred = oof[list(canary_g_idx)].argmax(axis=1)
    cg_true = y[list(canary_g_idx)]
    cg_f1 = f1_score(cg_true, cg_pred, average="macro")
    cg_per_class = f1_score(cg_true, cg_pred, average=None)
    print(f"\nCanary-G F1: {cg_f1:.6f}  (per-class: {np.round(cg_per_class, 6)})")

    # ---- Evaluate: Canary-S ----
    cs_pred = oof[list(canary_s_idx)].argmax(axis=1)
    cs_true = y[list(canary_s_idx)]
    cs_f1 = f1_score(cs_true, cs_pred, average="macro")
    cs_per_class = f1_score(cs_true, cs_pred, average=None)
    print(f"Canary-S F1: {cs_f1:.6f}  (per-class: {np.round(cs_per_class, 6)})")

    # ---- Overall Canary (weighted by sample count) ----
    overall_pred = oof[list(canary_g_idx | canary_s_idx)].argmax(axis=1)
    overall_true = y[list(canary_g_idx | canary_s_idx)]
    overall_f1 = f1_score(overall_true, overall_pred, average="macro")
    n_g = len(canary_g_idx)
    n_s = len(canary_s_idx)
    weighted = (cg_f1 * n_g + cs_f1 * n_s) / (n_g + n_s)
    print(f"Overall Canary (combined): {overall_f1:.6f}")
    print(f"Weighted (G×{n_g}+S×{n_s})/({n_g+n_s}): {weighted:.6f}")

    # ---- Sanity: random OOF (all training samples) ----
    train_cv_idx = list(set(train_idx) - canary_g_idx)
    oof_train = oof[train_cv_idx]
    y_train_cv = y[train_cv_idx]
    oof_f1 = f1_score(y_train_cv, oof_train.argmax(axis=1), average="macro")
    print(f"\nRandom OOF (sanity): {oof_f1:.6f}")

    # ---- Test prediction and distribution ----
    test = pd.read_csv(TEST_PATH)
    X_test = test[features].to_numpy(dtype=np.float32)
    test_probs = np.zeros((len(test), 3), dtype=np.float32)
    for m in models:
        test_probs += m.predict_proba(X_test)
    test_probs /= len(models)
    test_pred = test_probs.argmax(axis=1)
    dist = {int(k): int(v) for k, v in pd.Series(test_pred).value_counts().sort_index().items()}
    print(f"Test prediction distribution: {dist}")

    # ---- Key stats for lookup-layer analysis ----
    all_keys = {}
    for i in range(len(train)):
        k = tuple(key_arr_all[i])
        all_keys.setdefault(k, {"count": 0, "classes": []})
        all_keys[k]["count"] += 1
        all_keys[k]["classes"].append(y[i])

    # Count lookup-eligible keys
    global_dist = np.bincount(y, minlength=3) / len(y)
    alpha = 10.0
    lookup_count = 0
    for k, v in all_keys.items():
        count = v["count"]
        if count < 30:
            continue
        cls_counts = np.bincount(v["classes"], minlength=3)
        posterior = (cls_counts + alpha * global_dist) / (count + alpha)
        ent = compute_entropy(posterior)
        if ent < 0.1 and posterior.max() >= 0.95:
            lookup_count += 1
    print(f"\nLookup-eligible keys (count>=30, entropy<0.1, max>=0.95): {lookup_count}")
    print(f"Baseline A complete.")


if __name__ == "__main__":
    main()
