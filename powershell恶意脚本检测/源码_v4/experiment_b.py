"""
v4.0 Experiment B — Lookup Layer + Residual LGBMClassifier + anchor retention
No KD, no pseudo-label. Pure 15-dim features.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

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

# Lookup params (from sweep)
LOOKUP_ALPHA = 1.0
LOOKUP_ENTROPY_THRESHOLD = 0.18
LOOKUP_MAX_PROB_THRESHOLD = 0.95
LOOKUP_MIN_COUNT = 30
ANCHOR_MIN = 5
ANCHOR_FRAC = 0.10

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=-1,
)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("name", "label")]


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def main():
    # ---- Load splits ----
    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        split_meta = json.load(f)
    canary_g_idx = set(split_meta["canary_g_indices"])
    canary_s_idx = set(split_meta["canary_s_indices"])
    print(f"Canary-G: {len(canary_g_idx)}, Canary-S: {len(canary_s_idx)}")

    # ---- Load data ----
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = feature_columns(train)
    y = train["label"].astype(int).to_numpy()

    # ---- Build key stats (on full train for lookup table) ----
    print("Building key stats and lookup table...")
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    key_to_labels: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        key_to_indices[k].append(int(i))
        key_to_labels[k].append(int(y[i]))

    global_dist = np.bincount(y, minlength=3).astype(np.float64) / len(y)

    # Determine lookup keys via Dirichlet shrinkage
    lookup_keys: set = set()
    lookup_posterior: Dict[Tuple, np.ndarray] = {}
    for k, lbls in key_to_labels.items():
        count = len(lbls)
        if count < LOOKUP_MIN_COUNT:
            continue
        cls_counts = np.bincount(lbls, minlength=3).astype(np.float64)
        posterior = (cls_counts + LOOKUP_ALPHA * global_dist) / (count + LOOKUP_ALPHA)
        if entropy(posterior) < LOOKUP_ENTROPY_THRESHOLD and posterior.max() >= LOOKUP_MAX_PROB_THRESHOLD:
            lookup_keys.add(k)
            lookup_posterior[k] = posterior / posterior.sum()  # re-normalize

    lookup_indices = set()
    for k in lookup_keys:
        lookup_indices.update(key_to_indices[k])
    print(f"Lookup keys: {len(lookup_keys)} covering {len(lookup_indices)} samples ({100*len(lookup_indices)/len(train):.1f}%)")

    # ---- Build training set for residual ----
    # Exclude Canary-G entirely
    # For lookup keys: keep only anchor samples in residual, rest handled by lookup
    residual_train_idx: List[int] = []
    anchor_count = 0
    for i in range(len(train)):
        if i in canary_g_idx:
            continue  # Canary-G never in training
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys:
            # Anchor: keep only a fraction of lookup key samples
            indices = key_to_indices[k]
            n_anchor = max(ANCHOR_MIN, int(len(indices) * ANCHOR_FRAC))
            # Use first n_anchor indices as anchor (deterministic per key)
            anchor_set = set(sorted(indices)[:n_anchor])
            if i in anchor_set:
                residual_train_idx.append(i)
                anchor_count += 1
            # else: handled by lookup layer, skip
        else:
            residual_train_idx.append(i)

    residual_train_idx = sorted(residual_train_idx)
    n_lookup_direct = len(lookup_indices - canary_g_idx) - anchor_count
    print(f"Residual training: {len(residual_train_idx)} samples ({anchor_count} anchors, {n_lookup_direct} handled by lookup)")
    print(f"Total training: {len(residual_train_idx) + n_lookup_direct} (excl. Canary-G)")

    # ---- Train residual LGBM via StratifiedGroupKFold ----
    X_all = train[features].to_numpy(dtype=np.float32)
    X_residual = X_all[residual_train_idx]
    y_residual = y[residual_train_idx]

    key_arr = np.array([tuple(int(v) for v in row) for row in train[features].itertuples(index=False, name=None)])
    key_train = key_arr[residual_train_idx]
    gid = {}
    groups = []
    for k in key_train:
        kt = tuple(k)
        if kt not in gid:
            gid[kt] = len(gid)
        groups.append(gid[kt])

    print(f"Training residual on {len(residual_train_idx)} samples ({len(gid)} unique keys)...")
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits_list = list(cv.split(X_residual, y_residual, np.array(groups, dtype=int)))

    # OOF for ALL training samples (lookup fills its own)
    oof = np.zeros((len(train), 3), dtype=np.float32)
    # Fill lookup predictions
    for i in range(len(train)):
        if i in canary_g_idx:
            continue
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys and i not in set(residual_train_idx):
            oof[i] = lookup_posterior[k].astype(np.float32)

    models = []
    for fold, (tr_idx, va_idx) in enumerate(splits_list):
        fold_seed = RANDOM_SEED + fold * 100
        m = LGBMClassifier(random_state=fold_seed, **LGBM_PARAMS)
        m.fit(X_residual[tr_idx], y_residual[tr_idx])
        orig_va = np.array(residual_train_idx)[va_idx]
        oof[orig_va] = m.predict_proba(X_all[orig_va])
        models.append(m)

    # ---- Evaluate Canary-G ----
    cg_pred = oof[list(canary_g_idx)].argmax(axis=1)
    cg_true = y[list(canary_g_idx)]
    cg_f1 = f1_score(cg_true, cg_pred, average="macro")
    cg_per_class = f1_score(cg_true, cg_pred, average=None)
    print(f"\n=== Results ===")
    print(f"Canary-G F1:  {cg_f1:.6f}  (per-class: {np.round(cg_per_class, 6)})")

    # Canary-S
    cs_pred = oof[list(canary_s_idx)].argmax(axis=1)
    cs_true = y[list(canary_s_idx)]
    cs_f1 = f1_score(cs_true, cs_pred, average="macro")
    cs_per_class = f1_score(cs_true, cs_pred, average=None)
    print(f"Canary-S F1:  {cs_f1:.6f}  (per-class: {np.round(cs_per_class, 6)})")

    # Weighted main signal
    n_g, n_s = len(canary_g_idx), len(canary_s_idx)
    weighted = (cg_f1 * n_g + cs_f1 * n_s) / (n_g + n_s)
    print(f"Weighted:     {weighted:.6f}")

    # ---- Compare with A baseline ----
    a_cg = 0.236816
    a_cs = 0.711667
    a_weighted = 0.497680
    print(f"\n=== vs Baseline A ===")
    print(f"Canary-G:  {cg_f1:.6f} vs {a_cg:.6f}  ({'+' if cg_f1>a_cg else ''}{cg_f1-a_cg:+.6f})")
    print(f"Canary-S:  {cs_f1:.6f} vs {a_cs:.6f}  ({'+' if cs_f1>a_cs else ''}{cs_f1-a_cs:+.6f})")
    print(f"Weighted:  {weighted:.6f} vs {a_weighted:.6f}  ({'+' if weighted>a_weighted else ''}{weighted-a_weighted:+.6f})")

    # Judgment
    if cs_f1 > a_cs + 0.001 and cg_f1 > a_cg - 0.01:
        print("\n>>> B EFFECTIVE: Canary-S improved, Canary-G stable. Continue to C. <<<")
    elif cg_f1 < a_cg - 0.02:
        print("\n>>> B FAILED: Canary-G dropped significantly. Lookup layer too aggressive. <<<")
    else:
        print("\n>>> B MARGINAL: check thresholds. <<<")

    # Test prediction
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    X_test = test[features].to_numpy(dtype=np.float32)
    # Batch predict all test samples with residual
    residual_pred = np.zeros((len(test), 3), dtype=np.float32)
    for m in models:
        residual_pred += m.predict_proba(X_test)
    residual_pred /= len(models)

    n_lookup_test = 0
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        if k in lookup_keys:
            test_preds[i] = lookup_posterior[k].astype(np.float32)
            n_lookup_test += 1
        else:
            test_preds[i] = residual_pred[i]

    test_pred_labels = test_preds.argmax(axis=1)
    dist = {int(k): int(v) for k, v in pd.Series(test_pred_labels).value_counts().sort_index().items()}
    print(f"\nLookup hits in test: {n_lookup_test}/{len(test)} ({100*n_lookup_test/len(test):.1f}%)")
    print(f"Test prediction distribution: {dist}")

    # Save submission
    sub_path = PKG_ROOT / "提交结果" / "submission_v4_experiment_b.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_pred_labels}).to_csv(sub_path, index=False)
    print(f"Saved: {sub_path}")


if __name__ == "__main__":
    main()
