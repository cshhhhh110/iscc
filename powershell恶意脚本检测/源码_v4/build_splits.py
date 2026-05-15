"""
v4.0 — Build fixed Canary-G and Canary-S splits.
Run once to generate the split files, then reuse for all A/B/C experiments.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
TEST_PATH = PKG_ROOT / "data_test.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
RANDOM_SEED = 2026

FEATURES = None  # filled after loading


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("name", "label")]


def make_key(row) -> Tuple[int, ...]:
    return tuple(int(v) for v in row)


def entropy(probs: np.ndarray) -> float:
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def main():
    global FEATURES
    rng = np.random.default_rng(RANDOM_SEED)

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    FEATURES = feature_columns(train)

    # ---- Build key-level stats ----
    print("Building key stats...")
    key_arr = [make_key(row) for row in train[FEATURES].itertuples(index=False, name=None)]
    labels = train["label"].astype(int).to_numpy()

    key_to_rows: Dict[Tuple, List[int]] = {}
    key_to_labels: Dict[Tuple, List[int]] = {}
    for i, k in enumerate(key_arr):
        key_to_rows.setdefault(k, []).append(i)
        key_to_labels.setdefault(k, []).append(labels[i])

    key_stats = {}
    for k, lbls in key_to_labels.items():
        lbl_arr = np.array(lbls)
        count = len(lbl_arr)
        class_counts = np.bincount(lbl_arr, minlength=3)
        n_unique = int((np.bincount(lbl_arr, minlength=3) > 0).sum())
        key_stats[k] = {
            "count": count,
            "label_nunique": n_unique,
            "class_dist": class_counts.tolist(),
        }

    all_keys = list(key_stats.keys())
    print(f"Total unique keys: {len(all_keys)}")

    # ---- Canary-G: stratified sample 200 keys (count bucket × label_nunique) ----
    print("Building Canary-G (group-unseen, 200 keys)...")

    def count_bucket(c: int) -> str:
        if c <= 2: return "1-2"
        if c <= 9: return "3-9"
        if c <= 29: return "10-29"
        return "30+"

    strata: Dict[str, List[Tuple]] = {}
    for k, s in key_stats.items():
        bucket = f"{count_bucket(s['count'])}_{'det' if s['label_nunique']==1 else 'amb'}"
        strata.setdefault(bucket, []).append(k)

    print(f"Strata: { {k: len(v) for k, v in strata.items()} }")

    # Proportional allocation of 200 keys across strata
    total_keys = sum(len(v) for v in strata.values())
    canary_g_keys: set = set()
    for bucket_name, bucket_keys in strata.items():
        n_alloc = max(1, int(200 * len(bucket_keys) / total_keys))
        chosen = rng.choice(sorted(bucket_keys), size=min(n_alloc, len(bucket_keys)), replace=False)
        canary_g_keys.update(tuple(k) for k in chosen)

    # Trim to exactly 200
    if len(canary_g_keys) > 200:
        canary_g_keys = set(rng.choice(sorted(canary_g_keys), size=200, replace=False))
    print(f"Canary-G keys: {len(canary_g_keys)}")

    # Canary-G indices (all samples from these keys)
    canary_g_indices = []
    for k in canary_g_keys:
        canary_g_indices.extend(key_to_rows[k])
    canary_g_indices = sorted(set(canary_g_indices))
    print(f"Canary-G samples: {len(canary_g_indices)} ({100*len(canary_g_indices)/len(train):.1f}%)")

    # ---- Canary-S: holdout 20% from lookup candidate keys (count >= 30) ----
    print("Building Canary-S (seen holdout, count>=30 keys)...")

    lookup_candidate_keys = [k for k, s in key_stats.items() if s["count"] >= 30]
    print(f"Lookup candidate keys (count>=30): {len(lookup_candidate_keys)}")

    canary_s_indices = []
    for k in lookup_candidate_keys:
        indices = sorted(key_to_rows[k])
        n_holdout = max(3, int(len(indices) * 0.2))
        n_train = len(indices) - n_holdout
        if n_train < 10:
            continue  # skip if training side would be too small
        holdout = rng.choice(indices, size=n_holdout, replace=False)
        canary_s_indices.extend(int(i) for i in holdout)

    canary_s_indices = sorted(set(canary_s_indices))
    print(f"Canary-S samples: {len(canary_s_indices)} ({100*len(canary_s_indices)/len(train):.1f}%)")

    # ---- Verify no overlap between canary sets ----
    overlap_g_s = set(canary_g_indices) & set(canary_s_indices)
    if overlap_g_s:
        print(f"WARNING: {len(overlap_g_s)} samples in both canary sets, removing from Canary-S")
        canary_s_indices = sorted(set(canary_s_indices) - set(canary_g_indices))
        print(f"Canary-S after dedup: {len(canary_s_indices)}")

    # ---- Save splits ----
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    meta = {
        "canary_g_keys": [list(k) for k in sorted(canary_g_keys)],
        "canary_g_indices": [int(x) for x in canary_g_indices],
        "canary_s_indices": [int(x) for x in canary_s_indices],
        "lookup_candidate_keys": [list(k) for k in sorted(lookup_candidate_keys)],
        "n_total": int(len(train)),
        "n_canary_g": int(len(canary_g_indices)),
        "n_canary_s": int(len(canary_s_indices)),
        "random_seed": RANDOM_SEED,
        "strata_distribution": {k: int(len(v)) for k, v in strata.items()},
    }

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(SPLIT_DIR / "canary_splits.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, cls=NpEncoder, ensure_ascii=False, indent=2)

    # Also save train_canary.csv (excluding Canary-G) for training
    train_out = train.drop(index=canary_g_indices).reset_index(drop=True)
    train_out.to_csv(SPLIT_DIR / "train_no_canary_g.csv", index=False)

    print(f"\nSaved to {SPLIT_DIR}/")
    print("Done. Canary splits are fixed and ready for evaluation.")
    print(f"\nNext: run baseline A using these splits.")


if __name__ == "__main__":
    main()
