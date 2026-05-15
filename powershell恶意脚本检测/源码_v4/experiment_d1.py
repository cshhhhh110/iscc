"""
v4.0 D1 — Weighted Hamming Backoff (inference only, no retraining)

For each key needing backoff (low count / high entropy):
  exact posterior -> weighted Hamming top-K nearest neighbor posterior -> global prior
Uses MI-weighted Hamming distance + neighbor consistency check.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, mutual_info_score

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
TEST_PATH = PKG_ROOT / "data_test.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
K_NEIGHBORS = 5
BACKOFF_ENTROPY_THRESHOLD = 0.3
BACKOFF_COUNT_THRESHOLD = 20
BACKOFF_NEIGHBOR_CONSISTENCY = 0.8  # fraction of top-K that agree on class


def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-np.sum(p * np.log(p)))


def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = [c for c in train.columns if c not in ("name", "label")]
    y = train["label"].astype(int).to_numpy()

    # ---- Load splits ----
    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        meta = json.load(f)
    canary_g_idx = set(meta["canary_g_indices"])
    canary_s_idx = set(meta["canary_s_indices"])
    weak_seen_idx = set(meta.get("weak_seen_indices", []))

    # ---- Build key-level stats (exclude Canary-G) ----
    key_to_idx: Dict[Tuple, List[int]] = defaultdict(list)
    key_to_labels: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        if i in canary_g_idx: continue
        k = tuple(int(v) for v in row)
        key_to_idx[k].append(i)
        key_to_labels[k].append(int(y[i]))

    all_train_keys = list(key_to_idx.keys())
    global_dist = np.bincount(y, minlength=3).astype(np.float64)
    global_dist = global_dist / global_dist.sum()

    # Key posteriors
    key_posterior: Dict[Tuple, np.ndarray] = {}
    key_count: Dict[Tuple, int] = {}
    key_entropy: Dict[Tuple, float] = {}
    key_mode: Dict[Tuple, int] = {}
    for k, lbls in key_to_labels.items():
        cnt = len(lbls)
        key_count[k] = cnt
        cc = np.bincount(lbls, minlength=3).astype(np.float64)
        post = (cc + 1.0 * global_dist) / (cnt + 1.0)
        post = post / post.sum()
        key_posterior[k] = post
        key_entropy[k] = entropy(post)
        key_mode[k] = int(post.argmax())

    # ---- Compute MI weights for Hamming distance ----
    print("Computing MI feature weights...")
    mi_weights = np.zeros(len(features), dtype=np.float64)
    for fi, col in enumerate(features):
        mi_weights[fi] = mutual_info_score(train[col], y)
    # Clip extreme weights and normalize
    mi_weights = np.clip(mi_weights, 0.01, mi_weights.max())
    mi_weights = mi_weights / mi_weights.sum() * len(features)  # mean=1
    print(f"MI weights: {dict(zip(features[:5], np.round(mi_weights[:5], 3)))} ...")

    # ---- Compute weighted Hamming nearest neighbors ----
    print("Building nearest-neighbor index...")
    # Convert all train keys to array for fast Hamming
    train_keys_arr = np.array([list(k) for k in all_train_keys], dtype=np.int16)
    n_train_keys = len(all_train_keys)

    def weighted_hamming(k1_tuple, k2_arr):
        """Weighted Hamming distance from key tuple to array of keys"""
        k1 = np.array(list(k1_tuple), dtype=np.int16)
        diffs = (k2_arr != k1).astype(np.float64)
        return diffs @ mi_weights  # weighted sum

    def find_neighbors(key_tuple, k=K_NEIGHBORS):
        dists = weighted_hamming(key_tuple, train_keys_arr)
        nn_idx = np.argsort(dists)[:k]
        neighbors = [all_train_keys[i] for i in nn_idx]
        return neighbors, dists[nn_idx]

    def neighbor_posterior(neighbors, dists):
        """Weighted average of neighbor posteriors, closer = higher weight"""
        weights = 1.0 / (np.array(dists) + 1.0)
        weights = weights / weights.sum()
        post = np.zeros(3, dtype=np.float64)
        for nk, w in zip(neighbors, weights):
            post += w * key_posterior.get(nk, global_dist)
        return normalize(post.reshape(1, -1))[0]

    def neighbor_consistency(neighbors):
        """Fraction of top-K neighbors that agree on mode"""
        modes = [key_mode.get(nk, -1) for nk in neighbors]
        if -1 in modes: return 0.0
        most_common = max(set(modes), key=modes.count)
        return modes.count(most_common) / len(modes)

    # ---- Backoff inference ----
    print("Running backoff inference...")

    # Load C1 student test predictions (we need a baseline prediction for residual)
    # For now, we'll build the backoff posterior for ALL test keys and compare
    test_key_to_idx: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        test_key_to_idx[tuple(int(v) for v in row)].append(i)

    test_keys_all = list(test_key_to_idx.keys())

    # Backoff posterior for each test key
    backoff_post = {}
    backoff_source = {}  # 'exact' | 'neighbor' | 'global'

    for k in test_keys_all:
        if k in key_posterior and key_count[k] >= BACKOFF_COUNT_THRESHOLD and key_entropy[k] < BACKOFF_ENTROPY_THRESHOLD:
            # Reliable: use exact posterior
            backoff_post[k] = key_posterior[k]
            backoff_source[k] = "exact"
        else:
            # Backoff: try nearest neighbor
            neighbors, dists = find_neighbors(k, K_NEIGHBORS)
            consistency = neighbor_consistency(neighbors)
            if consistency >= BACKOFF_NEIGHBOR_CONSISTENCY:
                backoff_post[k] = neighbor_posterior(neighbors, dists)
                backoff_source[k] = f"neighbor(cons={consistency:.2f})"
            else:
                backoff_post[k] = global_dist
                backoff_source[k] = "global"

    # Stats
    n_exact = sum(1 for v in backoff_source.values() if v == "exact")
    n_neighbor = sum(1 for v in backoff_source.values() if "neighbor" in str(v))
    n_global = sum(1 for v in backoff_source.values() if v == "global")
    print(f"Backoff breakdown: exact={n_exact}, neighbor={n_neighbor}, global={n_global}")

    # Sample coverage
    n_exact_samp = sum(len(test_key_to_idx[k]) for k, v in backoff_source.items() if v == "exact")
    n_neighbor_samp = sum(len(test_key_to_idx[k]) for k, v in backoff_source.items() if "neighbor" in str(v))
    n_global_samp = sum(len(test_key_to_idx[k]) for k, v in backoff_source.items() if v == "global")
    print(f"Sample coverage: exact={n_exact_samp}, neighbor={n_neighbor_samp}, global={n_global_samp}")

    # ---- Evaluate on canary sets (using train OOF predictions as proxy) ----
    # Build backoff pred for all train samples
    backoff_pred_train = np.zeros(len(train), dtype=int)
    backoff_proba_train = np.zeros((len(train), 3), dtype=np.float32)

    for i in range(len(train)):
        if i in canary_g_idx:
            # For Canary-G, simulate with backoff (these keys are unseen in training)
            k = tuple(int(v) for v in train[features].iloc[i])
            if k in backoff_post:
                backoff_proba_train[i] = backoff_post[k]
            else:
                neighbors, dists = find_neighbors(k, K_NEIGHBORS)
                backoff_proba_train[i] = neighbor_posterior(neighbors, dists)
            backoff_pred_train[i] = backoff_proba_train[i].argmax()

    # Evaluate on all canaries using backoff
    def eval_set(name, indices):
        y_true = y[list(indices)]
        y_pred = []
        for i in indices:
            k = tuple(int(v) for v in train[features].iloc[i])
            if k in backoff_post:
                y_pred.append(backoff_post[k].argmax())
            else:
                neighbors, dists = find_neighbors(k, K_NEIGHBORS)
                post = neighbor_posterior(neighbors, dists)
                y_pred.append(post.argmax())
        y_pred = np.array(y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        f1_per = f1_score(y_true, y_pred, average=None)
        return f1, f1_per

    cg_f1, cg_per = eval_set("Canary-G", canary_g_idx)
    cs_f1, cs_per = eval_set("Canary-S", canary_s_idx)
    ws_f1, ws_per = eval_set("Weak-Seen", weak_seen_idx)

    print(f"\n=== D1 Backoff Results ===")
    print(f"Canary-G:     {cg_f1:.6f}  {np.round(cg_per, 6)}")
    print(f"Canary-S:     {cs_f1:.6f}  {np.round(cs_per, 6)}")
    if weak_seen_idx:
        print(f"Weak-Seen:    {ws_f1:.6f}  {np.round(ws_per, 6)}")
        # Baseline: what does pure global prior give on weak-seen?
        gp_pred = np.array([global_dist.argmax() for _ in weak_seen_idx])
        gp_f1 = f1_score(y[list(weak_seen_idx)], gp_pred, average="macro")
        print(f"Weak-Seen (global prior baseline): {gp_f1:.6f}")
        print(f"Weak-Seen vs global prior: {'+' if ws_f1>gp_f1 else ''}{ws_f1-gp_f1:+.6f}")

    # ---- Test submission ----
    test_preds = np.zeros(len(test), dtype=int)
    for i in range(len(test)):
        k = tuple(int(v) for v in test[features].iloc[i])
        test_preds[i] = backoff_post[k].argmax()

    dist = {int(k): int(v) for k, v in pd.Series(test_preds).value_counts().sort_index().items()}
    print(f"\nTest distribution: {dist}")

    sub_path = PKG_ROOT / "提交结果" / "submission_v4_d1.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_preds}).to_csv(sub_path, index=False)
    print(f"Saved: {sub_path}")


if __name__ == "__main__":
    main()
