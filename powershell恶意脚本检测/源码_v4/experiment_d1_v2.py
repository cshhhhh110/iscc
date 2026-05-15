"""
v4.0 D1 v2 — Tight Backoff (narrow gate, not default fallback)

Rules:
  count>=30 & entropy<0.12 & max>=0.95 : exact posterior (no backoff)
  10<=count<30 | 0.12<=entropy<0.30   : 0.8*exact + 0.2*neighbor
  count<10 | entropy>=0.30             : neighbor backoff (+ global if inconsistent)
  unseen key                            : neighbor, global fallback if inconsistent
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
K_NEIGHBORS = 7  # increased from 5 for stability


def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-np.sum(p * np.log(p)))


def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    if probs.ndim == 1:
        return probs / probs.sum()
    return probs / probs.sum(axis=1, keepdims=True)


def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = [c for c in train.columns if c not in ("name", "label")]
    y = train["label"].astype(int).to_numpy()

    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        meta = json.load(f)
    canary_g_idx = set(meta["canary_g_indices"])
    canary_s_idx = set(meta["canary_s_indices"])
    weak_seen_idx = set(meta.get("weak_seen_indices", []))

    # ---- Key stats (excl Canary-G) ----
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

    # ---- MI weights ----
    print("Computing MI weights...")
    mi_weights = np.zeros(len(features), dtype=np.float64)
    for fi, col in enumerate(features):
        mi_weights[fi] = mutual_info_score(train[col], y)
    mi_weights = np.clip(mi_weights, 0.01, mi_weights.max())
    mi_weights = mi_weights / mi_weights.sum() * len(features)
    print(f"MI ranges: [{mi_weights.min():.2f}, {mi_weights.max():.2f}]")

    # ---- Nearest neighbor index ----
    train_keys_arr = np.array([list(k) for k in all_train_keys], dtype=np.int16)

    def weighted_hamming(k1_tuple):
        k1 = np.array(list(k1_tuple), dtype=np.int16)
        diffs = (train_keys_arr != k1).astype(np.float64)
        return diffs @ mi_weights

    def find_neighbors(key_tuple, k=K_NEIGHBORS):
        dists = weighted_hamming(key_tuple)
        nn_idx = np.argsort(dists)[:k]
        return [all_train_keys[i] for i in nn_idx], dists[nn_idx]

    def neighbor_posterior(neighbors, dists):
        w = 1.0 / (np.array(dists) + 0.5)
        w = w / w.sum()
        post = np.zeros(3, dtype=np.float64)
        for nk, wi in zip(neighbors, w):
            post += wi * key_posterior.get(nk, global_dist)
        return normalize(post.reshape(1, -1))[0]

    def neighbor_consistency(neighbors):
        modes = [key_mode.get(nk, -1) for nk in neighbors]
        valid_modes = [m for m in modes if m >= 0]
        if not valid_modes: return 0.0, -1
        best = max(set(valid_modes), key=valid_modes.count)
        return valid_modes.count(best) / len(valid_modes), best

    # ---- Tight Backoff ----
    print("Running tight backoff...")
    test_key_to_idx = defaultdict(list)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        test_key_to_idx[tuple(int(v) for v in row)].append(i)

    backoff_post = {}
    tiers = {"exact": 0, "interp": 0, "neighbor": 0, "global": 0}
    tier_samps = {"exact": 0, "interp": 0, "neighbor": 0, "global": 0}

    for k in test_key_to_idx:
        cnt = key_count.get(k, 0)
        ent = key_entropy.get(k, 1.0)
        exact_post = key_posterior.get(k, None)
        n_samp = len(test_key_to_idx[k])

        if cnt >= 30 and ent < 0.12 and exact_post is not None and exact_post.max() >= 0.95:
            # Tier 1: Exact — highly reliable seen key
            backoff_post[k] = exact_post
            tiers["exact"] += 1
            tier_samps["exact"] += n_samp

        elif cnt >= 10 and ent < 0.30 and exact_post is not None:
            # Tier 2: Interpolation — mildly uncertain
            neighbors, dists = find_neighbors(k, K_NEIGHBORS)
            nb_post = neighbor_posterior(neighbors, dists)
            backoff_post[k] = normalize(0.8 * exact_post + 0.2 * nb_post)
            tiers["interp"] += 1
            tier_samps["interp"] += n_samp

        elif cnt > 0 and exact_post is not None:
            # Tier 3: Neighbor backoff — weak seen key
            neighbors, dists = find_neighbors(k, K_NEIGHBORS)
            cons, cons_mode = neighbor_consistency(neighbors)
            if cons >= 0.6:
                nb_post = neighbor_posterior(neighbors, dists)
                backoff_post[k] = normalize(0.3 * exact_post + 0.7 * nb_post)
            else:
                backoff_post[k] = normalize(0.5 * exact_post + 0.5 * global_dist)
            tiers["neighbor"] += 1
            tier_samps["neighbor"] += n_samp

        else:
            # Tier 4: Unseen key — neighbor with global fallback
            neighbors, dists = find_neighbors(k, K_NEIGHBORS)
            cons, cons_mode = neighbor_consistency(neighbors)
            if cons >= 0.6:
                nb_post = neighbor_posterior(neighbors, dists)
                backoff_post[k] = normalize(0.7 * nb_post + 0.3 * global_dist)
                tiers["global"] += 1
            else:
                backoff_post[k] = global_dist  # pure global
                tiers["global"] += 1
            tier_samps["global"] += n_samp

    print(f"Tiers: exact={tiers['exact']}, interp={tiers['interp']}, neighbor={tiers['neighbor']}, global={tiers['global']}")
    print(f"Samples: exact={tier_samps['exact']}, interp={tier_samps['interp']}, neighbor={tier_samps['neighbor']}, global={tier_samps['global']}")

    # ---- Evaluate ----
    def eval_set(name, indices):
        y_true = y[list(indices)]
        y_pred = []
        for i in indices:
            k = tuple(int(v) for v in train[features].iloc[i])
            if k in backoff_post:
                y_pred.append(backoff_post[k].argmax())
            else:
                neighbors, dists = find_neighbors(k, K_NEIGHBORS)
                cons, _ = neighbor_consistency(neighbors)
                if cons >= 0.6:
                    y_pred.append(neighbor_posterior(neighbors, dists).argmax())
                else:
                    y_pred.append(int(global_dist.argmax()))
        y_pred = np.array(y_pred)
        f1 = f1_score(y_true, y_pred, average="macro")
        f1_per = f1_score(y_true, y_pred, average=None)
        return f1, f1_per

    cg_f1, cg_per = eval_set("Canary-G", canary_g_idx)
    cs_f1, cs_per = eval_set("Canary-S", canary_s_idx)
    ws_f1, ws_per = eval_set("Weak-Seen", weak_seen_idx)

    print(f"\n=== D1 v2 Tight Backoff ===")
    print(f"Canary-G:  {cg_f1:.6f}  {np.round(cg_per, 6)}")
    print(f"Canary-S:  {cs_f1:.6f}  {np.round(cs_per, 6)}")
    if weak_seen_idx:
        print(f"Weak-Seen: {ws_f1:.6f}  {np.round(ws_per, 6)}")

    # vs C1 (best from before)
    c1_cg, c1_cs = 0.236816, 0.747390
    print(f"\n=== vs C1 ===")
    print(f"Canary-G:  {cg_f1:.6f} vs {c1_cg:.6f}  ({'+' if cg_f1>c1_cg else ''}{cg_f1-c1_cg:+.6f})")
    print(f"Canary-S:  {cs_f1:.6f} vs {c1_cs:.6f}  ({'+' if cs_f1>c1_cs else ''}{cs_f1-c1_cs:+.6f})")
    if weak_seen_idx:
        # vs D1 v1
        print(f"Weak-Seen: {ws_f1:.6f} (D1 v1 was 0.480)")

    # Test submission
    test_preds = np.zeros(len(test), dtype=int)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        test_preds[i] = backoff_post[k].argmax()

    dist = {int(k): int(v) for k, v in pd.Series(test_preds).value_counts().sort_index().items()}
    c1_dist = {0: 13338, 1: 3347, 2: 3315}
    print(f"\nTest distribution: {dist}")
    for cls in [0, 1, 2]:
        c1v = c1_dist.get(cls, 0)
        d1v = dist.get(cls, 0)
        shift = 100 * (d1v - c1v) / max(c1v, 1)
        flag = " ⚠️" if abs(shift) > 10 else ""
        print(f"  Class {cls}: {d1v} vs C1={c1v} ({'+' if shift>0 else ''}{shift:.1f}%){flag}")

    sub_path = PKG_ROOT / "提交结果" / "submission_v4_d1_v2.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_preds}).to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")


if __name__ == "__main__":
    main()
