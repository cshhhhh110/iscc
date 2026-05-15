"""
v4.1 Router — Learns "when to trust exact vs neighbor vs prior"
Small LGBMClassifier per key. Trained on canary holdout labels only.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, mutual_info_score

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
TEST_PATH = PKG_ROOT / "data_test.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
K_NEIGHBORS = 7


def entropy(p):
    p = np.clip(p, 1e-12, 1); return float(-np.sum(p * np.log(p)))


def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    if probs.ndim == 1: return probs / probs.sum()
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

    # All canary indices for router training (never in training set)
    router_canary_idx = canary_g_idx | canary_s_idx | weak_seen_idx
    print(f"Router training pool: {len(router_canary_idx)} canary samples")

    # ---- Key stats (excl Canary-G from stats) ----
    key_to_labels: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        if i not in canary_g_idx:
            k = tuple(int(v) for v in row)
            key_to_labels[k].append(int(y[i]))

    all_seen_keys = set(key_to_labels.keys())
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

    # ---- MI weights + neighbor index ----
    mi_weights = np.zeros(len(features), dtype=np.float64)
    for fi, col in enumerate(features):
        mi_weights[fi] = mutual_info_score(train[col], y)
    mi_weights = np.clip(mi_weights, 0.01, mi_weights.max())
    mi_weights = mi_weights / mi_weights.sum() * len(features)

    train_keys_list = list(all_seen_keys)
    train_keys_arr = np.array([list(k) for k in train_keys_list], dtype=np.int16)

    def find_neighbors(key_tuple, k=K_NEIGHBORS):
        k1 = np.array(list(key_tuple), dtype=np.int16)
        diffs = (train_keys_arr != k1).astype(np.float64)
        dists = diffs @ mi_weights
        nn_idx = np.argsort(dists)[:k]
        return [train_keys_list[i] for i in nn_idx], dists[nn_idx]

    def neighbor_posterior(neighbors, dists):
        w = 1.0 / (np.array(dists) + 0.5)
        w = w / w.sum()
        post = np.zeros(3, dtype=np.float64)
        for nk, wi in zip(neighbors, w):
            post += wi * key_posterior.get(nk, global_dist)
        return normalize(post)

    def neighbor_consistency(neighbors):
        modes = [key_mode.get(nk, -1) for nk in neighbors]
        valid = [m for m in modes if m >= 0]
        if not valid: return 0.0, -1
        best = max(set(valid), key=valid.count)
        return valid.count(best) / len(valid), best

    # ---- Build router training set: one row per canary key ----
    print("Building router training data...")
    canary_keys = set()
    for i in router_canary_idx:
        canary_keys.add(tuple(int(v) for v in train[features].iloc[i]))

    router_rows = []
    for k in canary_keys:
        cnt = key_count.get(k, 0)
        ent = key_entropy.get(k, 1.0)
        exact_post = key_posterior.get(k, None)
        if exact_post is None:
            exact_post = global_dist

        neighbors, dists = find_neighbors(k, K_NEIGHBORS)
        cons, cons_mode = neighbor_consistency(neighbors)
        nb_post = neighbor_posterior(neighbors, dists)

        # Blended backoff (same as D1 v2 logic)
        if cnt >= 30 and ent < 0.12 and exact_post.max() >= 0.95:
            nb_blend = exact_post
        elif cnt >= 10 and ent < 0.30:
            nb_blend = normalize(0.8 * exact_post + 0.2 * nb_post)
        elif cnt > 0:
            if cons >= 0.6:
                nb_blend = normalize(0.3 * exact_post + 0.7 * nb_post)
            else:
                nb_blend = normalize(0.5 * exact_post + 0.5 * global_dist)
        else:
            if cons >= 0.6:
                nb_blend = normalize(0.7 * nb_post + 0.3 * global_dist)
            else:
                nb_blend = global_dist

        # KL divergence exact vs neighbor
        kl_exact_nb = 0.0
        if exact_post is not None:
            for j in range(3):
                p_e = exact_post[j] if exact_post[j] > 1e-12 else 1e-12
                p_n = nb_blend[j] if nb_blend[j] > 1e-12 else 1e-12
                kl_exact_nb += p_e * np.log(p_e / p_n)

        # Margin in exact
        sorted_e = np.sort(exact_post)[::-1]
        margin_exact = sorted_e[0] - sorted_e[1]

        # Features
        router_rows.append({
            "key": k,
            "count": min(cnt, 100),
            "log_count": np.log1p(cnt),
            "entropy": ent,
            "margin_exact": margin_exact,
            "max_prob_exact": exact_post.max(),
            "neighbor_consistency": cons,
            "kl_exact_neighbor": min(kl_exact_nb, 5.0),
            "is_seen": int(cnt > 0),
            "is_lookup": int(cnt >= 30 and ent < 0.12 and exact_post.max() >= 0.95),
            "nearest_dist": float(dists[0]),
            "exact_mode": int(exact_post.argmax()),
            "neighbor_mode": int(nb_blend.argmax()),
            "prior_mode": int(global_dist.argmax()),
        })

    router_df = pd.DataFrame(router_rows)
    print(f"Router keys: {len(router_df)}")
    print(f"  seen: {(router_df['is_seen']==1).sum()}, unseen: {(router_df['is_seen']==0).sum()}")

    # ---- Generate router labels: which source is best for each key? ----
    # For each canary key, compute per-sample F1 contribution for each source
    key_to_canary_idx: Dict[Tuple, List[int]] = defaultdict(list)
    for i in router_canary_idx:
        k = tuple(int(v) for v in train[features].iloc[i])
        key_to_canary_idx[k].append(i)

    best_source = {}
    for row in router_rows:
        k = row["key"]
        indices = key_to_canary_idx.get(k, [])
        if not indices or len(indices) < 3:
            # Not enough data: default to exact if seen, neighbor if unseen
            best_source[k] = 0 if row["is_seen"] else 1
            continue

        y_true = y[indices]
        scores = []
        for source in range(3):
            # source 0 = exact, 1 = neighbor/blend, 2 = prior
            if source == 0:
                post = key_posterior.get(k, global_dist)
            elif source == 1:
                # Use the blended backoff
                neighbors2, dists2 = find_neighbors(k, K_NEIGHBORS)
                cons2, _ = neighbor_consistency(neighbors2)
                nb2 = neighbor_posterior(neighbors2, dists2)
                cnt2 = key_count.get(k, 0)
                ent2 = key_entropy.get(k, 1.0)
                ep = key_posterior.get(k, global_dist)
                if cnt2 >= 30 and ent2 < 0.12 and ep.max() >= 0.95:
                    post = ep
                elif cnt2 >= 10 and ent2 < 0.30:
                    post = normalize(0.8 * ep + 0.2 * nb2)
                elif cnt2 > 0:
                    if cons2 >= 0.6:
                        post = normalize(0.3 * ep + 0.7 * nb2)
                    else:
                        post = normalize(0.5 * ep + 0.5 * global_dist)
                else:
                    if cons2 >= 0.6:
                        post = normalize(0.7 * nb2 + 0.3 * global_dist)
                    else:
                        post = global_dist
            else:
                post = global_dist

            pred = np.array([post.argmax()] * len(indices))
            scores.append(f1_score(y_true, pred, average="macro"))

        best_source[k] = int(np.argmax(scores))

    router_df["label"] = router_df["key"].map(best_source)
    src_counts = router_df["label"].value_counts().sort_index()
    print(f"Label distribution: exact={src_counts.get(0,0)}, neighbor={src_counts.get(1,0)}, prior={src_counts.get(2,0)}")

    # ---- Hard protections ----
    # High count + low entropy keys must stay exact
    protect_mask = (router_df["count"] >= 30) & (router_df["entropy"] < 0.15) & (router_df["margin_exact"] > 0.5)
    router_df.loc[protect_mask, "label"] = 0
    print(f"Protected exact keys: {protect_mask.sum()}")

    # Class1/2 protection: don't let backoff erase minority classes
    for cls in [1, 2]:
        cls_mask = (router_df["exact_mode"] == cls) & (router_df["count"] >= 10)
        overridden = cls_mask & (router_df["label"] != 0)
        router_df.loc[cls_mask, "label"] = 0
        print(f"Class {cls} protected exact: {overridden.sum()}")

    # ---- Train router ----
    feature_cols = ["count", "log_count", "entropy", "margin_exact", "max_prob_exact",
                     "neighbor_consistency", "kl_exact_neighbor", "is_seen", "is_lookup", "nearest_dist"]
    X = router_df[feature_cols].to_numpy(dtype=np.float32)
    y_router = router_df["label"].astype(int).to_numpy()

    # Simple train/val split for router
    rng = np.random.default_rng(42)
    r_idx = rng.permutation(len(X))
    n_train_r = int(len(X) * 0.7)
    X_rtr, X_rva = X[r_idx[:n_train_r]], X[r_idx[n_train_r:]]
    y_rtr, y_rva = y_router[r_idx[:n_train_r]], y_router[r_idx[n_train_r:]]

    router = LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        random_state=42, n_jobs=4, verbosity=-1,
    )
    router.fit(X_rtr, y_rtr)
    r_acc = router.score(X_rva, y_rva)
    print(f"Router val accuracy: {r_acc:.4f}")

    # ---- Apply router to all test keys ----
    test_key_to_idx = defaultdict(list)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        test_key_to_idx[tuple(int(v) for v in row)].append(i)

    test_preds = np.zeros(len(test), dtype=int)
    source_counts_test = {0: 0, 1: 0, 2: 0}

    for k, indices in test_key_to_idx.items():
        cnt = key_count.get(k, 0)
        ent = key_entropy.get(k, 1.0)
        ep = key_posterior.get(k, global_dist)
        neighbors, dists = find_neighbors(k, K_NEIGHBORS)
        cons, _ = neighbor_consistency(neighbors)
        nb = neighbor_posterior(neighbors, dists)

        # Build router features
        kl_en = 0.0
        bp = ep
        for j in range(3):
            pe = ep[j] if ep[j] > 1e-12 else 1e-12
            pn = nb[j] if nb[j] > 1e-12 else 1e-12
            kl_en += pe * np.log(pe / pn)
        sorted_e = np.sort(ep)[::-1]
        margin_e = sorted_e[0] - sorted_e[1]

        rf = np.array([[
            min(cnt, 100), np.log1p(cnt), ent, margin_e, ep.max(),
            cons, min(kl_en, 5.0), int(cnt > 0),
            int(cnt >= 30 and ent < 0.12 and ep.max() >= 0.95), float(dists[0]),
        ]], dtype=np.float32)

        src = int(router.predict(rf)[0])

        # Build blended neighbor posterior
        if cnt >= 30 and ent < 0.12 and ep.max() >= 0.95:
            nb_blend = ep
        elif cnt >= 10 and ent < 0.30:
            nb_blend = normalize(0.8 * ep + 0.2 * nb)
        elif cnt > 0:
            if cons >= 0.6:
                nb_blend = normalize(0.3 * ep + 0.7 * nb)
            else:
                nb_blend = normalize(0.5 * ep + 0.5 * global_dist)
        else:
            if cons >= 0.6:
                nb_blend = normalize(0.7 * nb + 0.3 * global_dist)
            else:
                nb_blend = global_dist

        if src == 0:
            post = ep
        elif src == 1:
            post = nb_blend
        else:
            post = global_dist

        source_counts_test[src] += len(indices)
        for idx in indices:
            test_preds[idx] = int(post.argmax())

    # ---- Evaluate on canary ----
    def eval_router(name, indices):
        yt = y[list(indices)]
        yp = []
        for i in indices:
            k = tuple(int(v) for v in train[features].iloc[i])
            cnt = key_count.get(k, 0)
            ent = key_entropy.get(k, 1.0)
            ep = key_posterior.get(k, global_dist)
            nbrs, d = find_neighbors(k, K_NEIGHBORS)
            cs, _ = neighbor_consistency(nbrs)
            nb = neighbor_posterior(nbrs, d)

            kle = 0.0
            for j in range(3):
                pe = ep[j] if ep[j] > 1e-12 else 1e-12
                pn = nb[j] if nb[j] > 1e-12 else 1e-12
                kle += pe * np.log(pe / pn)
            se = np.sort(ep)[::-1]
            me = se[0] - se[1]

            rf = np.array([[
                min(cnt, 100), np.log1p(cnt), ent, me, ep.max(),
                cs, min(kle, 5.0), int(cnt > 0),
                int(cnt >= 30 and ent < 0.12 and ep.max() >= 0.95), float(d[0]),
            ]], dtype=np.float32)
            src = int(router.predict(rf)[0])

            if cnt >= 30 and ent < 0.12 and ep.max() >= 0.95:
                nb_bl = ep
            elif cnt >= 10 and ent < 0.30:
                nb_bl = normalize(0.8 * ep + 0.2 * nb)
            elif cnt > 0:
                if cs >= 0.6:
                    nb_bl = normalize(0.3 * ep + 0.7 * nb)
                else:
                    nb_bl = normalize(0.5 * ep + 0.5 * global_dist)
            else:
                if cs >= 0.6:
                    nb_bl = normalize(0.7 * nb + 0.3 * global_dist)
                else:
                    nb_bl = global_dist

            post = ep if src == 0 else (nb_bl if src == 1 else global_dist)
            yp.append(int(post.argmax()))
        yp = np.array(yp)
        f1 = f1_score(yt, yp, average="macro")
        f1_per = f1_score(yt, yp, average=None)
        return f1, f1_per

    cg_f1, cg_per = eval_router("Canary-G", canary_g_idx)
    cs_f1, cs_per = eval_router("Canary-S", canary_s_idx)
    ws_f1, ws_per = eval_router("Weak-Seen", weak_seen_idx)

    print(f"\n=== Router Results ===")
    print(f"Canary-G:  {cg_f1:.6f}  {np.round(cg_per, 6)}")
    print(f"Canary-S:  {cs_f1:.6f}  {np.round(cs_per, 6)}")
    if weak_seen_idx:
        print(f"Weak-Seen: {ws_f1:.6f}  {np.round(ws_per, 6)}")

    # vs C1 and D1 v2
    c1_cg, c1_cs = 0.236816, 0.747390
    d1v2_cg, d1v2_cs = 0.322772, 0.721429
    print(f"\n=== Comparison ===")
    print(f"        Canary-G  Canary-S")
    print(f"C1:      {c1_cg:.4f}    {c1_cs:.4f}")
    print(f"D1 v2:   {d1v2_cg:.4f}    {d1v2_cs:.4f}")
    print(f"Router:  {cg_f1:.4f}    {cs_f1:.4f}")

    print(f"\nRouter source usage: exact={source_counts_test[0]}, neighbor={source_counts_test[1]}, prior={source_counts_test[2]}")
    print(f"  ({100*source_counts_test[0]/len(test):.1f}% / {100*source_counts_test[1]/len(test):.1f}% / {100*source_counts_test[2]/len(test):.1f}%)")

    # Test distribution
    dist = {int(k): int(v) for k, v in pd.Series(test_preds).value_counts().sort_index().items()}
    print(f"Test distribution: {dist}")

    sub_path = PKG_ROOT / "提交结果" / "submission_v4_router.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_preds}).to_csv(sub_path, index=False)
    print(f"Saved: {sub_path}")


if __name__ == "__main__":
    main()
