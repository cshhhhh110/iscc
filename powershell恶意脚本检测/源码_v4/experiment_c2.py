"""
v4.0 Experiment C2 — C1 + Strict Pseudo-Label
Gate: 4-teacher agree, conf>=0.97, margin>=0.25, std<0.05
Caps: class0=300, class2=100, class1=all (2 only)
"""
from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
TEST_PATH = PKG_ROOT / "data_test.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
N_FOLDS = 5
RANDOM_SEED = 2026
N_STUDENTS = 3

# Lookup
LOOKUP_ALPHA = 1.0
LOOKUP_ENTROPY_THRESHOLD = 0.18
LOOKUP_MAX_PROB_THRESHOLD = 0.95
LOOKUP_MIN_COUNT = 30
ANCHOR_MIN = 5
ANCHOR_FRAC = 0.10

# Pseudo gate
PSEUDO_CONF = 0.97
PSEUDO_MARGIN = 0.25
PSEUDO_STD = 0.05
PSEUDO_CAPS = {0: 300, 1: 9999, 2: 100}
PSEUDO_WEIGHT = 0.3

LGBM_TEACHER_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=-1,
)
XGB_TEACHER_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=0,
)
LGBM_STUDENT_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=1, verbosity=-1,
)


def feature_columns(df): return [c for c in df.columns if c not in ("name", "label")]
def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))
def normalize(probs):
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)
def apply_smote(X_tr, y_tr, seed):
    from imblearn.over_sampling import SMOTE as S
    cnts = np.bincount(y_tr, minlength=3)
    k = max(1, min(5, int(cnts.min()) - 1))
    return S(k_neighbors=k, random_state=seed).fit_resample(X_tr, y_tr)
def blend_soft_target(teacher_oof):
    log_p = np.log(np.clip(teacher_oof, 1e-12, None))
    return normalize(0.05 * np.exp(log_p/1.0) + 0.95 * np.exp(log_p/3.0))
def make_model_features(df, features, pair_features, count_map, nunique_map, dist_map):
    part = df[features].copy()
    blocks = [part[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in features]
    for c1, c2 in pair_features:
        cross = part[c1].astype(np.int16) * 10 + part[c2].astype(np.int16)
        blocks.append(cross.to_numpy(dtype=np.float32).reshape(-1, 1))
    keys = [tuple(int(v) for v in row) for row in part.itertuples(index=False, name=None)]
    freq = np.array([count_map.get(k, 0) for k in keys], dtype=np.float32)
    conflict = np.array([nunique_map.get(k, 1) for k in keys], dtype=np.float32)
    default_dist = (0.49, 0.26, 0.24, 0)
    dist = np.array([dist_map.get(k, default_dist) for k in keys], dtype=np.float32)
    blocks.extend([
        freq.reshape(-1, 1), np.log1p(freq).reshape(-1, 1),
        part.sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
        (part > 0).sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
        conflict.reshape(-1, 1),
        (conflict == 1).astype(np.float32).reshape(-1, 1),
        (conflict >= 2).astype(np.float32).reshape(-1, 1),
        dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4],
    ])
    return np.column_stack(blocks)


def main():
    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        split_meta = json.load(f)
    canary_g_idx = set(split_meta["canary_g_indices"])
    canary_s_idx = set(split_meta["canary_s_indices"])
    print(f"Canary-G: {len(canary_g_idx)}, Canary-S: {len(canary_s_idx)}")

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    features = feature_columns(train)
    pair_features = list(combinations(features, 2))
    y = train["label"].astype(int).to_numpy()

    # ---- Key stats ----
    key_to_labels: Dict[Tuple, List[int]] = defaultdict(list)
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        key_to_labels[k].append(int(y[i]))
        key_to_indices[k].append(int(i))

    global_dist = np.bincount(y, minlength=3).astype(np.float64) / len(y)

    # Lookup table
    lookup_keys, lookup_posterior = set(), {}
    for k, lbls in key_to_labels.items():
        cnt = len(lbls)
        if cnt < LOOKUP_MIN_COUNT: continue
        cc = np.bincount(lbls, minlength=3).astype(np.float64)
        post = (cc + LOOKUP_ALPHA * global_dist) / (cnt + LOOKUP_ALPHA)
        post = post / post.sum()
        if entropy(post) < LOOKUP_ENTROPY_THRESHOLD and post.max() >= LOOKUP_MAX_PROB_THRESHOLD:
            lookup_keys.add(k); lookup_posterior[k] = post
    print(f"Lookup keys: {len(lookup_keys)}")

    # Residual training indices
    residual_train_idx = []
    for i in range(len(train)):
        if i in canary_g_idx: continue
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys:
            indices = key_to_indices[k]
            n_anchor = max(ANCHOR_MIN, int(len(indices) * ANCHOR_FRAC))
            anchor_set = set(sorted(indices)[:n_anchor])
            if i in anchor_set: residual_train_idx.append(i)
        else:
            residual_train_idx.append(i)
    residual_train_idx = sorted(residual_train_idx)
    print(f"Residual training: {len(residual_train_idx)} samples")

    # Maps for 132-dim
    count_map = {k: len(v) for k, v in key_to_indices.items()}
    nunique_map = {k: int(np.bincount(v, minlength=3).astype(bool).sum()) for k, v in key_to_labels.items()}
    dist_map = {}
    for k, lbls in key_to_labels.items():
        c = len(lbls); cc = np.bincount(lbls, minlength=3)
        dist_map[k] = (float(cc[0]/max(c,1)), float(cc[1]/max(c,1)), float(cc[2]/max(c,1)), c)

    x_all = make_model_features(train, features, pair_features, count_map, nunique_map, dist_map)
    x_test = make_model_features(test, features, pair_features, count_map, nunique_map, dist_map)

    # KD training set
    kd_train_idx = np.array([i for i in range(len(train)) if i not in canary_g_idx])
    X_kd, y_kd = x_all[kd_train_idx], y[kd_train_idx]

    key_arr = np.array([tuple(int(v) for v in row) for row in train[features].itertuples(index=False, name=None)])
    key_kd = key_arr[kd_train_idx]
    gid = {}; groups_kd = []
    for k in key_kd:
        kt = tuple(k)
        if kt not in gid: gid[kt] = len(gid)
        groups_kd.append(gid[kt])

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X_kd, y_kd, np.array(groups_kd, dtype=int)))

    # ---- Phase 1: Train Teachers ----
    print("\n=== Phase 1: Teachers ===")
    teacher_seeds = [200, 247, 294, 400]
    teacher_test = np.zeros((4, len(test), 3), dtype=np.float32)
    teacher_oofs = {}

    for t_idx, t_seed in enumerate(teacher_seeds):
        oof = np.zeros((len(kd_train_idx), 3), dtype=np.float32)
        for fold_id, (tr, va) in enumerate(splits):
            X_tr, X_va = X_kd[tr], X_kd[va]
            X_sm, y_sm = apply_smote(X_tr, y_kd[tr], t_seed + fold_id * 100)
            if t_idx < 3:
                m = LGBMClassifier(random_state=t_seed + fold_id, **LGBM_TEACHER_PARAMS)
            else:
                m = XGBClassifier(random_state=t_seed + fold_id, **XGB_TEACHER_PARAMS)
            m.fit(X_sm, y_sm)
            oof[va] = m.predict_proba(X_va)
            teacher_test[t_idx] += m.predict_proba(x_test) / N_FOLDS

        name = f"lgbm_{t_idx}" if t_idx < 3 else "xgb_0"
        teacher_oofs[name] = oof
        print(f"  {name}: {f1_score(y_kd, oof.argmax(axis=1), average='macro'):.6f}")

    teacher_blend = normalize(sum(np.clip(teacher_oofs[n], 1e-12, None) for n in teacher_oofs) / 4)
    print(f"  Blend: {f1_score(y_kd, teacher_blend.argmax(axis=1), average='macro'):.6f}")

    # ---- Gate for pseudo-label ----
    print("\n=== Pseudo-label Gate ===")
    t_preds = teacher_test.argmax(axis=2)
    agree_all = (t_preds[0] == t_preds[1]) & (t_preds[0] == t_preds[2]) & (t_preds[0] == t_preds[3])
    mean_conf = teacher_test.max(axis=2).mean(axis=0)
    margins_arr = np.zeros(len(test))
    for i in range(len(test)):
        sp = np.sort(teacher_test[:, i].mean(axis=0))[::-1]
        margins_arr[i] = sp[0] - sp[1]
    std_arr = teacher_test.max(axis=2).std(axis=0)

    test_keys = [tuple(int(v) for v in row) for row in test[features].itertuples(index=False, name=None)]
    is_lookup = np.array([k in lookup_keys for k in test_keys])

    gate_pass = agree_all & (mean_conf >= PSEUDO_CONF) & (margins_arr >= PSEUDO_MARGIN) & (std_arr < PSEUDO_STD) & ~is_lookup
    pseudo_labels_all = t_preds.mean(axis=0).astype(int)[gate_pass]

    # Per-class caps
    pseudo_indices = np.where(gate_pass)[0]
    pseudo_confs = mean_conf[gate_pass]
    pseudo_classes = pseudo_labels_all
    selected = []
    for cls in range(3):
        cls_idx = pseudo_indices[pseudo_classes == cls]
        cls_conf = pseudo_confs[pseudo_classes == cls]
        order = np.argsort(cls_conf)[::-1]  # descending confidence
        cap = PSEUDO_CAPS.get(cls, 9999)
        take = cls_idx[order[:cap]]
        selected.extend(int(i) for i in take)
    selected = sorted(selected)

    pseudo_cls_dist = {int(c): int((pseudo_classes == c).sum()) for c in range(3)}
    selected_cls = [int(t_preds.mean(axis=0).astype(int)[i]) for i in selected]
    selected_dist = {int(c): int(selected_cls.count(c)) for c in range(3)}

    print(f"  Gate pass: {gate_pass.sum()} (class0={pseudo_cls_dist[0]}, c1={pseudo_cls_dist[1]}, c2={pseudo_cls_dist[2]})")
    print(f"  After caps: {len(selected)} (class0={selected_dist[0]}, c1={selected_dist[1]}, c2={selected_dist[2]})")

    # ---- Build pseudo-label training data ----
    pseudo_df = test.iloc[selected][features].copy()
    pseudo_df["label"] = selected_cls
    pseudo_x = make_model_features(pseudo_df, features, pair_features, count_map, nunique_map, dist_map)
    pseudo_y = np.array(selected_cls, dtype=int)
    pseudo_w = np.full(len(pseudo_df), PSEUDO_WEIGHT, dtype=np.float32)
    print(f"  Pseudo samples: {len(pseudo_df)} (weight={PSEUDO_WEIGHT})")

    # ---- Phase 2: KD with pseudo-label ----
    print("\n=== Phase 2: KD with Pseudo-Label ===")
    soft_target = blend_soft_target(teacher_blend)
    hard_target = np.eye(3, dtype=np.float32)[y_kd]

    is_det = np.zeros(len(kd_train_idx), dtype=bool)
    for i in range(len(kd_train_idx)):
        orig_idx = kd_train_idx[i]
        k = tuple(int(v) for v in train[features].iloc[orig_idx])
        if k in lookup_keys or (k in nunique_map and nunique_map[k] == 1):
            is_det[i] = True

    kd_target = np.zeros_like(soft_target)
    for i in range(len(kd_train_idx)):
        if is_det[i]:
            kd_target[i] = 0.8 * soft_target[i] + 0.2 * hard_target[i]
        else:
            kd_target[i] = 0.6 * soft_target[i] + 0.4 * hard_target[i]

    # Extend with pseudo-label
    X_student = np.vstack([X_kd, pseudo_x])
    target_student = np.vstack([kd_target, np.eye(3, dtype=np.float32)[pseudo_y]])  # soft for pseudo
    weights_student = np.concatenate([np.ones(len(kd_train_idx), dtype=np.float32), pseudo_w])
    is_student = np.concatenate([np.ones(len(kd_train_idx), dtype=bool), np.zeros(len(pseudo_df), dtype=bool)])

    # Student training with pseudo incorporated (no fold split needed, just retrain)
    # Use same splits but extend training side
    print("  Training students...")
    student_oofs = []
    student_test_probs_list = []

    for student_id in range(N_STUDENTS):
        s_seed = 42 + student_id * 73
        oof = np.zeros((len(kd_train_idx), 3), dtype=float)
        test_probs = np.zeros((len(test), 3), dtype=float)

        for fold_id, (tr, va) in enumerate(splits):
            X_tr_real, y_tr_real = X_kd[tr], kd_target[tr]
            w_tr_real = np.ones(len(tr), dtype=np.float32)

            # Combine real train + all pseudo
            X_tr_comb = np.vstack([X_tr_real, pseudo_x])
            y_tr_comb = np.vstack([y_tr_real, np.eye(3, dtype=np.float32)[pseudo_y]])
            w_tr_comb = np.concatenate([w_tr_real, pseudo_w])

            for class_pos in range(3):
                reg = LGBMRegressor(random_state=s_seed, **LGBM_STUDENT_PARAMS)
                # Use sample_weight for pseudo
                reg.fit(X_tr_comb, y_tr_comb[:, class_pos], sample_weight=w_tr_comb)
                oof[va, class_pos] = reg.predict(X_kd[va])
                test_probs[:, class_pos] += reg.predict(x_test) / N_FOLDS
            oof[va] = normalize(oof[va])

        student_oofs.append(oof)
        student_test_probs_list.append(normalize(test_probs))

    student_oof_avg = normalize(np.mean(student_oofs, axis=0))
    student_test_avg = normalize(np.mean(student_test_probs_list, axis=0))
    print(f"  Student OOF: {f1_score(y_kd, student_oof_avg.argmax(axis=1), average='macro'):.6f}")

    # ---- Build full OOF ----
    oof_full = np.zeros((len(train), 3), dtype=np.float32)
    for i in range(len(train)):
        if i in canary_g_idx: continue
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys and i not in set(residual_train_idx):
            oof_full[i] = lookup_posterior[k].astype(np.float32)
    for pos, orig_idx in enumerate(kd_train_idx):
        if np.all(oof_full[orig_idx] == 0):
            oof_full[orig_idx] = student_oof_avg[pos]

    # ---- Evaluate ----
    def eval_canary(desc, idx_set):
        pred = oof_full[list(idx_set)].argmax(axis=1)
        true = y[list(idx_set)]
        return f1_score(true, pred, average="macro"), f1_score(true, pred, average=None)

    cg_f1, cg_per = eval_canary("G", canary_g_idx)
    cs_f1, cs_per = eval_canary("S", canary_s_idx)
    ng, ns = len(canary_g_idx), len(canary_s_idx)
    weighted = (cg_f1 * ng + cs_f1 * ns) / (ng + ns)

    print(f"\n=== C2 Results ===")
    print(f"Canary-G: {cg_f1:.6f}  {np.round(cg_per, 6)}")
    print(f"Canary-S: {cs_f1:.6f}  {np.round(cs_per, 6)}")
    print(f"Weighted: {weighted:.6f}")

    # vs C1
    c1_cg = 0.236816; c1_cs = 0.747390; c1_w = 0.517305
    print(f"\n=== vs C1 ===")
    print(f"Canary-G: {cg_f1:.6f} vs {c1_cg:.6f}  ({'+' if cg_f1>c1_cg else ''}{cg_f1-c1_cg:+.6f})")
    print(f"Canary-S: {cs_f1:.6f} vs {c1_cs:.6f}  ({'+' if cs_f1>c1_cs else ''}{cs_f1-c1_cs:+.6f})")
    print(f"Weighted: {weighted:.6f} vs {c1_w:.6f}  ({'+' if weighted>c1_w else ''}{weighted-c1_w:+.6f})")

    if cs_f1 > c1_cs - 0.002 and weighted > c1_w:
        print("\n>>> C2 EFFECTIVE: Weighted improved, Canary-S preserved. <<<")
    elif cs_f1 < c1_cs - 0.005:
        print("\n>>> C2 FAILED: Canary-S dropped. Pseudo-label noise. <<<")
    else:
        print("\n>>> C2 MARGINAL. <<<")

    # Test prediction
    n_lookup_test = 0
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        if k in lookup_keys:
            test_preds[i] = lookup_posterior[k].astype(np.float32); n_lookup_test += 1
        else:
            test_preds[i] = student_test_avg[i]
    test_labels = test_preds.argmax(axis=1)
    dist = {int(k): int(v) for k, v in pd.Series(test_labels).value_counts().sort_index().items()}
    print(f"\nLookup hits in test: {n_lookup_test} ({100*n_lookup_test/len(test):.1f}%)")
    print(f"Test distribution: {dist}")
    c1_dist = {0: 13338, 1: 3347, 2: 3315}
    for cls in [0, 1, 2]:
        c1_v = c1_dist.get(cls, 0)
        c2_v = dist.get(cls, 0)
        shift = 100 * (c2_v - c1_v) / max(c1_v, 1)
        flag = " ⚠️ >10%" if abs(shift) > 10 else ""
        print(f"  Class {cls}: {c2_v} vs C1={c1_v}  ({'+' if shift>0 else ''}{shift:.1f}%){flag}")

    # Save
    sub_path = PKG_ROOT / "提交结果" / "submission_v4_c2.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_labels}).to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")


if __name__ == "__main__":
    main()
