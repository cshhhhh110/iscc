"""
v4.0 Experiment C1 — B + Knowledge Distillation (no pseudo-label)
KD only affects residual layer. Lookup layer unchanged.
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

# Lookup params
LOOKUP_ALPHA = 1.0
LOOKUP_ENTROPY_THRESHOLD = 0.18
LOOKUP_MAX_PROB_THRESHOLD = 0.95
LOOKUP_MIN_COUNT = 30
ANCHOR_MIN = 5
ANCHOR_FRAC = 0.10

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


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in ("name", "label")]


def entropy(probs: np.ndarray) -> float:
    p = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def apply_smote(X_tr, y_tr, seed):
    from imblearn.over_sampling import SMOTE as SMOTE_cls
    cnts = np.bincount(y_tr, minlength=3)
    k = max(1, min(5, int(cnts.min()) - 1))
    sm = SMOTE_cls(k_neighbors=k, random_state=seed)
    return sm.fit_resample(X_tr, y_tr)


def blend_soft_target(teacher_oof: np.ndarray) -> np.ndarray:
    log_p = np.log(np.clip(teacher_oof, 1e-12, None))
    t1 = np.exp(log_p / 1.0)
    t3 = np.exp(log_p / 3.0)
    return normalize(0.05 * t1 + 0.95 * t3)


def main():
    # ---- Load splits ----
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

    # ---- Key stats & lookup table ----
    print("Building lookup table...")
    key_to_labels: Dict[Tuple, List[int]] = defaultdict(list)
    key_to_indices: Dict[Tuple, List[int]] = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        key_to_labels[k].append(int(y[i]))
        key_to_indices[k].append(int(i))

    global_dist = np.bincount(y, minlength=3).astype(np.float64) / len(y)

    lookup_keys: set = set()
    lookup_posterior: Dict[Tuple, np.ndarray] = {}
    for k, lbls in key_to_labels.items():
        count = len(lbls)
        if count < LOOKUP_MIN_COUNT:
            continue
        cls_counts = np.bincount(lbls, minlength=3).astype(np.float64)
        posterior = (cls_counts + LOOKUP_ALPHA * global_dist) / (count + LOOKUP_ALPHA)
        posterior = posterior / posterior.sum()
        if entropy(posterior) < LOOKUP_ENTROPY_THRESHOLD and posterior.max() >= LOOKUP_MAX_PROB_THRESHOLD:
            lookup_keys.add(k)
            lookup_posterior[k] = posterior

    print(f"Lookup keys: {len(lookup_keys)}")

    # ---- Build residual training set (exclude Canary-G, anchor for lookup keys) ----
    residual_train_idx: List[int] = []
    for i in range(len(train)):
        if i in canary_g_idx:
            continue
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys:
            indices = key_to_indices[k]
            n_anchor = max(ANCHOR_MIN, int(len(indices) * ANCHOR_FRAC))
            anchor_set = set(sorted(indices)[:n_anchor])
            if i in anchor_set:
                residual_train_idx.append(i)
        else:
            residual_train_idx.append(i)
    residual_train_idx = sorted(residual_train_idx)
    print(f"Residual training: {len(residual_train_idx)} samples")

    # ---- 132-dim features for KD ----
    print("Building 132-dim features...")
    from collections import Counter
    count_map = {k: len(v) for k, v in key_to_indices.items()}
    nunique_map = {k: int(np.bincount(lbls, minlength=3).astype(bool).sum()) for k, lbls in key_to_labels.items()}
    def build_dist_map():
        dm = {}
        for k, lbls in key_to_labels.items():
            c = len(lbls)
            cc = np.bincount(lbls, minlength=3)
            dm[k] = (float(cc[0]/max(c,1)), float(cc[1]/max(c,1)), float(cc[2]/max(c,1)), c)
        return dm
    dist_map = build_dist_map()

    def make_model_features(df, features, pair_features):
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
            freq.reshape(-1, 1),
            np.log1p(freq).reshape(-1, 1),
            part.sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
            (part > 0).sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1),
            conflict.reshape(-1, 1),
            (conflict == 1).astype(np.float32).reshape(-1, 1),
            (conflict >= 2).astype(np.float32).reshape(-1, 1),
            dist[:, 0:1], dist[:, 1:2], dist[:, 2:3], dist[:, 3:4],
        ])
        return np.column_stack(blocks)

    x_all_132 = make_model_features(train, features, pair_features)
    x_test_132 = make_model_features(test, features, pair_features)

    # ---- KD split: exclude Canary-G key samples ----
    kd_train_idx = np.array([i for i in range(len(train)) if i not in canary_g_idx])
    X_kd = x_all_132[kd_train_idx]
    y_kd = y[kd_train_idx]

    # Groups for StratifiedGroupKFold
    key_arr = np.array([tuple(int(v) for v in row) for row in train[features].itertuples(index=False, name=None)])
    key_kd = key_arr[kd_train_idx]
    gid = {}
    groups_kd_arr = []
    for k in key_kd:
        kt = tuple(k)
        if kt not in gid:
            gid[kt] = len(gid)
        groups_kd_arr.append(gid[kt])

    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    splits = list(cv.split(X_kd, y_kd, np.array(groups_kd_arr, dtype=int)))

    # ---- Phase 1: Train 4 Teachers ----
    print("\n=== Phase 1: Teacher Training ===")
    teacher_oofs = {}
    teacher_seeds = [200, 247, 294, 400]  # 3 LGBM + 1 XGB

    for t_idx, t_seed in enumerate(teacher_seeds):
        oof = np.zeros((len(kd_train_idx), 3), dtype=np.float32)
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            X_tr, X_va = X_kd[tr_idx], X_kd[va_idx]
            y_tr = y_kd[tr_idx]
            X_sm, y_sm = apply_smote(X_tr, y_tr, t_seed + fold_id * 100)
            if t_idx < 3:
                m = LGBMClassifier(random_state=t_seed + fold_id, **LGBM_TEACHER_PARAMS)
            else:
                m = XGBClassifier(random_state=t_seed + fold_id, **XGB_TEACHER_PARAMS)
            m.fit(X_sm, y_sm)
            oof[va_idx] = m.predict_proba(X_va)
        name = f"lgbm_{t_idx}" if t_idx < 3 else "xgb_0"
        teacher_oofs[name] = oof
        f1 = f1_score(y_kd, oof.argmax(axis=1), average="macro")
        print(f"  Teacher {name} OOF F1: {f1:.6f}")

    # Teacher blend
    teacher_names = sorted(teacher_oofs.keys())
    n_teachers = len(teacher_names)
    teacher_blend = normalize(sum(np.clip(teacher_oofs[n], 1e-12, None) for n in teacher_names) / n_teachers)
    teacher_f1 = f1_score(y_kd, teacher_blend.argmax(axis=1), average="macro")
    print(f"  Teacher blend F1: {teacher_f1:.6f}")

    # ---- Phase 2: Knowledge Distillation ----
    print("\n=== Phase 2: Knowledge Distillation ===")
    soft_target = blend_soft_target(teacher_blend)

    # Hard target for mixed supervision
    hard_target = np.eye(3, dtype=np.float32)[y_kd]

    # Classify samples as deterministic or ambiguous
    is_deterministic = np.zeros(len(kd_train_idx), dtype=bool)
    for i in range(len(kd_train_idx)):
        orig_idx = kd_train_idx[i]
        k = tuple(int(v) for v in train[features].iloc[orig_idx])
        if k in lookup_keys:
            is_deterministic[i] = True
        elif k in nunique_map and nunique_map[k] == 1:
            is_deterministic[i] = True
    n_det = int(is_deterministic.sum())
    n_amb = len(kd_train_idx) - n_det
    print(f"  Deterministic: {n_det}, Ambiguous: {n_amb}")

    # Mixed target
    kd_target = np.zeros_like(soft_target)
    for i in range(len(kd_train_idx)):
        if is_deterministic[i]:
            kd_target[i] = 0.8 * soft_target[i] + 0.2 * hard_target[i]
        else:
            kd_target[i] = 0.6 * soft_target[i] + 0.4 * hard_target[i]

    # ---- Phase 3: Train Students ----
    print("\n=== Phase 3: Student Training ===")
    student_oofs = []
    student_test_probs_list = []

    for student_id in range(N_STUDENTS):
        s_seed = 42 + student_id * 73
        oof = np.zeros((len(kd_train_idx), 3), dtype=float)
        test_probs = np.zeros((len(test), 3), dtype=float)
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            for class_pos in range(3):
                reg = LGBMRegressor(random_state=s_seed, **LGBM_STUDENT_PARAMS)
                reg.fit(X_kd[tr_idx], kd_target[tr_idx, class_pos])
                oof[va_idx, class_pos] = reg.predict(X_kd[va_idx])
                test_probs[:, class_pos] += reg.predict(x_test_132) / N_FOLDS
            oof[va_idx] = normalize(oof[va_idx])
        student_oofs.append(oof)
        student_test_probs_list.append(normalize(test_probs))

    student_oof_avg = normalize(np.mean(student_oofs, axis=0))
    student_test_avg = normalize(np.mean(student_test_probs_list, axis=0))
    student_oof_labels = student_oof_avg.argmax(axis=1)
    student_f1 = f1_score(y_kd, student_oof_labels, average="macro")
    print(f"  Student OOF F1: {student_f1:.6f}")

    # ---- Build full OOF (lookup + student residual) ----
    oof_full = np.zeros((len(train), 3), dtype=np.float32)
    for i in range(len(train)):
        if i in canary_g_idx:
            continue
        k = tuple(int(v) for v in train[features].iloc[i])
        if k in lookup_keys and i not in set(residual_train_idx):
            oof_full[i] = lookup_posterior[k].astype(np.float32)

    # Fill student predictions for residual training samples
    for pos, orig_idx in enumerate(kd_train_idx):
        if np.all(oof_full[orig_idx] == 0):  # only fill if not already filled by lookup
            oof_full[orig_idx] = student_oof_avg[pos]

    # ---- Evaluate ----
    def eval_canary(name, indices):
        pred = oof_full[list(indices)].argmax(axis=1)
        true = y[list(indices)]
        f1 = f1_score(true, pred, average="macro")
        per = f1_score(true, pred, average=None)
        return f1, per

    cg_f1, cg_per = eval_canary("Canary-G", canary_g_idx)
    cs_f1, cs_per = eval_canary("Canary-S", canary_s_idx)
    n_g, n_s = len(canary_g_idx), len(canary_s_idx)
    weighted = (cg_f1 * n_g + cs_f1 * n_s) / (n_g + n_s)

    print(f"\n=== C1 Results ===")
    print(f"Canary-G: {cg_f1:.6f}  per-class: {np.round(cg_per, 6)}")
    print(f"Canary-S: {cs_f1:.6f}  per-class: {np.round(cs_per, 6)}")
    print(f"Weighted: {weighted:.6f}")

    # ---- vs B ----
    b_cg = 0.236816
    b_cs = 0.714107
    b_w = 0.499020
    print(f"\n=== vs B ===")
    print(f"Canary-G: {cg_f1:.6f} vs {b_cg:.6f}  ({'+' if cg_f1>b_cg else ''}{cg_f1-b_cg:+.6f})")
    print(f"Canary-S: {cs_f1:.6f} vs {b_cs:.6f}  ({'+' if cs_f1>b_cs else ''}{cs_f1-b_cs:+.6f})")
    print(f"Weighted: {weighted:.6f} vs {b_w:.6f}  ({'+' if weighted>b_w else ''}{weighted-b_w:+.6f})")

    if cg_f1 > b_cg + 0.001 and cs_f1 > b_cs - 0.005 and weighted > b_w:
        print("\n>>> C1 EFFECTIVE: KD improved Canary-G, Canary-S preserved. <<<")
    elif cs_f1 < b_cs - 0.01:
        print("\n>>> C1 FAILED: Canary-S dropped. KD destabilized seen patterns. <<<")
    else:
        print("\n>>> C1 MARGINAL. <<<")

    # ---- Test prediction ----
    n_lookup_test = 0
    test_preds = np.zeros((len(test), 3), dtype=np.float32)
    for i, row in enumerate(test[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        if k in lookup_keys:
            test_preds[i] = lookup_posterior[k].astype(np.float32)
            n_lookup_test += 1
        else:
            test_preds[i] = student_test_avg[i]

    test_labels = test_preds.argmax(axis=1)
    dist = {int(k): int(v) for k, v in pd.Series(test_labels).value_counts().sort_index().items()}
    print(f"\nLookup hits in test: {n_lookup_test}/{len(test)} ({100*n_lookup_test/len(test):.1f}%)")
    print(f"Test prediction distribution: {dist}")

    # ---- Class shift check ----
    b_dist = {0: 13502, 1: 3360, 2: 3138}
    for cls in [0, 1, 2]:
        b_val = b_dist.get(cls, 0)
        c_val = dist.get(cls, 0)
        shift = 100 * (c_val - b_val) / max(b_val, 1)
        flag = " ⚠️ >10%" if abs(shift) > 10 else ""
        print(f"  Class {cls}: {c_val} vs B={b_val} ({'+' if shift>0 else ''}{shift:.1f}%){flag}")

    # Save
    sub_path = PKG_ROOT / "提交结果" / "submission_v4_c1.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"name": test["name"], "label": test_labels}).to_csv(sub_path, index=False)
    print(f"\nSaved: {sub_path}")


if __name__ == "__main__":
    main()
