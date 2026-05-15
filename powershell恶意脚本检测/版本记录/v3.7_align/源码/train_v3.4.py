"""
v3.7 align: exact_key5 feature alignment (teacher-student representation fix).

Key changes from v3.6:
- Always adds 5 exact_key frequency features to student input (was optional)
- These 5 features alone boost class1 by ~46 samples (3308→3354)
- Root cause: student lacked test/all split key frequency information
- Bias grid expanded for rescan around new distribution
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from itertools import combinations, product
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import (
    LABELS, N_FOLDS, N_STUDENTS,
    build_key_stats, candidate_metrics,
    default_model_dir, default_output_dir,
    feature_columns, key_tuple,
    make_model_features, normalize,
    package_root, resolve_data_path,
    save_key_stats, save_metadata, stats_maps,
    write_submission,
)

warnings.filterwarnings("ignore")

VERSION = "v3.7_align"
BASELINE_SUBMISSION_NAME = "submission_robust_model_balanced.csv"
T1_WEIGHT = 0.05  # override via --t1-weight

TEACHER_WEIGHTS = {
    "lgbm_tuned_00": 0.14273,
    "lgbm_tuned_01": 0.12394,
    "lgbm_tuned_02": 0.34562,
    "xgb_tuned_00": 0.38771,
}

# Narrow bias grid — c5 sweet-spot micro-search
STUDENT_WEIGHT_GRID = [1.00]
BIAS0_GRID = [1.00]
BIAS1_GRID = [1.04, 1.06, 1.08]
BIAS2_GRID = [1.06, 1.08, 1.10, 1.12, 1.14]  # expanded for exact_key5 rescan

LGBM_PARAMS = dict(
    n_estimators=300, learning_rate=0.04, num_leaves=31,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, n_jobs=4, verbosity=-1,
)


def exact_group_splits(train, y, features):
    group_ids = {}
    groups = []
    for row in train[features].itertuples(index=False, name=None):
        key = key_tuple(row)
        if key not in group_ids:
            group_ids[key] = len(group_ids)
        groups.append(group_ids[key])
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    return list(cv.split(train[features], y, np.asarray(groups, dtype=int)))


def load_teacher_predictions(teacher_dir, n_rows, n_test):
    teacher_oof = np.zeros((n_rows, len(LABELS)), dtype=float)
    teacher_test = np.zeros((n_test, len(LABELS)), dtype=float)
    for name, weight in TEACHER_WEIGHTS.items():
        path = teacher_dir / f"tuned_smoke_group_{name}.npz"
        data = np.load(path, allow_pickle=True)
        teacher_oof += weight * np.clip(data["oof"], 1e-12, None)
        teacher_test += weight * np.clip(data["test_probs"], 1e-12, None)
    return normalize(teacher_oof), normalize(teacher_test)


def reliable_blend_target(teacher_oof):
    log_probs = np.log(np.clip(teacher_oof, 1e-12, None))
    target = T1_WEIGHT * np.exp(log_probs / 1.0) + (1.0 - T1_WEIGHT) * np.exp(log_probs / 3.0)
    return normalize(target)


def apply_bias(probs, bias):
    return normalize(np.asarray(probs, dtype=float) * np.asarray(bias, dtype=float).reshape(1, -1))


def distribution_counts(labels):
    return {str(k): int(v) for k, v in pd.Series(labels).value_counts().reindex(LABELS, fill_value=0).items()}


def train_condition_aware_stage(x_train, x_test, target, y, splits, model_dir, sample_weight=None):
    stage_dir = model_dir / "condition_aware"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)

    student_oofs, student_tests = [], []
    for student_id in range(N_STUDENTS):
        seed = 42 + student_id * 73
        oof = np.zeros((len(y), len(LABELS)), dtype=float)
        test_probs = np.zeros((len(x_test), len(LABELS)), dtype=float)
        for fold_id, (tr_idx, va_idx) in enumerate(splits):
            for class_pos, class_id in enumerate(LABELS):
                reg = LGBMRegressor(random_state=seed, **LGBM_PARAMS)
                sw = sample_weight[tr_idx] if sample_weight is not None else None
                reg.fit(x_train[tr_idx], target[tr_idx, class_pos], sample_weight=sw)
                oof[va_idx, class_pos] = reg.predict(x_train[va_idx])
                test_probs[:, class_pos] += reg.predict(x_test) / len(splits)
                out = stage_dir / f"s{student_id}_fold{fold_id}_class{class_id}.txt"
                out.write_text(reg.booster_.model_to_string(), encoding="utf-8")
            oof[va_idx] = normalize(oof[va_idx])
        student_oofs.append(oof)
        student_tests.append(normalize(test_probs))
    return normalize(np.mean(student_oofs, axis=0)), normalize(np.mean(student_tests, axis=0))


def search_calibrations(student_oof, student_test, teacher_oof, teacher_test, y):
    """Full grid search, return all rows sorted by OOF macro_f1."""
    rows = []
    for student_weight, c0, c1, c2 in product(STUDENT_WEIGHT_GRID, BIAS0_GRID, BIAS1_GRID, BIAS2_GRID):
        bias = np.asarray([c0, c1, c2], dtype=float)
        blend_oof = normalize(student_weight * student_oof + (1.0 - student_weight) * teacher_oof)
        blend_test = normalize(student_weight * student_test + (1.0 - student_weight) * teacher_test)
        calibrated_oof = apply_bias(blend_oof, bias)
        calibrated_test = apply_bias(blend_test, bias)
        pred_oof = calibrated_oof.argmax(axis=1).astype(int)
        pred_test = calibrated_test.argmax(axis=1).astype(int)
        metrics = candidate_metrics(y, calibrated_oof)
        rows.append({
            "student_weight": float(student_weight),
            "bias": bias.tolist(),
            "macro_f1": metrics["macro_f1"],
            "class2_precision": metrics["class2_precision"],
            "class2_recall": metrics["class2_recall"],
            "class2_f1": metrics["class2_f1"],
            "test_label_distribution": distribution_counts(pred_test),
            "oof_label_distribution": metrics["label_distribution"],
            "test_c0": int(pred_test.tolist().count(0)),
            "test_c1": int(pred_test.tolist().count(1)),
            "test_c2": int(pred_test.tolist().count(2)),
        })
    rows.sort(key=lambda r: -r["macro_f1"])
    return rows


def select_candidates(all_rows):
    """
    Select candidates with class1 guard and c2 sweet-spot targeting.

    exact_key5 pushes class1 up ~46 samples; we must not pick a bias that
    gives it back.  Guard: c1 >= 0.98 * best_c1 (at most 2% drop).
    Sweet spot: c2 in 3450-3650 (empirically best platform scores).
    """
    pool = all_rows  # grid is small (1x3x5=15), use all rows
    best_c1 = max(r["test_c1"] for r in pool)

    candidates = {}

    # 1. Conservative: best OOF macro_f1, no constraints
    candidates["c1_conservative"] = max(pool, key=lambda r: r["macro_f1"])

    # 2-5. Target specific c2 levels with class1 guard
    for name, c2_target in [("c2_c3300", 3300), ("c3_c3400", 3400),
                              ("c4_c3500", 3500), ("c5_c3550", 3550)]:
        nearby = [r for r in pool if abs(r["test_c2"] - c2_target) <= 50]
        if not nearby:
            nearby = [r for r in pool if abs(r["test_c2"] - c2_target) <= 100]
        if not nearby:
            nearby = sorted(pool, key=lambda r: abs(r["test_c2"] - c2_target))[:10]
        guarded = [r for r in nearby if r["test_c1"] >= 0.98 * best_c1]
        candidates[name] = max(guarded if guarded else nearby, key=lambda r: r["macro_f1"])

    # 6. Best balanced: class1 guard + c2 in 3450-3650
    balanced = [r for r in pool if 3450 <= r["test_c2"] <= 3650 and r["test_c1"] >= 0.98 * best_c1]
    if not balanced:
        balanced = [r for r in pool if 3450 <= r["test_c2"] <= 3650]
    if balanced:
        candidates["c6_balanced"] = max(balanced, key=lambda r: r["macro_f1"])

    return candidates


def parse_args():
    parser = argparse.ArgumentParser(description="v3.7 align: exact_key5 KD training")
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--t1-weight", type=float, default=0.05,
                        help="Weight for T=1 in soft target (0.05=very soft, 0.35=sharper)")
    parser.add_argument("--sample-weight", action="store_true",
                        help="Enable class1 boost + conflict-key down-weight")
    return parser.parse_args()


def main():
    global T1_WEIGHT
    args = parse_args()
    T1_WEIGHT = args.t1_weight
    root = package_root(__file__)
    model_dir = Path(args.model_dir).resolve() if args.model_dir else default_model_dir(root) / VERSION
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(root) / VERSION
    teacher_dir = model_dir / "teacher_oof"
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = resolve_data_path(root, "train", args.train)
    test_path = resolve_data_path(root, "test", args.test)
    print(f"train: {train_path}")
    print(f"test : {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    y = train["label"].astype(int).to_numpy()
    features = feature_columns(train)
    pair_features = list(combinations(features, 2))

    key_stats = build_key_stats(train, features)
    save_key_stats(key_stats, model_dir)
    maps = stats_maps(key_stats, features)
    x_train = make_model_features(train, features, pair_features, maps)
    x_test = make_model_features(test, features, pair_features, maps)

    # ---- Add exact_key5 features (teacher-student representation alignment) ----
    from reproduce_teacher_npz import FeatureRecipe
    EXACT_KEY_FEATURES = [
        "exact_key_freq_train", "exact_key_freq_test", "exact_key_freq_all",
        "exact_key_logfreq_train", "exact_key_logfreq_test",
    ]
    recipe = FeatureRecipe(train, test)
    x_ek_train = np.column_stack([recipe.build_column(train, n) for n in EXACT_KEY_FEATURES]).astype(np.float32)
    x_ek_test = np.column_stack([recipe.build_column(test, n) for n in EXACT_KEY_FEATURES]).astype(np.float32)
    x_train = np.column_stack([x_train, x_ek_train]).astype(np.float32)
    x_test = np.column_stack([x_test, x_ek_test]).astype(np.float32)
    print(f"exact_key5 alignment: {x_train.shape[1]} dims (student 132 + exact_key 5)")

    splits = exact_group_splits(train, y, features)

    for name in TEACHER_WEIGHTS:
        npz_path = teacher_dir / f"tuned_smoke_group_{name}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Teacher npz missing: {npz_path}\nCopy teacher_oof/ from a previous model dir.")
    teacher_oof, teacher_test = load_teacher_predictions(teacher_dir, len(train), len(test))
    target = reliable_blend_target(teacher_oof)
    teacher_f1 = f1_score(y, teacher_oof.argmax(axis=1), average="macro")
    print(f"teacher_oof_macro_f1={teacher_f1:.6f}")

    # ---- Sample weights: boost class1, down-weight conflict keys ----
    if args.sample_weight:
        sample_w = np.ones(len(y), dtype=np.float32)
        # Per-class weights
        class_w = {0: 0.95, 1: 1.12, 2: 1.00}
        for cls, w in class_w.items():
            sample_w[y == cls] *= w
        # Conflict keys down-weight (pre-computed dict, O(n))
        key_to_lbls = {}
        for i, row in enumerate(train[features].itertuples(index=False, name=None)):
            k = key_tuple(row)
            key_to_lbls.setdefault(k, set()).add(y[i])
        conflict_count = 0
        for i, row in enumerate(train[features].itertuples(index=False, name=None)):
            k = key_tuple(row)
            if len(key_to_lbls.get(k, set())) >= 2:
                sample_w[i] *= 0.70
                conflict_count += 1
        print(f"sample_weight enabled: conflict_keys_downweighted={conflict_count}")
    else:
        sample_w = None

    student_oof, student_test = train_condition_aware_stage(x_train, x_test, target, y, splits, model_dir, sample_weight=sample_w)
    student_f1 = f1_score(y, student_oof.argmax(axis=1), average="macro")
    print(f"condition_aware_oof_macro_f1={student_f1:.6f}")

    np.savez_compressed(model_dir / f"teacher_ensemble_{VERSION}.npz", oof=teacher_oof, test_probs=teacher_test)
    np.savez_compressed(model_dir / f"student_predictions_{VERSION}.npz", oof=student_oof, test_probs=student_test)

    all_rows = search_calibrations(student_oof, student_test, teacher_oof, teacher_test, y)
    candidates = select_candidates(all_rows)

    candidate_configs = {}
    candidate_submission_names = {}
    candidate_predictions = {}

    print(f"\n{'='*80}")
    print(f"Candidates (VERSION={VERSION})")
    print(f"{'='*80}")
    print(f"{'Name':<20} {'sw':<6} {'bias':<22} {'c0':<6} {'c1':<6} {'c2':<6} {'OOF_f1':<10}")
    print("-" * 82)

    for name, cand in candidates.items():
        student_weight = float(cand["student_weight"])
        bias = np.asarray(cand["bias"], dtype=float)
        blend_test = normalize(student_weight * student_test + (1.0 - student_weight) * teacher_test)
        final_probs = apply_bias(blend_test, bias)
        pred = final_probs.argmax(axis=1).astype(int)
        candidate_predictions[name] = pred

        out_name = f"submission_{VERSION}_{name}.csv"
        candidate_submission_names[name] = out_name
        candidate_configs[name] = {"student_weight": student_weight, "bias": cand["bias"]}

        write_submission(test, pred, output_dir, out_name)

        print(f"{name:<20} {student_weight:<6.2f} [%s,%s,%s] {cand['test_c0']:<6} {cand['test_c1']:<6} {cand['test_c2']:<6} {cand['macro_f1']:<10.6f}"
              % (str(cand['bias'][0]), str(cand['bias'][1]), str(cand['bias'][2])))

    # Auto-select: best balanced candidate, fallback to conservative
    selected = "c6_balanced" if "c6_balanced" in candidates else "c1_conservative"
    write_submission(test, candidate_predictions[selected], output_dir, f"submission_{VERSION}.csv")

    metadata = {
        "version": VERSION,
        "method": "condition_aware_kd_v3.7_exact_key5_align",
        "submission_name": f"submission_{VERSION}.csv",
        "selected_candidate": selected,
        "feature_columns": features,
        "pair_features": [list(p) for p in pair_features],
        "labels": LABELS,
        "n_students": N_STUDENTS,
        "n_folds": N_FOLDS,
        "lgbm_params": LGBM_PARAMS,
        "teacher_weights": TEACHER_WEIGHTS,
        "student_weight_grid": STUDENT_WEIGHT_GRID,
        "bias_grid": {"class0": BIAS0_GRID, "class1": BIAS1_GRID, "class2": BIAS2_GRID},
        "candidate_configs": candidate_configs,
        "candidate_submission_names": candidate_submission_names,
        "teacher_oof_macro_f1": float(teacher_f1),
        "condition_aware_oof_macro_f1": float(student_f1),
    }
    save_metadata(model_dir, metadata)

    # Save full search results
    report = {
        "version": VERSION,
        "teacher_oof_macro_f1": float(teacher_f1),
        "condition_aware_oof_macro_f1": float(student_f1),
        "candidates": {name: {k: v for k, v in cand.items()} for name, cand in candidates.items()},
        "top_rows": all_rows[:30],
    }
    (model_dir / f"validation_report_{VERSION}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nSelected: {selected}")
    print(f"Output: {output_dir}")
    print(f"Models: {model_dir}")


if __name__ == "__main__":
    main()
