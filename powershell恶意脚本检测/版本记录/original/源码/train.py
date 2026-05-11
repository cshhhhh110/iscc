from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from common import (
    LABELS,
    MODEL_DIR,
    append_log,
    align_proba,
    build_extra_trees_model,
    build_hgb_model,
    dump_joblib_atomic,
    feature_columns,
    load_train,
    validate_features,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PowerShell malicious script classifier.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--n-estimators", type=int, default=500, help="ExtraTrees tree count.")
    parser.add_argument("--hgb-iter", type=int, default=350, help="HistGradientBoosting iterations.")
    parser.add_argument(
        "--blend-grid",
        type=float,
        nargs="*",
        default=[i / 20 for i in range(21)],
        help="Blend weights to scan for ExtraTrees vs HGB.",
    )
    parser.add_argument("--class1-bias-min", type=float, default=0.85, help="Minimum class 1 bias to scan.")
    parser.add_argument("--class1-bias-max", type=float, default=1.05, help="Maximum class 1 bias to scan.")
    parser.add_argument("--class2-bias-min", type=float, default=1.20, help="Minimum class 2 bias to scan.")
    parser.add_argument("--class2-bias-max", type=float, default=1.60, help="Maximum class 2 bias to scan.")
    parser.add_argument("--class-bias-step", type=float, default=0.025, help="Bias scan step.")
    return parser.parse_args()


def _labels_from_proba(proba: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS, dtype=int)[np.argmax(proba, axis=1)]


def _apply_class_bias(proba: np.ndarray, class_bias: list[float]) -> np.ndarray:
    bias = np.asarray(class_bias, dtype=np.float32)
    if bias.shape != (proba.shape[1],):
        raise ValueError(f"class_bias length {len(bias)} does not match probability columns {proba.shape[1]}")
    if not np.isfinite(bias).all() or (bias <= 0).any():
        raise ValueError(f"class_bias must contain positive finite values, got {bias.tolist()}")
    return proba * bias[None, :]


def _inclusive_grid(start: float, stop: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError(f"grid step must be positive, got {step}")
    if stop < start:
        raise ValueError(f"grid stop must be >= start, got start={start}, stop={stop}")
    values: list[float] = []
    current = float(start)
    epsilon = step / 1000.0
    while current <= stop + epsilon:
        values.append(round(current, 10))
        current += step
    return values


def _is_better_bias(
    score: float,
    class_bias: list[float],
    best_score: float,
    best_bias: list[float] | None,
    tol: float = 1e-12,
) -> bool:
    if score > best_score + tol:
        return True
    if best_bias is None or abs(score - best_score) > tol:
        return False
    if class_bias[2] < best_bias[2] - tol:
        return True
    if abs(class_bias[2] - best_bias[2]) <= tol and class_bias[1] < best_bias[1] - tol:
        return True
    return False


def _search_class_bias(
    proba: np.ndarray,
    y_true: pd.Series,
    class1_grid: list[float],
    class2_grid: list[float],
) -> tuple[list[float], float, np.ndarray, list[dict[str, float]]]:
    best_bias: list[float] | None = None
    best_score = -1.0
    best_pred: np.ndarray | None = None
    scan: list[dict[str, float]] = []

    for class1_bias in class1_grid:
        for class2_bias in class2_grid:
            class_bias = [1.0, float(class1_bias), float(class2_bias)]
            biased_proba = _apply_class_bias(proba, class_bias)
            pred = _labels_from_proba(biased_proba)
            score = float(f1_score(y_true, pred, average="macro"))
            scan.append(
                {
                    "class_0_bias": 1.0,
                    "class_1_bias": float(class1_bias),
                    "class_2_bias": float(class2_bias),
                    "macro_f1": score,
                }
            )
            if _is_better_bias(score, class_bias, best_score, best_bias):
                best_bias = class_bias
                best_score = score
                best_pred = pred

    if best_bias is None or best_pred is None:
        raise RuntimeError("class bias scan produced no valid candidates")
    return best_bias, best_score, best_pred, scan


def main() -> int:
    args = parse_args()
    train_df = load_train()
    features = feature_columns(train_df)
    validate_features(train_df, features)

    X = train_df[features].astype(np.int16)
    y = train_df["label"].astype(int)
    observed = sorted(y.unique().tolist())
    if observed != LABELS:
        raise ValueError(f"expected labels {LABELS}, got {observed}")

    min_class_count = int(y.value_counts().min())
    if args.folds < 2 or args.folds > min_class_count:
        raise ValueError(f"--folds must be between 2 and {min_class_count}, got {args.folds}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_et = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
    oof_hgb = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
    fold_reports = []

    for fold, (train_idx, valid_idx) in enumerate(tqdm(cv.split(X, y), total=args.folds, desc="cv"), start=1):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        et_model = build_extra_trees_model(random_state=args.seed + fold, n_estimators=args.n_estimators)
        et_model.fit(X_train, y_train)
        et_proba = align_proba(et_model, et_model.predict_proba(X_valid))
        oof_et[valid_idx] = et_proba
        et_pred = _labels_from_proba(et_proba)
        et_f1 = f1_score(y_valid, et_pred, average="macro")

        hgb_model = build_hgb_model(random_state=args.seed + fold, max_iter=args.hgb_iter)
        hgb_model.fit(X_train, y_train)
        hgb_proba = align_proba(hgb_model, hgb_model.predict_proba(X_valid))
        oof_hgb[valid_idx] = hgb_proba
        hgb_pred = _labels_from_proba(hgb_proba)
        hgb_f1 = f1_score(y_valid, hgb_pred, average="macro")

        fold_reports.append(
            {
                "fold": fold,
                "extra_trees_macro_f1": float(et_f1),
                "hgb_macro_f1": float(hgb_f1),
                "support": int(len(valid_idx)),
            }
        )

    et_pred = _labels_from_proba(oof_et)
    hgb_pred = _labels_from_proba(oof_hgb)
    et_score = float(f1_score(y, et_pred, average="macro"))
    hgb_score = float(f1_score(y, hgb_pred, average="macro"))

    blend_scan = []
    best_blend_score = -1.0
    best_blend_weight = None
    best_blend_pred = None
    for weight in args.blend_grid:
        weight = float(weight)
        if not 0.0 <= weight <= 1.0:
            continue
        blend_proba = weight * oof_et + (1.0 - weight) * oof_hgb
        blend_pred = _labels_from_proba(blend_proba)
        blend_score = float(f1_score(y, blend_pred, average="macro"))
        blend_scan.append({"weight_et": weight, "weight_hgb": 1.0 - weight, "macro_f1": blend_score})
        if blend_score > best_blend_score:
            best_blend_score = blend_score
            best_blend_weight = weight
            best_blend_pred = blend_pred

    if not blend_scan:
        raise ValueError("blend grid produced no valid weights")

    best_blend_proba = None
    if best_blend_weight is not None:
        best_blend_proba = float(best_blend_weight) * oof_et + (1.0 - float(best_blend_weight)) * oof_hgb

    class1_grid = _inclusive_grid(args.class1_bias_min, args.class1_bias_max, args.class_bias_step)
    class2_grid = _inclusive_grid(args.class2_bias_min, args.class2_bias_max, args.class_bias_step)
    if best_blend_proba is None:
        raise RuntimeError("best blend probability matrix was not computed")
    best_class_bias, best_class_bias_score, best_class_bias_pred, class_bias_scan = _search_class_bias(
        best_blend_proba, y, class1_grid, class2_grid
    )

    candidate_scores = {
        "extra_trees": et_score,
        "hgb": hgb_score,
        "blend": best_class_bias_score,
    }
    selected_model = max(candidate_scores, key=candidate_scores.get)

    selected_class_bias = [1.0, 1.0, 1.0]

    if selected_model == "extra_trees":
        final_model = build_extra_trees_model(random_state=args.seed, n_estimators=args.n_estimators)
        final_model.fit(X, y)
        bundle = {
            "model_type": "single",
            "model_name": "ExtraTreesClassifier",
            "model": final_model,
            "feature_columns": features,
            "labels": LABELS,
            "class_bias": [1.0, 1.0, 1.0],
            "config": {
                "seed": args.seed,
                "n_estimators": args.n_estimators,
                "model": "ExtraTreesClassifier",
            },
        }
        oof_pred = et_pred
    elif selected_model == "hgb":
        final_model = build_hgb_model(random_state=args.seed, max_iter=args.hgb_iter)
        final_model.fit(X, y)
        bundle = {
            "model_type": "single",
            "model_name": "HistGradientBoostingClassifier",
            "model": final_model,
            "feature_columns": features,
            "labels": LABELS,
            "class_bias": [1.0, 1.0, 1.0],
            "config": {
                "seed": args.seed,
                "max_iter": args.hgb_iter,
                "model": "HistGradientBoostingClassifier",
            },
        }
        oof_pred = hgb_pred
    else:
        selected_class_bias = [float(x) for x in best_class_bias]
        final_et = build_extra_trees_model(random_state=args.seed, n_estimators=args.n_estimators)
        final_hgb = build_hgb_model(random_state=args.seed, max_iter=args.hgb_iter)
        final_et.fit(X, y)
        final_hgb.fit(X, y)
        bundle = {
            "model_type": "blend",
            "model_name": "ExtraTreesClassifier+HistGradientBoostingClassifier",
            "models": [final_et, final_hgb],
            "blend_weights": [float(best_blend_weight), float(1.0 - best_blend_weight)],
            "feature_columns": features,
            "labels": LABELS,
            "class_bias": selected_class_bias,
            "config": {
                "seed": args.seed,
                "n_estimators": args.n_estimators,
                "hgb_iter": args.hgb_iter,
                "blend_grid": [float(x) for x in args.blend_grid],
                "selected_weight_et": float(best_blend_weight),
                "class1_bias_grid": class1_grid,
                "class2_bias_grid": class2_grid,
                "selected_class_bias": [float(x) for x in best_class_bias],
            },
        }
        oof_pred = best_class_bias_pred

    if oof_pred is None or len(oof_pred) != len(train_df):
        raise RuntimeError("OOF predictions were not fully populated")

    report = {
        "task": "ISCC PowerShell malicious script detection",
        "selected_model": selected_model,
        "selected_model_name": bundle["model_name"],
        "selected_class_bias": selected_class_bias,
        "random_state": args.seed,
        "n_estimators": args.n_estimators,
        "hgb_iter": args.hgb_iter,
        "folds": args.folds,
        "train_rows": int(len(train_df)),
        "feature_columns": features,
        "label_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "fold_reports": fold_reports,
        "candidate_scores": candidate_scores,
        "blend_scan": blend_scan,
        "blend_best_weight": float(best_blend_weight) if best_blend_weight is not None else None,
        "blend_macro_f1_before_class_bias": float(best_blend_score),
        "class_bias_grid": {
            "class_1": class1_grid,
            "class_2": class2_grid,
            "step": float(args.class_bias_step),
        },
        "class_bias_scan": class_bias_scan,
        "blend_class_bias": [float(x) for x in best_class_bias],
        "blend_class_bias_macro_f1": float(best_class_bias_score),
        "class_bias": selected_class_bias,
        "oof_macro_f1": float(f1_score(y, oof_pred, average="macro")),
        "classification_report": classification_report(y, oof_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, oof_pred, labels=LABELS).tolist(),
    }

    if hasattr(bundle.get("model", None), "feature_importances_"):
        report["feature_importances"] = {
            feature: float(score)
            for feature, score in sorted(
                zip(features, bundle["model"].feature_importances_), key=lambda item: item[1], reverse=True
            )
        }

    bundle["validation_report"] = report
    dump_joblib_atomic(bundle, MODEL_DIR / "model_bundle.joblib")
    write_json(MODEL_DIR / "validation_report.json", report)
    append_log(
        f"trained PowerShell baseline; selected={selected_model}; class_bias={bundle['class_bias']}; "
        f"oof_macro_f1={report['oof_macro_f1']:.6f}; model={MODEL_DIR / 'model_bundle.joblib'}"
    )

    print(f"OOF Macro-F1: {report['oof_macro_f1']:.6f}")
    print(f"Selected model: {selected_model}")
    print(f"Class bias: {bundle['class_bias']}")
    print(f"Saved model: {MODEL_DIR / 'model_bundle.joblib'}")
    print(f"Saved report: {MODEL_DIR / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
