from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from common import (
    LABELS,
    MODEL_DIR,
    align_proba,
    append_log,
    append_result,
    append_total_log,
    ARTIFACT_VERSION,
    apply_class_bias,
    apply_temperature,
    build_catboost_model,
    build_extra_trees_model,
    build_hgb_model,
    build_lgb_model,
    build_xgboost_model,
    dump_joblib_atomic,
    feature_columns,
    load_train,
    load_test,
    pseudo_label_test,
    sample_weighted_indices,
    scan_pseudo_thresholds,
    validate_features,
    write_json,
)
from feature_engineering import (
    apply_smote,
    build_feature_names_with_encoding,
    fit_target_encoder,
    generate_kmeans_features,
    generate_pairwise_interactions,
    transform_kmeans_features,
    transform_target_encoder,
)
from tabular_nn import (
    build_pattern_lookup_bundle,
    build_pattern_soft_targets,
    frame_to_categorical_array,
    infer_cardinalities,
    seed_everything,
    train_torch_fold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PowerShell malicious script classifier.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        choices=["all", "tree", "dcn", "tab_resnet", "pattern_lookup", "catboost", "xgboost", "lgb", "fusion"],
        help="Which candidate families to train.",
    )
    parser.add_argument("--n-estimators", type=int, default=500, help="ExtraTrees tree count.")
    parser.add_argument("--hgb-iter", type=int, default=350, help="HistGradientBoosting iterations.")
    parser.add_argument("--cb-iter", type=int, default=500, help="CatBoost iterations.")
    parser.add_argument("--cb-depth", type=int, default=6, help="CatBoost depth.")
    parser.add_argument("--cb-lr", type=float, default=0.05, help="CatBoost learning rate.")
    parser.add_argument("--xgb-n-est", type=int, default=500, help="XGBoost n_estimators.")
    parser.add_argument("--xgb-max-depth", type=int, default=6, help="XGBoost max_depth.")
    parser.add_argument("--xgb-lr", type=float, default=0.05, help="XGBoost learning rate.")
    parser.add_argument("--lgb-n-est", type=int, default=500, help="LightGBM n_estimators.")
    parser.add_argument("--lgb-max-depth", type=int, default=6, help="LightGBM max_depth.")
    parser.add_argument("--lgb-lr", type=float, default=0.05, help="LightGBM learning rate.")
    parser.add_argument("--dcn-max-epochs", type=int, default=90, help="DeepCrossNetwork max epochs.")
    parser.add_argument("--tab-resnet-max-epochs", type=int, default=90, help="TabResidualNet max epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Training batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Validation batch size.")
    parser.add_argument("--patience", type=int, default=12, help="Early stopping patience.")
    parser.add_argument("--lr-dcn", type=float, default=1.8e-3, help="DeepCrossNetwork learning rate.")
    parser.add_argument("--lr-tab-resnet", type=float, default=1.5e-3, help="TabResidualNet learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--label-smoothing", type=float, default=0.05, help="Soft target smoothing.")
    parser.add_argument("--sample-weight-power", type=float, default=0.5, help="Pattern weight dampening exponent.")
    parser.add_argument("--pattern-alpha", type=float, default=0.5, help="Pattern lookup smoothing alpha.")
    parser.add_argument("--fusion-step", type=float, default=0.2, help="Fusion weight grid step.")
    parser.add_argument(
        "--model-output",
        type=str,
        default=str(MODEL_DIR / f"model_bundle_{ARTIFACT_VERSION}.joblib"),
        help="Model bundle output path.",
    )
    parser.add_argument(
        "--report-output",
        type=str,
        default=str(MODEL_DIR / f"validation_report_{ARTIFACT_VERSION}.json"),
        help="Validation report output path.",
    )
    parser.add_argument("--no-feature-engineering", action="store_true", help="Disable target encoding and interaction features.")
    parser.add_argument("--te-alpha", type=float, default=5.0, help="Target encoding Laplace smoothing alpha.")
    parser.add_argument("--interactions", action="store_true", help="Generate pairwise interaction features (product, ratio, diff).")
    parser.add_argument("--interaction-top-k", type=int, default=60, help="Number of top interaction features to retain.")
    parser.add_argument("--kmeans", action="store_true", help="Add KMeans cluster-distance features.")
    parser.add_argument("--kmeans-clusters", type=int, default=10, help="Number of KMeans clusters.")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE oversampling for tree-based models.")
    parser.add_argument("--smote-k", type=int, default=5, help="SMOTE k_neighbors.")
    parser.add_argument("--loss-type", type=str, default="soft_ce", choices=["focal", "soft_ce"], help="NN loss function.")
    parser.add_argument("--focal-gamma", type=float, default=2.0, help="Focal loss gamma parameter.")
    parser.add_argument("--adversarial", action="store_true", help="Use adversarial validation weights for sampling.")
    parser.add_argument("--pseudo-label", action="store_true", help="Iterative pseudo-label semi-supervised learning.")
    parser.add_argument("--pseudo-threshold", type=float, default=0.95, help="Confidence threshold for pseudo-label selection.")
    parser.add_argument("--no-log", action="store_true", help="Do not append project or total logs.")
    return parser.parse_args()


def resolve_device(spec: str):
    import torch

    spec = spec.lower().strip()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def label_array() -> np.ndarray:
    return np.asarray(LABELS, dtype=int)


def labels_from_proba(proba: np.ndarray) -> np.ndarray:
    return label_array()[np.argmax(proba, axis=1)]


def float_grid(start: float, stop: float, step: float) -> list[float]:
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


def simplex_weights(names: list[str], step: float) -> list[dict[str, float]]:
    if not names:
        return []
    if len(names) == 1:
        return [{names[0]: 1.0}]
    if len(names) == 2:
        return [{names[0]: w, names[1]: 1.0 - w} for w in float_grid(0.0, 1.0, step)]
    n_steps = int(round(1.0 / step))
    if abs(n_steps * step - 1.0) > 1e-8:
        raise ValueError(f"step must divide 1.0 cleanly for simplex grid, got {step}")

    grid: list[dict[str, float]] = []

    def recurse(prefix: dict[str, float], idx: int, remaining_steps: int) -> None:
        if idx == len(names) - 1:
            prefix[names[idx]] = round(remaining_steps * step, 10)
            grid.append(dict(prefix))
            prefix.pop(names[idx], None)
            return
        for weight_steps in range(remaining_steps + 1):
            prefix[names[idx]] = round(weight_steps * step, 10)
            recurse(prefix, idx + 1, remaining_steps - weight_steps)
        prefix.pop(names[idx], None)

    recurse({}, 0, n_steps)
    return grid


def predict_labels_from_calibrated(raw_proba: np.ndarray, temperature: float, class_bias: Sequence[float]) -> np.ndarray:
    scaled = apply_temperature(raw_proba, temperature)
    scaled = apply_class_bias(scaled, class_bias)
    return labels_from_proba(scaled)


def search_calibration(
    raw_proba: np.ndarray,
    y_true: np.ndarray,
    temperature_grid: Sequence[float],
    class1_grid: Sequence[float],
    class2_grid: Sequence[float],
) -> tuple[dict[str, object], list[dict[str, float]]]:
    best_score = -1.0
    best_temperature = None
    best_class_bias = None
    best_pred = None
    scan: list[dict[str, float]] = []

    for temperature in temperature_grid:
        temp_proba = apply_temperature(raw_proba, temperature)
        for class1 in class1_grid:
            for class2 in class2_grid:
                class_bias = [1.0, float(class1), float(class2)]
                pred = labels_from_proba(apply_class_bias(temp_proba, class_bias))
                score = float(f1_score(y_true, pred, average="macro"))
                scan.append(
                    {
                        "temperature": float(temperature),
                        "class_0_bias": 1.0,
                        "class_1_bias": float(class1),
                        "class_2_bias": float(class2),
                        "macro_f1": score,
                    }
                )
                if (
                    score > best_score + 1e-12
                    or (
                        abs(score - best_score) <= 1e-12
                        and best_class_bias is not None
                        and (
                            class_bias[2] < best_class_bias[2] - 1e-12
                            or (
                                abs(class_bias[2] - best_class_bias[2]) <= 1e-12
                                and (
                                    class_bias[1] < best_class_bias[1] - 1e-12
                                    or (
                                        abs(class_bias[1] - best_class_bias[1]) <= 1e-12
                                        and float(temperature) < float(best_temperature) - 1e-12
                                    )
                                )
                            )
                        )
                    )
                ):
                    best_score = score
                    best_temperature = float(temperature)
                    best_class_bias = class_bias
                    best_pred = pred

    if best_temperature is None or best_class_bias is None or best_pred is None:
        raise RuntimeError("calibration search produced no valid result")

    return (
        {
            "temperature": best_temperature,
            "class_bias": best_class_bias,
            "macro_f1": best_score,
            "pred": best_pred,
        },
        scan,
    )


def search_tree_blend(
    oof_et: np.ndarray,
    oof_hgb: np.ndarray,
    y_true: np.ndarray,
    blend_grid: Sequence[float],
    temperature_grid: Sequence[float],
    class1_grid: Sequence[float],
    class2_grid: Sequence[float],
) -> tuple[dict[str, object], np.ndarray, list[dict[str, float]]]:
    best_result = None
    best_raw_proba = None
    scan: list[dict[str, float]] = []
    for weight_et in blend_grid:
        weight_et = float(weight_et)
        if not 0.0 <= weight_et <= 1.0:
            continue
        raw_proba = weight_et * oof_et + (1.0 - weight_et) * oof_hgb
        calibration, calibration_scan = search_calibration(
            raw_proba,
            y_true,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        scan.append(
            {
                "weight_et": weight_et,
                "weight_hgb": 1.0 - weight_et,
                "temperature": float(calibration["temperature"]),
                "class_1_bias": float(calibration["class_bias"][1]),
                "class_2_bias": float(calibration["class_bias"][2]),
                "macro_f1": float(calibration["macro_f1"]),
            }
        )
        if best_result is None or float(calibration["macro_f1"]) > float(best_result["macro_f1"]) + 1e-12:
            best_result = {
                "weight_et": weight_et,
                "weight_hgb": 1.0 - weight_et,
                "temperature": float(calibration["temperature"]),
                "class_bias": calibration["class_bias"],
                "macro_f1": float(calibration["macro_f1"]),
                "calibration_scan": calibration_scan,
            }
            best_raw_proba = raw_proba
        elif best_result is not None and abs(float(calibration["macro_f1"]) - float(best_result["macro_f1"])) <= 1e-12:
            current_bias = [1.0, float(calibration["class_bias"][1]), float(calibration["class_bias"][2])]
            best_bias = [1.0, float(best_result["class_bias"][1]), float(best_result["class_bias"][2])]
            if current_bias[2] < best_bias[2] - 1e-12 or (
                abs(current_bias[2] - best_bias[2]) <= 1e-12 and current_bias[1] < best_bias[1] - 1e-12
            ):
                best_result = {
                    "weight_et": weight_et,
                    "weight_hgb": 1.0 - weight_et,
                    "temperature": float(calibration["temperature"]),
                    "class_bias": calibration["class_bias"],
                    "macro_f1": float(calibration["macro_f1"]),
                    "calibration_scan": calibration_scan,
                }
                best_raw_proba = raw_proba

    if best_result is None or best_raw_proba is None:
        raise RuntimeError("tree blend search produced no valid result")

    return best_result, best_raw_proba, scan


def search_fusion_weights(
    candidate_raw_probs: dict[str, np.ndarray],
    candidate_names: list[str],
    y_true: np.ndarray,
    step: float,
    temperature_grid: Sequence[float],
    class1_grid: Sequence[float],
    class2_grid: Sequence[float],
) -> tuple[dict[str, object], np.ndarray, list[dict[str, float]]]:
    weight_grid = simplex_weights(candidate_names, step)
    best_result = None
    best_raw_proba = None
    scan: list[dict[str, float]] = []
    for weights in tqdm(weight_grid, desc="fusion search", leave=False):
        raw_proba = np.zeros_like(next(iter(candidate_raw_probs.values())), dtype=np.float32)
        for name, weight in weights.items():
            raw_proba += float(weight) * candidate_raw_probs[name]
        calibration, calibration_scan = search_calibration(
            raw_proba,
            y_true,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        row = {f"weight_{name}": float(weight) for name, weight in weights.items()}
        row.update(
            {
                "temperature": float(calibration["temperature"]),
                "class_1_bias": float(calibration["class_bias"][1]),
                "class_2_bias": float(calibration["class_bias"][2]),
                "macro_f1": float(calibration["macro_f1"]),
            }
        )
        scan.append(row)
        if best_result is None or float(calibration["macro_f1"]) > float(best_result["macro_f1"]) + 1e-12:
            best_result = {
                "weights": weights,
                "temperature": float(calibration["temperature"]),
                "class_bias": calibration["class_bias"],
                "macro_f1": float(calibration["macro_f1"]),
                "calibration_scan": calibration_scan,
            }
            best_raw_proba = raw_proba
        elif best_result is not None and abs(float(calibration["macro_f1"]) - float(best_result["macro_f1"])) <= 1e-12:
            current_bias = [1.0, float(calibration["class_bias"][1]), float(calibration["class_bias"][2])]
            best_bias = [1.0, float(best_result["class_bias"][1]), float(best_result["class_bias"][2])]
            if current_bias[2] < best_bias[2] - 1e-12 or (
                abs(current_bias[2] - best_bias[2]) <= 1e-12 and current_bias[1] < best_bias[1] - 1e-12
            ):
                best_result = {
                    "weights": weights,
                    "temperature": float(calibration["temperature"]),
                    "class_bias": calibration["class_bias"],
                    "macro_f1": float(calibration["macro_f1"]),
                    "calibration_scan": calibration_scan,
                }
                best_raw_proba = raw_proba

    if best_result is None or best_raw_proba is None:
        raise RuntimeError("fusion search produced no valid result")

    return best_result, best_raw_proba, scan


def train_sklearn_gbdt_cv(
    model_factory,
    model_name: str,
    model_type_key: str,
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    cv: StratifiedKFold,
    args: argparse.Namespace,
    temperature_grid: Sequence[float],
    class1_grid: Sequence[float],
    class2_grid: Sequence[float],
    train_df: pd.DataFrame | None = None,
    adv_weights: np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    """5-fold CV → calibration → bundle. Uses target encoding + interactions within folds if enabled."""
    use_fe = train_df is not None and not args.no_feature_engineering
    oof = np.zeros((len(y), len(LABELS)), dtype=np.float32)
    fold_reports_list: list[dict[str, object]] = []
    for fold, (train_idx, valid_idx) in enumerate(
        tqdm(list(cv.split(X, y)), total=args.folds, desc=f"{model_type_key} cv"), start=1
    ):
        if use_fe:
            X_tr, X_val, _, _ = _prepare_fold_augmented_features(
                train_df.iloc[train_idx], train_df.iloc[valid_idx], features, y[train_idx], args, seed_offset=fold,
            )
        else:
            X_tr, X_val = X[train_idx].astype(np.float32), X[valid_idx].astype(np.float32)

        y_tr = y[train_idx]
        if args.use_smote and use_fe:
            X_tr, y_tr = apply_smote(X_tr, y_tr, k_neighbors=args.smote_k, random_state=args.seed + fold)
        if adv_weights is not None:
            fold_w = adv_weights[train_idx]
            sampled = sample_weighted_indices(fold_w, n_samples=int(len(X_tr) * 1.5), random_state=args.seed + fold)
            X_tr, y_tr = X_tr[sampled], y_tr[sampled]

        model = model_factory(random_state=args.seed + fold)
        model.fit(X_tr, y_tr)
        oof[valid_idx] = align_proba(model, model.predict_proba(X_val), LABELS)
        pred = labels_from_proba(oof[valid_idx])
        f1 = float(f1_score(y[valid_idx], pred, average="macro"))
        fold_reports_list.append({"fold": fold, "macro_f1": f1, "support": int(len(valid_idx))})

    result, scan = search_calibration(
        oof, y,
        temperature_grid=temperature_grid,
        class1_grid=class1_grid,
        class2_grid=class2_grid,
    )

    if use_fe:
        X_full, te_full, interaction_specs_full, kmeans_model_full = _prepare_full_augmented_features(
            train_df, features, y, args,
        )
    else:
        te_full = None
        interaction_specs_full = None
        kmeans_model_full = None
        X_full = X.astype(np.float32)

    y_full = y
    if args.use_smote and use_fe:
        X_full, y_full = apply_smote(X_full, y, k_neighbors=args.smote_k, random_state=args.seed)

    final_model = model_factory(random_state=args.seed)
    final_model.fit(X_full, y_full)
    raw_bundle: dict[str, object] = {
        "model_type": "sklearn_gbdt",
        "model_name": model_name,
        "model": final_model,
        "feature_columns": features,
        "labels": LABELS,
        "temperature": 1.0,
        "class_bias": [1.0, 1.0, 1.0],
        "target_encoder": te_full,
        "interaction_specs": interaction_specs_full,
        "kmeans_model": kmeans_model_full,
        "config": {},
    }
    selected_bundle = dict(raw_bundle)
    selected_bundle.update(
        {
            "temperature": float(result["temperature"]),
            "class_bias": [float(x) for x in result["class_bias"]],
        }
    )
    return (
        {
            "oof": oof,
            "raw_bundle": raw_bundle,
            "selected_bundle": selected_bundle,
            "score": float(result["macro_f1"]),
            "details": {
                "temperature": float(result["temperature"]),
                "class_bias": [float(x) for x in result["class_bias"]],
                "macro_f1": float(result["macro_f1"]),
                "calibration_scan": scan,
            },
            "fold_reports": fold_reports_list,
        },
        oof,
    )


def resolve_arches(arch: str) -> list[str]:
    arch = arch.lower().strip()
    if arch in {"all", "fusion"}:
        return ["tree", "dcn", "tab_resnet", "pattern_lookup", "catboost", "xgboost", "lgb"]
    if arch in {"tree", "dcn", "tab_resnet", "pattern_lookup", "catboost", "xgboost", "lgb"}:
        return [arch]
    raise ValueError(f"unknown arch: {arch}")


def _prepare_fold_augmented_features(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    features: list[str],
    y_train: np.ndarray,
    args: argparse.Namespace,
    seed_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray, list | None, object | None]:
    """Build augmented feature matrices for a single CV fold.

    Returns (X_train_aug, X_valid_aug, interaction_specs, kmeans_model).
    """
    X_train = frame_to_categorical_array(train_df, features)
    X_valid = frame_to_categorical_array(valid_df, features)

    parts_train: list[np.ndarray] = [X_train.astype(np.float32)]
    parts_valid: list[np.ndarray] = [X_valid.astype(np.float32)]

    if not args.no_feature_engineering:
        te_fold = fit_target_encoder(train_df, features, y_train, alpha=args.te_alpha)
        parts_train.append(transform_target_encoder(train_df, features, te_fold))
        parts_valid.append(transform_target_encoder(valid_df, features, te_fold))

    interaction_specs = None
    if args.interactions:
        inter_train, interaction_specs, _ = generate_pairwise_interactions(
            train_df, features, y_train, top_k=args.interaction_top_k, random_state=args.seed + seed_offset,
        )
        vals_train = train_df[features].to_numpy(dtype=np.float32)
        vals_valid = valid_df[features].to_numpy(dtype=np.float32)
        inter_valid_parts = []
        for i, j, itype in interaction_specs:
            if itype == "prod":
                inter_valid_parts.append((vals_valid[:, i] * vals_valid[:, j]).astype(np.float32))
            elif itype == "diff":
                inter_valid_parts.append((vals_valid[:, i] - vals_valid[:, j]).astype(np.float32))
            elif itype == "ratio":
                inter_valid_parts.append(np.divide(vals_valid[:, i], vals_valid[:, j] + 1.0, dtype=np.float32))
        parts_train.append(inter_train)
        parts_valid.append(np.column_stack(inter_valid_parts).astype(np.float32))

    kmeans_model = None
    if args.kmeans:
        kmeans_train, kmeans_model = generate_kmeans_features(
            train_df, features, n_clusters=args.kmeans_clusters, random_state=args.seed + seed_offset,
        )
        kmeans_valid = transform_kmeans_features(valid_df, features, kmeans_model)
        parts_train.append(kmeans_train)
        parts_valid.append(kmeans_valid)

    X_train_aug = np.column_stack(parts_train).astype(np.float32)
    X_valid_aug = np.column_stack(parts_valid).astype(np.float32)
    return X_train_aug, X_valid_aug, interaction_specs, kmeans_model


def _prepare_full_augmented_features(
    train_df: pd.DataFrame,
    features: list[str],
    y: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object | None, list | None, object | None]:
    """Build augmented feature matrix on full training data for final model.

    Returns (X_full_aug, te_full, interaction_specs_full, kmeans_model_full).
    """
    X = frame_to_categorical_array(train_df, features)
    parts: list[np.ndarray] = [X.astype(np.float32)]

    te_full = None
    if not args.no_feature_engineering:
        te_full = fit_target_encoder(train_df, features, y, alpha=args.te_alpha)
        parts.append(transform_target_encoder(train_df, features, te_full))

    interaction_specs_full = None
    if args.interactions:
        inter_full, interaction_specs_full, _ = generate_pairwise_interactions(
            train_df, features, y, top_k=args.interaction_top_k, random_state=args.seed,
        )
        parts.append(inter_full)

    kmeans_model_full = None
    if args.kmeans:
        kmeans_full, kmeans_model_full = generate_kmeans_features(
            train_df, features, n_clusters=args.kmeans_clusters, random_state=args.seed,
        )
        parts.append(kmeans_full)

    X_full_aug = np.column_stack(parts).astype(np.float32)
    return X_full_aug, te_full, interaction_specs_full, kmeans_model_full


def make_pattern_lookup_bundle(
    lookup: dict[tuple[int, ...], np.ndarray],
    default_proba: np.ndarray,
    features: list[str],
    alpha: float,
) -> dict[str, object]:
    return {
        "model_type": "pattern_lookup",
        "model_name": "PatternLookup",
        "lookup": lookup,
        "default_proba": np.asarray(default_proba, dtype=np.float32),
        "feature_columns": features,
        "labels": LABELS,
        "temperature": 1.0,
        "class_bias": [1.0, 1.0, 1.0],
        "config": {
            "pattern_alpha": float(alpha),
        },
    }


def _run_training_pass(
    train_df: pd.DataFrame,
    features: list[str],
    cardinalities: list[int],
    adv_weights: np.ndarray | None,
    adv_auc: float | None,
    args: argparse.Namespace,
    device: torch.device,
    temperature_grid: Sequence[float],
    class1_grid: Sequence[float],
    class2_grid: Sequence[float],
    tree_blend_grid: Sequence[float],
) -> tuple[dict[str, object], dict[str, object], dict[str, float], float]:
    """Run one full candidate training + fusion + calibration pass.

    Returns (bundle, report, candidate_scores, oof_macro_f1).
    """
    X = frame_to_categorical_array(train_df, features)
    y = train_df["label"].astype(int).to_numpy()
    observed = sorted(np.unique(y).tolist())
    if observed != LABELS:
        raise ValueError(f"expected labels {LABELS}, got {observed}")

    min_class_count = int(np.bincount(y, minlength=len(LABELS)).min())
    if args.folds < 2 or args.folds > min_class_count:
        raise ValueError(f"--folds must be between 2 and {min_class_count}, got {args.folds}")

    pattern_targets, pattern_weights, pattern_stats = build_pattern_soft_targets(
        train_df,
        features,
        alpha=args.pattern_alpha,
        sample_weight_power=args.sample_weight_power,
    )
    feature_unique_counts = {column: int(train_df[column].nunique()) for column in features}
    label_distribution = {int(k): int(v) for k, v in train_df["label"].value_counts().sort_index().items()}

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    train_arches = resolve_arches(args.arch)

    candidate_raw_oof: dict[str, np.ndarray] = {}
    candidate_raw_bundles: dict[str, dict[str, object]] = {}
    candidate_scores: dict[str, float] = {}
    candidate_details: dict[str, dict[str, object]] = {}
    fold_reports: dict[str, list[dict[str, object]]] = {}

    if "tree" in train_arches:
        oof_et = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
        oof_hgb = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
        tree_fold_reports: list[dict[str, object]] = []
        for fold, (train_idx, valid_idx) in enumerate(tqdm(list(cv.split(X, y)), total=args.folds, desc="tree cv"), start=1):
            y_train, y_valid = y[train_idx], y[valid_idx]

            X_train, X_valid, _, _ = _prepare_fold_augmented_features(
                train_df.iloc[train_idx], train_df.iloc[valid_idx], features, y_train, args, seed_offset=fold,
            )
            if args.use_smote:
                X_train, y_train = apply_smote(X_train, y_train, k_neighbors=args.smote_k, random_state=args.seed + fold)
            if adv_weights is not None:
                fold_w = adv_weights[train_idx]
                sampled = sample_weighted_indices(fold_w, n_samples=int(len(X_train) * 1.5), random_state=args.seed + fold)
                X_train, y_train = X_train[sampled], y_train[sampled]

            et_model = build_extra_trees_model(random_state=args.seed + fold, n_estimators=args.n_estimators)
            et_model.fit(X_train, y_train)
            et_proba = et_model.predict_proba(X_valid)
            oof_et[valid_idx] = et_proba[:, np.argsort(et_model.classes_)]
            et_pred = labels_from_proba(oof_et[valid_idx])
            et_f1 = float(f1_score(y_valid, et_pred, average="macro"))

            hgb_model = build_hgb_model(random_state=args.seed + fold, max_iter=args.hgb_iter)
            hgb_model.fit(X_train, y_train)
            hgb_proba = hgb_model.predict_proba(X_valid)
            oof_hgb[valid_idx] = hgb_proba[:, np.argsort(hgb_model.classes_)]
            hgb_pred = labels_from_proba(oof_hgb[valid_idx])
            hgb_f1 = float(f1_score(y_valid, hgb_pred, average="macro"))

            tree_fold_reports.append(
                {
                    "fold": fold,
                    "extra_trees_macro_f1": et_f1,
                    "hgb_macro_f1": hgb_f1,
                    "support": int(len(valid_idx)),
                }
            )

        tree_result, tree_raw_oof, tree_scan = search_tree_blend(
            oof_et,
            oof_hgb,
            y,
            blend_grid=tree_blend_grid,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        X_full, tree_te_full, tree_interaction_specs, tree_kmeans_model = _prepare_full_augmented_features(
            train_df, features, y, args,
        )
        y_full = y
        if args.use_smote:
            X_full, y_full = apply_smote(X_full, y, k_neighbors=args.smote_k, random_state=args.seed)
        final_et = build_extra_trees_model(random_state=args.seed, n_estimators=args.n_estimators)
        final_hgb = build_hgb_model(random_state=args.seed, max_iter=args.hgb_iter)
        final_et.fit(X_full, y_full)
        final_hgb.fit(X_full, y_full)
        tree_raw_bundle = {
            "model_type": "tree_blend",
            "model_name": "ExtraTreesClassifier+HistGradientBoostingClassifier",
            "models": [final_et, final_hgb],
            "blend_weights": [float(tree_result["weight_et"]), float(tree_result["weight_hgb"])],
            "feature_columns": features,
            "labels": LABELS,
            "temperature": 1.0,
            "class_bias": [1.0, 1.0, 1.0],
            "target_encoder": tree_te_full,
            "interaction_specs": tree_interaction_specs,
            "kmeans_model": tree_kmeans_model,
            "config": {
                "seed": args.seed,
                "n_estimators": args.n_estimators,
                "hgb_iter": args.hgb_iter,
                "blend_weights": [float(tree_result["weight_et"]), float(tree_result["weight_hgb"])],
                "interactions": args.interactions,
                "kmeans": args.kmeans,
                "use_smote": args.use_smote,
            },
        }
        tree_selected_bundle = dict(tree_raw_bundle)
        tree_selected_bundle.update(
            {
                "temperature": float(tree_result["temperature"]),
                "class_bias": [float(x) for x in tree_result["class_bias"]],
            }
        )
        candidate_raw_oof["tree"] = tree_raw_oof
        candidate_raw_bundles["tree"] = tree_raw_bundle
        candidate_scores["tree"] = float(tree_result["macro_f1"])
        candidate_details["tree"] = {
            "blend_weights": [float(tree_result["weight_et"]), float(tree_result["weight_hgb"])],
            "temperature": float(tree_result["temperature"]),
            "class_bias": [float(x) for x in tree_result["class_bias"]],
            "macro_f1": float(tree_result["macro_f1"]),
            "tree_scan": tree_scan,
            "calibration_scan": tree_result["calibration_scan"],
        }
        fold_reports["tree"] = tree_fold_reports
        candidate_raw_bundles["tree_selected"] = tree_selected_bundle

    if "dcn" in train_arches:
        dcn_oof = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
        dcn_fold_models: list[dict[str, object]] = []
        dcn_fold_reports: list[dict[str, object]] = []
        for fold, (train_idx, valid_idx) in enumerate(tqdm(list(cv.split(X, y)), total=args.folds, desc="dcn cv"), start=1):
            train_frame = train_df.iloc[train_idx]
            valid_frame = train_df.iloc[valid_idx]
            fold_targets, fold_weights, fold_stats = build_pattern_soft_targets(
                train_frame,
                features,
                alpha=args.pattern_alpha,
                sample_weight_power=args.sample_weight_power,
            )
            fold_result = train_torch_fold(
                arch="dcn",
                cardinalities=cardinalities,
                train_features=X[train_idx],
                train_targets=fold_targets,
                train_labels=y[train_idx],
                train_weights=fold_weights,
                valid_features=X[valid_idx],
                valid_labels=y[valid_idx],
                config={
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "max_epochs": args.dcn_max_epochs,
                    "patience": args.patience,
                    "lr": args.lr_dcn,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "max_grad_norm": 1.0,
                    "amp": True,
                    "embed_dim": 8,
                    "cross_layers": 3,
                    "deep_dims": (128, 64),
                    "dropout": 0.12,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                },
                device=device,
                seed=args.seed + 1000 + fold,
                desc=f"dcn fold {fold}",
            )
            dcn_oof[valid_idx] = fold_result["val_proba"]
            dcn_fold_models.append(
                {
                    "arch": "dcn",
                    "config": fold_result["config"],
                    "cardinalities": fold_result["cardinalities"],
                    "num_classes": fold_result["num_classes"],
                    "state_dict": fold_result["state_dict"],
                    "eval_batch_size": fold_result["eval_batch_size"],
                }
            )
            dcn_fold_reports.append(
                {
                    "fold": fold,
                    "best_epoch": int(fold_result["best_epoch"]),
                    "macro_f1": float(fold_result["best_macro_f1"]),
                    "val_loss": float(fold_result["best_val_loss"]),
                    "support": int(len(valid_idx)),
                    "pattern_stats": fold_stats,
                }
            )
        dcn_result, dcn_scan = search_calibration(
            dcn_oof,
            y,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        dcn_raw_bundle = {
            "model_type": "torch_ensemble",
            "model_name": "DeepCrossNetwork",
            "fold_models": dcn_fold_models,
            "feature_columns": features,
            "labels": LABELS,
            "temperature": 1.0,
            "class_bias": [1.0, 1.0, 1.0],
            "config": {
                "arch": "dcn",
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "max_epochs": args.dcn_max_epochs,
                "patience": args.patience,
                "lr": args.lr_dcn,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "embed_dim": 8,
                "cross_layers": 3,
                "deep_dims": (128, 64),
                "dropout": 0.12,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
            },
        }
        dcn_selected_bundle = dict(dcn_raw_bundle)
        dcn_selected_bundle.update(
            {
                "temperature": float(dcn_result["temperature"]),
                "class_bias": [float(x) for x in dcn_result["class_bias"]],
            }
        )
        candidate_raw_oof["dcn"] = dcn_oof
        candidate_raw_bundles["dcn"] = dcn_raw_bundle
        candidate_scores["dcn"] = float(dcn_result["macro_f1"])
        candidate_details["dcn"] = {
            "temperature": float(dcn_result["temperature"]),
            "class_bias": [float(x) for x in dcn_result["class_bias"]],
            "macro_f1": float(dcn_result["macro_f1"]),
            "calibration_scan": dcn_scan,
        }
        fold_reports["dcn"] = dcn_fold_reports
        candidate_raw_bundles["dcn_selected"] = dcn_selected_bundle

    if "tab_resnet" in train_arches:
        tab_resnet_oof = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
        tab_resnet_fold_models: list[dict[str, object]] = []
        tab_resnet_fold_reports: list[dict[str, object]] = []
        for fold, (train_idx, valid_idx) in enumerate(tqdm(list(cv.split(X, y)), total=args.folds, desc="tab_resnet cv"), start=1):
            train_frame = train_df.iloc[train_idx]
            valid_frame = train_df.iloc[valid_idx]
            fold_targets, fold_weights, fold_stats = build_pattern_soft_targets(
                train_frame,
                features,
                alpha=args.pattern_alpha,
                sample_weight_power=args.sample_weight_power,
            )
            fold_result = train_torch_fold(
                arch="tab_resnet",
                cardinalities=cardinalities,
                train_features=X[train_idx],
                train_targets=fold_targets,
                train_labels=y[train_idx],
                train_weights=fold_weights,
                valid_features=X[valid_idx],
                valid_labels=y[valid_idx],
                config={
                    "batch_size": args.batch_size,
                    "eval_batch_size": args.eval_batch_size,
                    "max_epochs": args.tab_resnet_max_epochs,
                    "patience": args.patience,
                    "lr": args.lr_tab_resnet,
                    "weight_decay": args.weight_decay,
                    "label_smoothing": args.label_smoothing,
                    "max_grad_norm": 1.0,
                    "amp": True,
                    "embed_dim": 8,
                    "width": 192,
                    "num_blocks": 4,
                    "expansion": 2,
                    "dropout": 0.12,
                    "loss_type": args.loss_type,
                    "focal_gamma": args.focal_gamma,
                },
                device=device,
                seed=args.seed + 2000 + fold,
                desc=f"tab_resnet fold {fold}",
            )
            tab_resnet_oof[valid_idx] = fold_result["val_proba"]
            tab_resnet_fold_models.append(
                {
                    "arch": "tab_resnet",
                    "config": fold_result["config"],
                    "cardinalities": fold_result["cardinalities"],
                    "num_classes": fold_result["num_classes"],
                    "state_dict": fold_result["state_dict"],
                    "eval_batch_size": fold_result["eval_batch_size"],
                }
            )
            tab_resnet_fold_reports.append(
                {
                    "fold": fold,
                    "best_epoch": int(fold_result["best_epoch"]),
                    "macro_f1": float(fold_result["best_macro_f1"]),
                    "val_loss": float(fold_result["best_val_loss"]),
                    "support": int(len(valid_idx)),
                    "pattern_stats": fold_stats,
                }
            )
        tab_resnet_result, tab_resnet_scan = search_calibration(
            tab_resnet_oof,
            y,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        tab_resnet_raw_bundle = {
            "model_type": "torch_ensemble",
            "model_name": "TabResidualNet",
            "fold_models": tab_resnet_fold_models,
            "feature_columns": features,
            "labels": LABELS,
            "temperature": 1.0,
            "class_bias": [1.0, 1.0, 1.0],
            "config": {
                "arch": "tab_resnet",
                "batch_size": args.batch_size,
                "eval_batch_size": args.eval_batch_size,
                "max_epochs": args.tab_resnet_max_epochs,
                "patience": args.patience,
                "lr": args.lr_tab_resnet,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "embed_dim": 8,
                "width": 192,
                "num_blocks": 4,
                "expansion": 2,
                "dropout": 0.12,
                "loss_type": args.loss_type,
                "focal_gamma": args.focal_gamma,
            },
        }
        tab_resnet_selected_bundle = dict(tab_resnet_raw_bundle)
        tab_resnet_selected_bundle.update(
            {
                "temperature": float(tab_resnet_result["temperature"]),
                "class_bias": [float(x) for x in tab_resnet_result["class_bias"]],
            }
        )
        candidate_raw_oof["tab_resnet"] = tab_resnet_oof
        candidate_raw_bundles["tab_resnet"] = tab_resnet_raw_bundle
        candidate_scores["tab_resnet"] = float(tab_resnet_result["macro_f1"])
        candidate_details["tab_resnet"] = {
            "temperature": float(tab_resnet_result["temperature"]),
            "class_bias": [float(x) for x in tab_resnet_result["class_bias"]],
            "macro_f1": float(tab_resnet_result["macro_f1"]),
            "calibration_scan": tab_resnet_scan,
        }
        fold_reports["tab_resnet"] = tab_resnet_fold_reports
        candidate_raw_bundles["tab_resnet_selected"] = tab_resnet_selected_bundle

    if "pattern_lookup" in train_arches:
        pattern_oof = np.zeros((len(train_df), len(LABELS)), dtype=np.float32)
        pattern_fold_reports: list[dict[str, object]] = []
        for fold, (train_idx, valid_idx) in enumerate(tqdm(list(cv.split(X, y)), total=args.folds, desc="pattern cv"), start=1):
            train_frame = train_df.iloc[train_idx]
            lookup, default_proba, fold_stats = build_pattern_lookup_bundle(
                train_frame,
                features,
                alpha=args.pattern_alpha,
            )
            fold_bundle = make_pattern_lookup_bundle(
                lookup=lookup,
                default_proba=default_proba,
                features=features,
                alpha=args.pattern_alpha,
            )
            pattern_oof[valid_idx] = predict_bundle_proba(fold_bundle, X[valid_idx])
            pattern_pred = labels_from_proba(pattern_oof[valid_idx])
            pattern_f1 = float(f1_score(y[valid_idx], pattern_pred, average="macro"))
            pattern_fold_reports.append(
                {
                    "fold": fold,
                    "macro_f1": pattern_f1,
                    "support": int(len(valid_idx)),
                    "pattern_stats": fold_stats,
                }
            )
        pattern_result, pattern_scan = search_calibration(
            pattern_oof,
            y,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        pattern_lookup, pattern_default_proba, pattern_stats_full = build_pattern_lookup_bundle(
            train_df,
            features,
            alpha=args.pattern_alpha,
        )
        pattern_raw_bundle = make_pattern_lookup_bundle(
            lookup=pattern_lookup,
            default_proba=pattern_default_proba,
            features=features,
            alpha=args.pattern_alpha,
        )
        pattern_selected_bundle = dict(pattern_raw_bundle)
        pattern_selected_bundle.update(
            {
                "temperature": float(pattern_result["temperature"]),
                "class_bias": [float(x) for x in pattern_result["class_bias"]],
            }
        )
        candidate_raw_oof["pattern_lookup"] = pattern_oof
        candidate_raw_bundles["pattern_lookup"] = pattern_raw_bundle
        candidate_scores["pattern_lookup"] = float(pattern_result["macro_f1"])
        candidate_details["pattern_lookup"] = {
            "temperature": float(pattern_result["temperature"]),
            "class_bias": [float(x) for x in pattern_result["class_bias"]],
            "macro_f1": float(pattern_result["macro_f1"]),
            "pattern_stats": pattern_stats_full,
            "calibration_scan": pattern_scan,
        }
        fold_reports["pattern_lookup"] = pattern_fold_reports
        candidate_raw_bundles["pattern_lookup_selected"] = pattern_selected_bundle

    if "catboost" in train_arches:
        cb_result, _ = train_sklearn_gbdt_cv(
            model_factory=lambda random_state: build_catboost_model(
                random_state=random_state,
                iterations=args.cb_iter,
                depth=args.cb_depth,
                lr=args.cb_lr,
            ),
            model_name="CatBoostClassifier",
            model_type_key="catboost",
            X=X, y=y, features=features, cv=cv, args=args,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
            train_df=train_df,
            adv_weights=adv_weights,
        )
        candidate_raw_oof["catboost"] = cb_result["oof"]
        candidate_raw_bundles["catboost"] = cb_result["raw_bundle"]
        candidate_scores["catboost"] = cb_result["score"]
        candidate_details["catboost"] = cb_result["details"]
        fold_reports["catboost"] = cb_result["fold_reports"]
        candidate_raw_bundles["catboost_selected"] = cb_result["selected_bundle"]

    if "xgboost" in train_arches:
        xgb_result, _ = train_sklearn_gbdt_cv(
            model_factory=lambda random_state: build_xgboost_model(
                random_state=random_state,
                n_estimators=args.xgb_n_est,
                max_depth=args.xgb_max_depth,
                lr=args.xgb_lr,
            ),
            model_name="XGBClassifier",
            model_type_key="xgboost",
            X=X, y=y, features=features, cv=cv, args=args,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
            train_df=train_df,
            adv_weights=adv_weights,
        )
        candidate_raw_oof["xgboost"] = xgb_result["oof"]
        candidate_raw_bundles["xgboost"] = xgb_result["raw_bundle"]
        candidate_scores["xgboost"] = xgb_result["score"]
        candidate_details["xgboost"] = xgb_result["details"]
        fold_reports["xgboost"] = xgb_result["fold_reports"]
        candidate_raw_bundles["xgboost_selected"] = xgb_result["selected_bundle"]

    if "lgb" in train_arches:
        lgb_result, _ = train_sklearn_gbdt_cv(
            model_factory=lambda random_state: build_lgb_model(
                random_state=random_state,
                n_estimators=args.lgb_n_est,
                max_depth=args.lgb_max_depth,
                lr=args.lgb_lr,
            ),
            model_name="LGBMClassifier",
            model_type_key="lgb",
            X=X, y=y, features=features, cv=cv, args=args,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
            train_df=train_df,
            adv_weights=adv_weights,
        )
        candidate_raw_oof["lgb"] = lgb_result["oof"]
        candidate_raw_bundles["lgb"] = lgb_result["raw_bundle"]
        candidate_scores["lgb"] = lgb_result["score"]
        candidate_details["lgb"] = lgb_result["details"]
        fold_reports["lgb"] = lgb_result["fold_reports"]
        candidate_raw_bundles["lgb_selected"] = lgb_result["selected_bundle"]

    fusion_result = None
    fusion_raw_oof = None
    fusion_scan = []
    if len(candidate_raw_oof) >= 2:
        fusion_result, fusion_raw_oof, fusion_scan = search_fusion_weights(
            candidate_raw_probs={name: candidate_raw_oof[name] for name in candidate_raw_oof},
            candidate_names=list(candidate_raw_oof.keys()),
            y_true=y,
            step=args.fusion_step,
            temperature_grid=temperature_grid,
            class1_grid=class1_grid,
            class2_grid=class2_grid,
        )
        candidate_scores["fusion"] = float(fusion_result["macro_f1"])
        candidate_details["fusion"] = {
            "weights": fusion_result["weights"],
            "temperature": float(fusion_result["temperature"]),
            "class_bias": [float(x) for x in fusion_result["class_bias"]],
            "macro_f1": float(fusion_result["macro_f1"]),
            "calibration_scan": fusion_result["calibration_scan"],
            "fusion_scan": fusion_scan,
        }

    if not candidate_scores:
        raise RuntimeError("no candidate models were trained")

    selected_model_name = max(candidate_scores, key=candidate_scores.get)

    if selected_model_name == "fusion":
        fusion_component_names = list(candidate_raw_oof.keys())
        selected_model_bundle = {
            "model_type": "fusion",
            "model_name": "Fusion(" + "+".join(candidate_raw_bundles[name]["model_name"] for name in fusion_component_names) + ")",
            "components": {
                name: candidate_raw_bundles[name]
                for name in fusion_component_names
            },
            "component_weights": {name: float(weight) for name, weight in fusion_result["weights"].items()},
            "feature_columns": features,
            "labels": LABELS,
            "temperature": float(fusion_result["temperature"]),
            "class_bias": [float(x) for x in fusion_result["class_bias"]],
            "config": {
                "fusion_step": args.fusion_step,
                "component_weights": {name: float(weight) for name, weight in fusion_result["weights"].items()},
            },
        }
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(fusion_raw_oof, float(fusion_result["temperature"])),
                fusion_result["class_bias"],
            )
        )
    elif selected_model_name == "tree":
        selected_model_bundle = candidate_raw_bundles["tree_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["tree"], candidate_details["tree"]["temperature"]),
                candidate_details["tree"]["class_bias"],
            )
        )
    elif selected_model_name == "pattern_lookup":
        selected_model_bundle = candidate_raw_bundles["pattern_lookup_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["pattern_lookup"], candidate_details["pattern_lookup"]["temperature"]),
                candidate_details["pattern_lookup"]["class_bias"],
            )
        )
    elif selected_model_name == "dcn":
        selected_model_bundle = candidate_raw_bundles["dcn_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["dcn"], candidate_details["dcn"]["temperature"]),
                candidate_details["dcn"]["class_bias"],
            )
        )
    elif selected_model_name == "tab_resnet":
        selected_model_bundle = candidate_raw_bundles["tab_resnet_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["tab_resnet"], candidate_details["tab_resnet"]["temperature"]),
                candidate_details["tab_resnet"]["class_bias"],
            )
        )
    elif selected_model_name == "catboost":
        selected_model_bundle = candidate_raw_bundles["catboost_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["catboost"], candidate_details["catboost"]["temperature"]),
                candidate_details["catboost"]["class_bias"],
            )
        )
    elif selected_model_name == "xgboost":
        selected_model_bundle = candidate_raw_bundles["xgboost_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["xgboost"], candidate_details["xgboost"]["temperature"]),
                candidate_details["xgboost"]["class_bias"],
            )
        )
    elif selected_model_name == "lgb":
        selected_model_bundle = candidate_raw_bundles["lgb_selected"]
        final_oof_pred = labels_from_proba(
            apply_class_bias(
                apply_temperature(candidate_raw_oof["lgb"], candidate_details["lgb"]["temperature"]),
                candidate_details["lgb"]["class_bias"],
            )
        )
    else:
        raise RuntimeError(f"unknown selected model {selected_model_name}")

    if len(final_oof_pred) != len(y):
        raise RuntimeError("OOF predictions were not fully populated")

    report = {
        "task": "ISCC PowerShell malicious script detection",
        "version": ARTIFACT_VERSION,
        "selected_model": selected_model_name,
        "selected_model_name": selected_model_bundle["model_name"],
        "selected_model_bundle_type": selected_model_bundle["model_type"],
        "random_state": args.seed,
        "device": str(device),
        "folds": args.folds,
        "train_rows": int(len(train_df)),
        "feature_columns": features,
        "feature_unique_counts": feature_unique_counts,
        "cardinalities": cardinalities,
        "label_distribution": label_distribution,
        "pattern_stats": pattern_stats,
        "candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "fold_reports": fold_reports,
        "oof_macro_f1": float(f1_score(y, final_oof_pred, average="macro")),
        "classification_report": classification_report(y, final_oof_pred, labels=LABELS, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, final_oof_pred, labels=LABELS).tolist(),
    }

    bundle = {
        "version": ARTIFACT_VERSION,
        "task": "ISCC PowerShell malicious script detection",
        "selected_model": selected_model_name,
        "selected_model_name": selected_model_bundle["model_name"],
        "selected_model_bundle": selected_model_bundle,
        "feature_columns": features,
        "cardinalities": cardinalities,
        "labels": LABELS,
        "candidate_scores": candidate_scores,
        "candidate_details": candidate_details,
        "fold_reports": fold_reports,
        "pattern_stats": pattern_stats,
        "validation_report": report,
    }

    oof_macro_f1 = float(report["oof_macro_f1"])
    return bundle, report, candidate_scores, oof_macro_f1


def main() -> int:
    t0 = time.perf_counter()
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_df = load_train()
    features = feature_columns(train_df)
    validate_features(train_df, features)
    cardinalities = infer_cardinalities(train_df, features)

    temperature_grid = float_grid(0.80, 1.20, 0.05)
    class1_grid = float_grid(0.85, 1.05, 0.025)
    class2_grid = float_grid(1.20, 1.60, 0.025)
    tree_blend_grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 21)]

    # adversary removed (v1.7 proved it hurts); build teacher once
    bundle_t, report_t, scores_t, oof_t = _run_training_pass(
        train_df, features, cardinalities, None, None,
        args, device, temperature_grid, class1_grid, class2_grid, tree_blend_grid,
    )

    pseudo_log = None
    if args.pseudo_label:
        test_df_pl = load_test()
        label_dist = {int(k): int(v) for k, v in train_df["label"].value_counts().to_dict().items()}
        scan = scan_pseudo_thresholds(
            bundle_t, test_df_pl, features, label_dist,
            thresholds=[0.99, 0.97, 0.95, 0.92, 0.90, 0.85],
            cap_ratio=0.20, min_samples=200, device=device,
        )
        if scan:
            best = scan[0]
            pseudo_df, _ = pseudo_label_test(
                bundle_t, test_df_pl, features,
                threshold=best["threshold"],
                max_per_class=best.get("max_per_class"),
                device=device,
            )
            if len(pseudo_df) >= 200:
                train_df_ext = pd.concat([train_df, pseudo_df], ignore_index=True)
                cardinalities_ext = infer_cardinalities(train_df_ext, features)
                bundle_s, report_s, scores_s, oof_s = _run_training_pass(
                    train_df_ext, features, cardinalities_ext, None, None,
                    args, device, temperature_grid, class1_grid, class2_grid, tree_blend_grid,
                )
                pseudo_log = {
                    "teacher_oof": round(float(oof_t), 6),
                    "student_oof": round(float(oof_s), 6),
                    "threshold_scan": scan[:5],
                    "selected_threshold": best["threshold"],
                    "pseudo_samples": int(len(pseudo_df)),
                    "pseudo_per_class": {int(k): int(v) for k, v in pseudo_df["label"].value_counts().to_dict().items()},
                }
                bundle, report, candidate_scores, oof_f1 = bundle_s, report_s, scores_s, oof_s
            else:
                bundle, report, candidate_scores, oof_f1 = bundle_t, report_t, scores_t, oof_t
        else:
            bundle, report, candidate_scores, oof_f1 = bundle_t, report_t, scores_t, oof_t
        del test_df_pl
    else:
        bundle, report, candidate_scores, oof_f1 = bundle_t, report_t, scores_t, oof_t

    if pseudo_log:
        report["pseudo_label"] = pseudo_log

    # save
    model_output = Path(args.model_output)
    report_output = Path(args.report_output)
    dump_joblib_atomic(bundle, model_output)
    write_json(report_output, report)

    train_seconds = time.perf_counter() - t0

    if not args.no_log:
        append_log(
            f"trained {ARTIFACT_VERSION} PowerShell models; selected={bundle['selected_model']}; "
            f"oof_macro_f1={report['oof_macro_f1']:.6f}; device={device}; bundle={model_output}"
        )
        append_total_log(
            f"powershell CMD {ARTIFACT_VERSION}; selected={bundle['selected_model']}; oof_macro_f1={report['oof_macro_f1']:.6f}"
        )

    append_result(
        version=ARTIFACT_VERSION,
        selected_model=bundle["selected_model"],
        oof_macro_f1=report["oof_macro_f1"],
        candidate_scores=candidate_scores,
        device=str(device),
        train_seconds=train_seconds,
        interactions=args.interactions,
        kmeans=args.kmeans,
        use_smote=args.use_smote,
        loss_type=args.loss_type,
        folds=args.folds,
    )

    print(f"Device: {device}")
    print(f"Selected model: {bundle['selected_model']}")
    print(f"OOF Macro-F1: {report['oof_macro_f1']:.6f}")
    if pseudo_log:
        print(f"Teacher OOF: {pseudo_log['teacher_oof']}")
        print(f"Student OOF: {pseudo_log['student_oof']}")
        print(f"Pseudo samples: {pseudo_log['pseudo_samples']} @ th={pseudo_log['selected_threshold']}")
    print(f"Train time: {train_seconds / 60:.1f} min")
    print(f"Saved model: {model_output}")
    print(f"Saved report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
