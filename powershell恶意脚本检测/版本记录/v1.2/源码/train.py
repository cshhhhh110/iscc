from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from common import (
    LABELS,
    MODEL_DIR,
    append_log,
    append_total_log,
    ARTIFACT_VERSION,
    apply_class_bias,
    apply_temperature,
    build_extra_trees_model,
    build_hgb_model,
    dump_joblib_atomic,
    feature_columns,
    load_train,
    validate_features,
    write_json,
)
from tabular_nn import build_pattern_soft_targets, frame_to_categorical_array, infer_cardinalities, seed_everything, train_torch_fold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PowerShell malicious script classifier.")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Training device.")
    parser.add_argument(
        "--arch",
        type=str,
        default="all",
        choices=["all", "tree", "dcn", "tab_resnet", "fusion"],
        help="Which candidate families to train.",
    )
    parser.add_argument("--n-estimators", type=int, default=500, help="ExtraTrees tree count.")
    parser.add_argument("--hgb-iter", type=int, default=350, help="HistGradientBoosting iterations.")
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
    if len(names) == 3:
        grid: list[dict[str, float]] = []
        n_steps = int(round(1.0 / step))
        for a in range(n_steps + 1):
            for b in range(n_steps + 1 - a):
                c = n_steps - a - b
                weights = {
                    names[0]: round(a * step, 10),
                    names[1]: round(b * step, 10),
                    names[2]: round(c * step, 10),
                }
                grid.append(weights)
        return grid
    raise ValueError("simplex grid only supports up to 3 names")


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
    for weights in weight_grid:
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


def resolve_arches(arch: str) -> list[str]:
    arch = arch.lower().strip()
    if arch in {"all", "fusion"}:
        return ["tree", "dcn", "tab_resnet"]
    if arch in {"tree", "dcn", "tab_resnet"}:
        return [arch]
    raise ValueError(f"unknown arch: {arch}")


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_df = load_train()
    features = feature_columns(train_df)
    validate_features(train_df, features)

    X = frame_to_categorical_array(train_df, features)
    y = train_df["label"].astype(int).to_numpy()
    observed = sorted(np.unique(y).tolist())
    if observed != LABELS:
        raise ValueError(f"expected labels {LABELS}, got {observed}")

    cardinalities = infer_cardinalities(train_df, features)
    min_class_count = int(np.bincount(y, minlength=len(LABELS)).min())
    if args.folds < 2 or args.folds > min_class_count:
        raise ValueError(f"--folds must be between 2 and {min_class_count}, got {args.folds}")

    temperature_grid = float_grid(0.80, 1.20, 0.05)
    class1_grid = float_grid(0.85, 1.05, 0.025)
    class2_grid = float_grid(1.20, 1.60, 0.025)
    tree_blend_grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 21)]

    pattern_targets, pattern_weights, pattern_stats = build_pattern_soft_targets(
        train_df,
        features,
        alpha=0.5,
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
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

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
        final_et = build_extra_trees_model(random_state=args.seed, n_estimators=args.n_estimators)
        final_hgb = build_hgb_model(random_state=args.seed, max_iter=args.hgb_iter)
        final_et.fit(X, y)
        final_hgb.fit(X, y)
        tree_raw_bundle = {
            "model_type": "tree_blend",
            "model_name": "ExtraTreesClassifier+HistGradientBoostingClassifier",
            "models": [final_et, final_hgb],
            "blend_weights": [float(tree_result["weight_et"]), float(tree_result["weight_hgb"])],
            "feature_columns": features,
            "labels": LABELS,
            "temperature": 1.0,
            "class_bias": [1.0, 1.0, 1.0],
            "config": {
                "seed": args.seed,
                "n_estimators": args.n_estimators,
                "hgb_iter": args.hgb_iter,
                "blend_weights": [float(tree_result["weight_et"]), float(tree_result["weight_hgb"])],
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
                alpha=0.5,
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
                alpha=0.5,
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
        selected_model_bundle = {
            "model_type": "fusion",
            "model_name": "Fusion(Tree+DCN+TabResidualNet)",
            "components": {
                name: candidate_raw_bundles[f"{name}"]
                for name in candidate_raw_oof.keys()
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
    else:
        raise RuntimeError(f"unknown selected model {selected_model_name}")

    if len(final_oof_pred) != len(y):
        raise RuntimeError("OOF predictions were not fully populated")

    report = {
        "task": "ISCC PowerShell malicious script detection",
        "version": "v1.2",
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
        "version": "v1.2",
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

    model_output = Path(args.model_output)
    report_output = Path(args.report_output)
    dump_joblib_atomic(bundle, model_output)
    write_json(report_output, report)

    if not args.no_log:
        append_log(
            f"trained v1.2 PowerShell models; selected={selected_model_name}; "
            f"oof_macro_f1={report['oof_macro_f1']:.6f}; device={device}; bundle={model_output}"
        )
        append_total_log(
            f"powershell恶意脚本检测 v1.2; selected={selected_model_name}; oof_macro_f1={report['oof_macro_f1']:.6f}"
        )

    print(f"Device: {device}")
    print(f"Selected model: {selected_model_name}")
    print(f"OOF Macro-F1: {report['oof_macro_f1']:.6f}")
    print(f"Saved model: {model_output}")
    print(f"Saved report: {report_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
