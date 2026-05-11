from __future__ import annotations

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PROJECT_ROOT / "data_train.csv"
TEST_PATH = PROJECT_ROOT / "data_test.csv"
MODEL_DIR = PROJECT_ROOT / "模型"
RESULT_DIR = PROJECT_ROOT / "提交结果"
PROJECT_LOG_PATH = PROJECT_ROOT / "ACTION_LOG.md"
TOTAL_LOG_PATH = PROJECT_ROOT.parent / "ACTION_LOG.md"
RESULTS_CSV = PROJECT_ROOT / "results.csv"

LABELS = [0, 1, 2]
TARGET_COLUMN = "label"
ID_COLUMN = "name"
FEATURE_COUNT = 15
ARTIFACT_VERSION = "v1.8"


def append_log(message: str) -> None:
    PROJECT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with PROJECT_LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"- {timestamp} {message}\n")


def append_total_log(message: str) -> None:
    try:
        TOTAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with TOTAL_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(f"- {timestamp} {message}\n")
    except OSError:
        pass


def append_result(
    version: str,
    selected_model: str,
    oof_macro_f1: float,
    candidate_scores: dict[str, float],
    device: str,
    train_seconds: float,
    interactions: bool = False,
    kmeans: bool = False,
    use_smote: bool = False,
    loss_type: str = "soft_ce",
    folds: int = 5,
) -> None:
    """Append a compact result row to results.csv for version comparison."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "version": version,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "device": device,
        "folds": folds,
        "interactions": interactions,
        "kmeans": kmeans,
        "smote": use_smote,
        "loss_type": loss_type,
        "selected": selected_model,
        "oof_macro_f1": round(oof_macro_f1, 6),
        "train_minutes": round(train_seconds / 60.0, 1),
    }
    for name, score in sorted(candidate_scores.items()):
        row[f"cand_{name}"] = round(score, 6)

    new_row_df = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        try:
            existing = pd.read_csv(RESULTS_CSV)
            combined = pd.concat([existing, new_row_df], ignore_index=True)
            combined.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
        except Exception:
            # CSV is corrupted (e.g. inconsistent columns), overwrite
            new_row_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    else:
        new_row_df.to_csv(RESULTS_CSV, index=False, encoding="utf-8")


def compute_adversarial_weights(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    random_state: int = 42,
) -> tuple[np.ndarray, float]:
    """Train LightGBM discriminator to distinguish train vs test.

    Returns (sample_weights, roc_auc).
    sample_weights are normalized to mean=1. Higher weight = more test-like.
    """
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold as SKF

    X_train = train_df[features].to_numpy(dtype=np.float32)
    X_test = test_df[features].to_numpy(dtype=np.float32)
    X = np.vstack([X_train, X_test])
    y = np.hstack([np.zeros(len(X_train)), np.ones(len(X_test))])

    aucs = []
    cv_adv = SKF(n_splits=3, shuffle=True, random_state=random_state)
    for tr_idx, val_idx in cv_adv.split(X, y):
        disc = LGBMClassifier(n_estimators=100, max_depth=4, random_state=random_state, verbose=-1)
        disc.fit(X[tr_idx], y[tr_idx])
        y_pred = disc.predict_proba(X[val_idx])[:, 1]
        aucs.append(roc_auc_score(y[val_idx], y_pred))

    avg_auc = float(np.mean(aucs))

    disc = LGBMClassifier(n_estimators=100, max_depth=4, random_state=random_state, verbose=-1)
    disc.fit(X, y)
    test_proba = disc.predict_proba(X_train)[:, 1]

    weights = test_proba + 0.3
    weights = weights / float(weights.mean())
    return weights.astype(np.float32), avg_auc


def sample_weighted_indices(
    weights: np.ndarray,
    n_samples: int | None = None,
    random_state: int = 42,
) -> np.ndarray:
    """Sample indices with replacement according to weights.

    Returns indices into the weight array. Oversamples high-weight items.
    """
    rng = np.random.RandomState(random_state)
    if n_samples is None:
        n_samples = len(weights)
    proba = weights / weights.sum()
    return rng.choice(len(weights), size=n_samples, replace=True, p=proba)


def pseudo_label_test(
    bundle: dict,
    test_df: pd.DataFrame,
    features: list[str],
    threshold: float = 0.95,
    max_per_class: dict[int, int] | None = None,
    device=None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Predict test set with bundle, return high-confidence pseudo-labeled samples.

    Returns (pseudo_labeled_df, stats).
    stats: {threshold, total, per_class_N, capped_from}
    """
    proba = predict_bundle_proba(bundle, test_df, device=device)
    max_proba = proba.max(axis=1)
    pseudo_labels = np.argmax(proba, axis=1).astype(int)

    mask = max_proba >= threshold
    df_full = pd.DataFrame({
        "__max_proba": max_proba,
        "__pseudo_label": pseudo_labels,
    }, index=test_df.index)

    capped = {}
    for cls in sorted(np.unique(pseudo_labels)):
        cls_mask = mask & (pseudo_labels == cls)
        cls_indices = test_df.index[cls_mask]
        cap = max_per_class.get(int(cls)) if max_per_class else None
        if cap is not None and len(cls_indices) > cap:
            # keep top-cap by confidence
            cls_scores = df_full.loc[cls_indices, "__max_proba"].sort_values(ascending=False)
            keep = cls_scores.index[:cap]
            drop = cls_scores.index[cap:]
            mask[drop] = False
            capped[str(cls)] = int(len(drop))

    pseudo_df = test_df.loc[mask, features].copy()
    pseudo_df["label"] = pseudo_labels[mask]
    pseudo_df["_pseudo"] = True

    stats = {
        "threshold": float(threshold),
        "total": int(mask.sum()),
        "per_class": {int(k): int(v) for k, v in pd.Series(pseudo_labels[mask]).value_counts().to_dict().items()},
        "capped": capped,
    }
    return pseudo_df, stats


def scan_pseudo_thresholds(
    bundle: dict,
    test_df: pd.DataFrame,
    features: list[str],
    label_counts: dict[int, int],
    thresholds: list[float] | None = None,
    cap_ratio: float = 0.20,
    min_samples: int = 200,
    device=None,
) -> list[dict[str, object]]:
    """Scan thresholds for pseudo-label selection, return sorted by quality.

    Returns list of {threshold, total, per_class, capped, score}.
    score = total / max_possible but penalized for imbalance.
    """
    if thresholds is None:
        thresholds = [0.99, 0.97, 0.95, 0.92, 0.90, 0.85, 0.80]

    max_per_class = {int(k): max(1, int(v * cap_ratio)) for k, v in label_counts.items()}
    results = []
    for th in thresholds:
        _, stats = pseudo_label_test(bundle, test_df, features, threshold=th, max_per_class=max_per_class, device=device)
        if stats["total"] < min_samples:
            continue
        # Score: prefer more samples and balanced classes
        per_class = stats.get("per_class", {})
        if per_class:
            counts = np.array(list(per_class.values()), dtype=np.float32)
            balance = 1.0 - float(counts.std() / (counts.mean() + 1e-9))
        else:
            balance = 0.0
        score = float(stats["total"]) * (0.5 + 0.5 * max(0.0, balance))
        results.append({**stats, "score": round(score, 1), "max_per_class": {int(k): int(v) for k, v in max_per_class.items()}})

    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def load_train(path: Path = TRAIN_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {ID_COLUMN, TARGET_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"training data missing columns: {sorted(missing)}")
    return df


def load_test(path: Path = TEST_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if ID_COLUMN not in df.columns:
        raise ValueError(f"test data missing column: {ID_COLUMN}")
    return df


def feature_columns(train_df: pd.DataFrame) -> list[str]:
    cols = [c for c in train_df.columns if c not in {ID_COLUMN, TARGET_COLUMN}]
    if len(cols) != FEATURE_COUNT:
        raise ValueError(f"expected {FEATURE_COUNT} feature columns, got {len(cols)}: {cols}")
    return cols


def validate_features(df: pd.DataFrame, columns: Iterable[str]) -> None:
    columns = list(columns)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"data missing feature columns: {missing}")
    if df[columns].isna().any().any():
        raise ValueError("feature matrix contains missing values")


def build_extra_trees_model(random_state: int = 2026, n_estimators: int = 500) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight=None,
        max_features=None,
        min_samples_leaf=1,
        n_jobs=1,
    )


def build_hgb_model(random_state: int = 2026, max_iter: int = 350) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        random_state=random_state,
        max_iter=max_iter,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.03,
        categorical_features=list(range(FEATURE_COUNT)),
    )


def build_catboost_model(random_state: int = 2026, iterations: int = 500, depth: int = 6, lr: float = 0.05):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=lr,
        random_seed=random_state,
        verbose=False,
        thread_count=1,
    )


def build_xgboost_model(random_state: int = 2026, n_estimators: int = 500, max_depth: int = 6, lr: float = 0.05):
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=random_state,
        enable_categorical=True,
        n_jobs=1,
        verbosity=0,
    )


def build_lgb_model(random_state: int = 2026, n_estimators: int = 500, max_depth: int = 6, lr: float = 0.05):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=lr,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )


def align_proba(model, proba: np.ndarray, label_order: Iterable[int] = LABELS) -> np.ndarray:
    label_order = [int(x) for x in label_order]
    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    aligned = np.zeros((proba.shape[0], len(label_order)), dtype=np.float32)
    for out_idx, label in enumerate(label_order):
        aligned[:, out_idx] = proba[:, class_to_index[label]]
    return aligned


def apply_temperature(proba: np.ndarray, temperature: float) -> np.ndarray:
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if abs(temperature - 1.0) <= 1e-12:
        return proba.astype(np.float32, copy=False)
    scaled = np.power(np.clip(proba, 1e-12, 1.0), 1.0 / temperature).astype(np.float32, copy=False)
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled_sum = np.clip(scaled_sum, 1e-12, None)
    return scaled / scaled_sum


def apply_class_bias(proba: np.ndarray, class_bias: Iterable[float]) -> np.ndarray:
    bias = np.asarray(list(class_bias), dtype=np.float32)
    if bias.ndim != 1:
        raise ValueError(f"class_bias must be 1D, got shape {bias.shape}")
    if not np.isfinite(bias).all() or (bias <= 0).any():
        raise ValueError(f"class_bias must contain positive finite values, got {bias.tolist()}")
    if proba.shape[1] != len(bias):
        raise ValueError(f"class_bias length {len(bias)} does not match probability columns {proba.shape[1]}")
    scaled = proba * bias[None, :]
    scaled_sum = scaled.sum(axis=1, keepdims=True)
    scaled_sum = np.clip(scaled_sum, 1e-12, None)
    return scaled / scaled_sum


def _needs_target_encoding(bundle: Mapping[str, object]) -> bool:
    return bundle.get("target_encoder") is not None


def _augment_X_for_bundle(bundle: Mapping[str, object], X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Apply target encoding, interaction features, and KMeans if bundle has them."""
    encoder = bundle.get("target_encoder")
    interaction_specs = bundle.get("interaction_specs")
    kmeans_model = bundle.get("kmeans_model")
    feature_cols = bundle.get("feature_columns")

    if encoder is None and interaction_specs is None and kmeans_model is None:
        if isinstance(X, pd.DataFrame):
            X = X[feature_cols].to_numpy(dtype=np.int64, copy=True)
        return np.asarray(X, dtype=np.int64)

    if isinstance(X, np.ndarray):
        raise ValueError("feature augmentation requires a DataFrame, got ndarray")

    parts: list[np.ndarray] = []

    # Base categorical features
    X_cat = X[feature_cols].to_numpy(dtype=np.int64, copy=True)
    parts.append(X_cat.astype(np.float32))

    # Target encoding
    if encoder is not None:
        from feature_engineering import transform_target_encoder
        te = transform_target_encoder(X, feature_cols, encoder)
        parts.append(te)

    # Legacy interaction pairs (from select_interactions)
    interaction_pairs = bundle.get("interaction_pairs")
    if interaction_pairs:
        from feature_engineering import generate_interactions_from_pairs
        inter = generate_interactions_from_pairs(X, feature_cols, interaction_pairs)
        parts.append(inter.astype(np.float32))

    # New pairwise interaction features (prod/ratio/diff)
    if interaction_specs is not None:
        vals = X[feature_cols].to_numpy(dtype=np.float32)
        n = len(feature_cols)
        for i, j, itype in interaction_specs:
            fi = vals[:, i]
            fj = vals[:, j]
            if itype == "prod":
                parts.append((fi * fj).astype(np.float32))
            elif itype == "diff":
                parts.append((fi - fj).astype(np.float32))
            elif itype == "ratio":
                parts.append(np.divide(fi, fj + 1.0, dtype=np.float32))

    # KMeans cluster distances
    if kmeans_model is not None:
        kmeans_feat = kmeans_model.transform(X[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        parts.append(kmeans_feat)

    return np.column_stack(parts).astype(np.float32)


def _predict_sklearn_gbdt_proba(bundle: Mapping[str, object], X: pd.DataFrame | np.ndarray) -> np.ndarray:
    model = bundle["model"]
    labels = bundle.get("labels", LABELS)
    X = _augment_X_for_bundle(bundle, X)
    return align_proba(model, model.predict_proba(X), labels)


def _predict_tree_bundle_proba(bundle: Mapping[str, object], X: pd.DataFrame | np.ndarray) -> np.ndarray:
    model_type = bundle.get("model_type", "tree_blend")
    if model_type not in {"tree_blend", "blend"}:
        raise ValueError(f"unknown tree bundle type: {model_type}")
    models = bundle["models"]
    weights = bundle["blend_weights"]
    if len(models) != len(weights):
        raise ValueError("tree blend weights do not match model count")
    labels = bundle.get("labels", LABELS)
    X = _augment_X_for_bundle(bundle, X)
    proba = np.zeros((len(X), len(labels)), dtype=np.float32)
    for weight, model in zip(weights, models):
        proba += float(weight) * align_proba(model, model.predict_proba(X), labels)
    return proba


def _predict_torch_ensemble_proba(bundle: Mapping[str, object], X: np.ndarray, device=None) -> np.ndarray:
    from tabular_nn import predict_torch_ensemble_proba

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy(dtype=np.int64, copy=True)
    else:
        X = np.asarray(X, dtype=np.int64)
    fold_bundles = bundle.get("fold_models", [])
    return predict_torch_ensemble_proba(fold_bundles, X, device=device)


def _predict_pattern_lookup_proba(bundle: Mapping[str, object], X: pd.DataFrame | np.ndarray) -> np.ndarray:
    labels = bundle.get("labels", LABELS)
    lookup = bundle.get("lookup", {})
    default_proba = np.asarray(bundle.get("default_proba", [1.0 / len(labels)] * len(labels)), dtype=np.float32)
    if isinstance(X, pd.DataFrame):
        values = X.to_numpy(dtype=np.int64, copy=True)
    else:
        values = np.asarray(X, dtype=np.int64)
    proba = np.empty((len(values), len(labels)), dtype=np.float32)
    for row_idx, row in enumerate(values):
        proba[row_idx] = lookup.get(tuple(int(v) for v in row), default_proba)
    return proba


def predict_bundle_proba(
    bundle: Mapping[str, object],
    X: pd.DataFrame | np.ndarray,
    device=None,
) -> np.ndarray:
    if "selected_model_bundle" in bundle:
        bundle = bundle["selected_model_bundle"]

    model_type = bundle.get("model_type")
    if model_type == "sklearn_gbdt":
        proba = _predict_sklearn_gbdt_proba(bundle, X)
    elif model_type in {"tree_blend", "blend"}:
        proba = _predict_tree_bundle_proba(bundle, X)
    elif model_type == "torch_ensemble":
        proba = _predict_torch_ensemble_proba(bundle, X, device=device)
    elif model_type == "pattern_lookup":
        proba = _predict_pattern_lookup_proba(bundle, X)
    elif model_type == "fusion":
        components = bundle.get("components", {})
        weights = bundle.get("component_weights", {})
        if not components or not weights:
            raise ValueError("fusion bundle missing components or weights")
        labels = bundle.get("labels", LABELS)
        proba = np.zeros((len(X), len(labels)), dtype=np.float32)
        for name, component in components.items():
            weight = float(weights[name])
            proba += weight * predict_bundle_proba(component, X, device=device)
    else:
        raise ValueError(f"unknown model_type: {model_type}")

    proba = apply_temperature(proba, float(bundle.get("temperature", 1.0)))
    proba = apply_class_bias(proba, bundle.get("class_bias", [1.0] * proba.shape[1]))
    return proba


def predict_from_bundle(
    bundle: Mapping[str, object],
    X: pd.DataFrame | np.ndarray,
    device=None,
) -> np.ndarray:
    labels = np.asarray(bundle.get("labels", LABELS), dtype=int)
    proba = predict_bundle_proba(bundle, X, device=device)
    return labels[np.argmax(proba, axis=1)]


def dump_joblib_atomic(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    joblib.dump(payload, tmp_path, compress=3)
    tmp_path.replace(path)


def write_csv_atomic(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    df.to_csv(tmp_path, **kwargs)
    tmp_path.replace(path)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
