from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def fit_target_encoder(
    frame: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    alpha: float = 5.0,
    num_classes: int = 3,
) -> dict[str, dict[int, np.ndarray]]:
    """> {feature: {value: np.array([P0, P1, P2])}}"""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    prior = counts / counts.sum()

    encoder: dict[str, dict[int, np.ndarray]] = {}
    for feat in features:
        feat_map: dict[int, np.ndarray] = {}
        for val in sorted(int(v) for v in frame[feat].unique()):
            mask = (frame[feat] == val).to_numpy()
            cnt = np.bincount(labels[mask], minlength=num_classes).astype(np.float64)
            smoothed = (cnt + alpha * prior) / (cnt.sum() + alpha)
            feat_map[val] = smoothed.astype(np.float32)
        encoder[feat] = feat_map
    return encoder


def transform_target_encoder(
    frame: pd.DataFrame,
    features: list[str],
    encoder: dict[str, dict[int, np.ndarray]],
    num_classes: int = 3,
) -> np.ndarray:
    """Return (n_samples, len(features) * (num_classes - 1)) float32 matrix."""
    fallback = np.full(num_classes, 1.0 / num_classes, dtype=np.float32)
    parts = []
    for feat in features:
        feat_encoder = encoder[feat]
        mapped = np.array(
            [feat_encoder.get(int(v), fallback) for v in frame[feat]],
            dtype=np.float32,
        )  # (n, num_classes)
        parts.append(mapped[:, 1:])  # skip class 0
    return np.column_stack(parts).astype(np.float32)


def select_interactions(
    frame: pd.DataFrame,
    features: list[str],
    labels: np.ndarray,
    top_k: int = 30,
    random_state: int = 42,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """MI-select top_k pairwise feature index pairs. Returns (pairs, mi_scores)."""
    codes: list[np.ndarray] = []
    pairs: list[tuple[int, int]] = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            code = frame[features[i]].astype(np.int64) * 10 + frame[features[j]].astype(np.int64)
            codes.append(code)
            pairs.append((i, j))

    X_interact = np.column_stack(codes).astype(np.int64)
    mi_scores = mutual_info_classif(
        X_interact, labels, discrete_features=True, random_state=random_state
    )
    # Build full-length score array for later selection
    return pairs, mi_scores


def generate_interactions_from_pairs(
    frame: pd.DataFrame,
    features: list[str],
    pair_indices: list[tuple[int, int]],
) -> np.ndarray:
    """Generate interaction codes for specified (i,j) pairs."""
    codes = []
    for i, j in pair_indices:
        code = frame[features[i]].astype(np.int64) * 10 + frame[features[j]].astype(np.int64)
        codes.append(code)
    return np.column_stack(codes).astype(np.int64)


def select_top_interaction_pairs(
    pairs: list[tuple[int, int]],
    mi_scores: np.ndarray,
    top_k: int = 30,
) -> list[tuple[int, int]]:
    """Return top_k pairs sorted by MI score descending."""
    top_indices = np.argsort(mi_scores)[-top_k:][::-1]
    return [pairs[int(i)] for i in top_indices]


def build_feature_names_with_encoding(
    features: list[str],
    num_classes: int = 3,
) -> list[str]:
    encoded_names = []
    for feat in features:
        for c in range(1, num_classes):
            encoded_names.append(f"{feat}_te_p{c}")
    return encoded_names


def generate_pairwise_interactions(
    frame: pd.DataFrame,
    features: list[str],
    labels: np.ndarray | None = None,
    top_k: int = 60,
    random_state: int = 42,
) -> tuple[np.ndarray, list[tuple[int, int, str]], list[str]]:
    """Generate pairwise interaction features: product, ratio, and difference.

    Returns (interaction_matrix, interaction_specs, interaction_names)
    where each spec is (feat_i_idx, feat_j_idx, interaction_type).
    """
    n = len(features)
    values = frame[features].to_numpy(dtype=np.float32)

    product_list: list[np.ndarray] = []
    diff_list: list[np.ndarray] = []
    ratio_list: list[np.ndarray] = []
    specs: list[tuple[int, int, str]] = []
    names: list[str] = []

    for i in range(n):
        for j in range(i + 1, n):
            fi = values[:, i]
            fj = values[:, j]

            prod = fi * fj
            product_list.append(prod)
            specs.append((i, j, "prod"))
            names.append(f"{features[i]}_x_{features[j]}")

            diff_val = fi - fj
            diff_list.append(diff_val)
            specs.append((i, j, "diff"))
            names.append(f"{features[i]}_minus_{features[j]}")

            ratio_val = np.divide(fi, fj + 1.0, dtype=np.float32)
            ratio_list.append(ratio_val)
            specs.append((i, j, "ratio"))
            names.append(f"{features[i]}_div_{features[j]}")

    all_interactions = np.column_stack(product_list + diff_list + ratio_list).astype(np.float32)

    if labels is not None and top_k > 0 and top_k < all_interactions.shape[1]:
        selected_indices = _select_interactions_by_importance(all_interactions, labels, top_k, random_state)
        all_interactions = all_interactions[:, selected_indices]
        specs = [specs[int(idx)] for idx in selected_indices]
        names = [names[int(idx)] for idx in selected_indices]

    return all_interactions, specs, names


def _select_interactions_by_importance(
    X_inter: np.ndarray,
    labels: np.ndarray,
    top_k: int,
    random_state: int = 42,
) -> np.ndarray:
    """Select top_k interaction features by LightGBM importance."""
    from lightgbm import LGBMClassifier

    model = LGBMClassifier(
        n_estimators=100,
        max_depth=4,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    model.fit(X_inter, labels)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[-top_k:][::-1]
    return top_indices


def generate_kmeans_features(
    frame: pd.DataFrame,
    features: list[str],
    n_clusters: int = 10,
    random_state: int = 42,
):
    """Fit KMeans on the feature matrix and return distances to each cluster center.

    Returns (distance_features, kmeans_model).
    """
    from sklearn.cluster import KMeans

    X = frame[features].to_numpy(dtype=np.float32)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    distances = kmeans.transform(X).astype(np.float32)
    return distances, kmeans


def transform_kmeans_features(
    frame: pd.DataFrame,
    features: list[str],
    kmeans_model,
) -> np.ndarray:
    """Transform using a fitted KMeans model, returning distance features."""
    X = frame[features].to_numpy(dtype=np.float32)
    return kmeans_model.transform(X).astype(np.float32)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling. Returns (X_resampled, y_resampled)."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "SMOTE requires imbalanced-learn. Install it with: pip install imbalanced-learn"
        )

    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
