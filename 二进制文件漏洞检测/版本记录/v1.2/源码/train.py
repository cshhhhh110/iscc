"""Training entrypoint for the ISCC binary vulnerability v1.2 pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import joblib
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from byte_features import DEFAULT_BYTE_LENGTH, rows_to_byte_matrix
from dataset import binary_path, read_csv_rows
from features import extract_features, get_feature_columns
from models import (
    CWE_MAPPING_NAME,
    CWE_MODEL_NAME,
    FEATURE_COLUMNS_NAME,
    FUSION_CONFIG_NAME,
    LEGACY_TRAIN_CACHE_NAME,
    LABEL_MODEL_NAME,
    NEURAL_BUNDLE_NAME,
    SUBMISSION_NAME,
    TABULAR_BUNDLE_NAME,
    TRAIN_BYTE_CACHE_NAME,
    TRAIN_CACHE_NAME,
    ensure_model_dir,
)
from nn_models import (
    ByteMetaMultiTaskNet,
    TabularNormalizer,
    apply_tabular_normalizer,
    build_cwe_class_weights,
    fit_tabular_normalizer,
    predict_multitask,
)
from utils import write_json


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "train.csv"
BINARIES_DIR = ROOT / "binaries"
MODEL_DIR = ROOT / "模型"
OUTPUT_DIR = ROOT / "提交结果"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the v1.2 ISCC competition baseline.")
    parser.add_argument("--epochs", type=int, default=10, help="Neural training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size.")
    parser.add_argument("--byte-length", type=int, default=DEFAULT_BYTE_LENGTH, help="Fixed byte-window length.")
    parser.add_argument("--lr", type=float, default=7e-4, help="Neural learning rate.")
    parser.add_argument("--dropout", type=float, default=0.20, help="Neural dropout rate.")
    parser.add_argument("--byte-embedding-dim", type=int, default=24, help="Byte embedding width for the neural branch.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience for neural training.")
    parser.add_argument("--cwe-loss-weight", type=float, default=1.25, help="Loss weight for the CWE head.")
    parser.add_argument("--retrain-tree", action="store_true", help="Force retraining of the tabular ensemble.")
    return parser.parse_args()


def _rows_to_matrix(rows: List[Dict[str, str]]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    feature_columns = get_feature_columns()
    matrix = np.zeros((len(rows), len(feature_columns)), dtype=np.float32)
    y_label = np.zeros(len(rows), dtype=np.int32)
    cwe_ids: List[str] = []
    binary_ids: List[str] = []

    for index, row in enumerate(tqdm(rows, desc="Extracting tabular features", total=len(rows))):
        binary_ids.append(row["binary_id"])
        y_label[index] = int(row["label"])
        cwe_ids.append(row["cwe_id"])
        feats = extract_features(binary_path(BINARIES_DIR, row["binary_id"]))
        matrix[index] = np.asarray([feats[name] for name in feature_columns], dtype=np.float32)

    return matrix, y_label, cwe_ids, binary_ids


def _load_or_build_tabular_cache(rows: List[Dict[str, str]]) -> Dict[str, object]:
    versioned_cache = MODEL_DIR / TRAIN_CACHE_NAME
    legacy_cache = MODEL_DIR / LEGACY_TRAIN_CACHE_NAME
    if versioned_cache.exists():
        return joblib.load(versioned_cache)
    if legacy_cache.exists():
        cache = joblib.load(legacy_cache)
        joblib.dump(cache, versioned_cache)
        return cache

    X, y_label, cwe_ids, binary_ids = _rows_to_matrix(rows)
    cache = {
        "X": X,
        "y_label": y_label,
        "cwe_ids": cwe_ids,
        "binary_ids": binary_ids,
        "feature_columns": get_feature_columns(),
    }
    joblib.dump(cache, versioned_cache)
    return cache


def _load_or_build_byte_cache(rows: List[Dict[str, str]], byte_length: int) -> Dict[str, object]:
    cache_path = MODEL_DIR / TRAIN_BYTE_CACHE_NAME
    if cache_path.exists():
        cache = joblib.load(cache_path)
        cached_length = int(cache.get("byte_length", 0))
        cached_matrix = cache.get("X_byte")
        if cached_length == byte_length and getattr(cached_matrix, "shape", (0, 0))[1] == byte_length:
            return cache
        print("warning: byte cache length differs from current config; rebuilding byte cache.")
    X_byte, binary_ids = rows_to_byte_matrix(rows, BINARIES_DIR, byte_length=byte_length, desc="Extracting byte windows")
    cache = {
        "X_byte": X_byte,
        "binary_ids": binary_ids,
        "byte_length": byte_length,
    }
    joblib.dump(cache, cache_path)
    return cache


def _aligned_positive_probability(model, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    classes = np.asarray(getattr(model, "classes_", [0, 1]))
    if len(classes) == 1:
        return np.zeros(X.shape[0], dtype=np.float32)
    if 1 in classes:
        positive_index = int(np.where(classes == 1)[0][0])
    else:
        positive_index = min(1, proba.shape[1] - 1)
    return np.asarray(proba[:, positive_index], dtype=np.float32)


def _aligned_cwe_probability(model, X: np.ndarray, num_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=np.float32)
    aligned = np.zeros((X.shape[0], num_classes), dtype=np.float32)
    model_classes = np.asarray(getattr(model, "classes_", np.arange(raw.shape[1])))
    for source_index, class_index in enumerate(model_classes):
        class_int = int(class_index)
        if 0 <= class_int < num_classes:
            aligned[:, class_int] = raw[:, source_index]
    row_sum = aligned.sum(axis=1, keepdims=True)
    aligned = aligned / np.maximum(row_sum, 1e-12)
    return aligned


def _best_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    thresholds = np.arange(0.08, 0.93, 0.005)
    best_threshold = 0.50
    best_score = -1.0
    for threshold in thresholds:
        pred = (proba >= threshold).astype(int)
        score = f1_score(y_true, pred)
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)
    return best_threshold


def _search_binary_fusion(y_true: np.ndarray, tree_probs: np.ndarray, neural_probs: np.ndarray) -> Tuple[float, float, float]:
    best_weight = 1.0
    best_threshold = 0.5
    best_score = -1.0
    best_acc = -1.0
    for neural_weight in np.linspace(0.0, 1.0, 41):
        fused = neural_weight * neural_probs + (1.0 - neural_weight) * tree_probs
        threshold = _best_threshold(y_true, fused)
        pred = (fused >= threshold).astype(int)
        score = f1_score(y_true, pred)
        acc = accuracy_score(y_true, pred)
        if score > best_score or (score == best_score and acc > best_acc):
            best_score = score
            best_acc = acc
            best_weight = float(neural_weight)
            best_threshold = float(threshold)
    return best_weight, best_threshold, best_score


def _search_cwe_fusion(y_true: np.ndarray, tree_probs: np.ndarray, neural_probs: np.ndarray) -> Tuple[float, float]:
    if len(y_true) == 0:
        return 1.0, 0.0
    best_weight = 1.0
    best_macro = -1.0
    best_acc = -1.0
    class_labels = list(range(tree_probs.shape[1]))
    for neural_weight in np.linspace(0.0, 1.0, 41):
        fused = neural_weight * neural_probs + (1.0 - neural_weight) * tree_probs
        pred = fused.argmax(axis=1)
        macro = f1_score(y_true, pred, average="macro", labels=class_labels, zero_division=0)
        acc = accuracy_score(y_true, pred)
        if macro > best_macro or (macro == best_macro and acc > best_acc):
            best_macro = float(macro)
            best_acc = float(acc)
            best_weight = float(neural_weight)
    return best_weight, best_macro


def _split_indices(y_label: np.ndarray, cwe_ids: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_label))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=y_label,
    )

    train_idx_set = set(train_idx.tolist())
    val_idx_list = list(val_idx.tolist())
    train_cwe_counts = {}
    for idx in train_idx:
        if y_label[idx] == 1:
            cwe = cwe_ids[idx]
            train_cwe_counts[cwe] = train_cwe_counts.get(cwe, 0) + 1

    moved = 0
    for cwe in sorted(set(cwe_ids)):
        if not cwe:
            continue
        if train_cwe_counts.get(cwe, 0) > 0:
            continue
        candidates = [idx for idx in val_idx_list if y_label[idx] == 1 and cwe_ids[idx] == cwe]
        if candidates:
            chosen = candidates[0]
            val_idx_list.remove(chosen)
            train_idx_set.add(chosen)
            moved += 1

    if moved:
        print(f"note: moved {moved} rare CWE samples into the training fold for coverage.")

    train_idx = np.array(sorted(train_idx_set), dtype=np.int64)
    val_idx = np.array(sorted(val_idx_list), dtype=np.int64)
    return train_idx, val_idx


def _build_label_ensemble(random_state: int) -> VotingClassifier:
    hgb = HistGradientBoostingClassifier(
        random_state=random_state,
        learning_rate=0.045,
        max_iter=420,
        max_leaf_nodes=31,
        max_depth=9,
        min_samples_leaf=16,
        l2_regularization=0.02,
        early_stopping=True,
        validation_fraction=0.12,
        verbose=1,
    )
    extra = ExtraTreesClassifier(
        n_estimators=650,
        random_state=random_state + 11,
        n_jobs=-1,
        class_weight="balanced",
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=False,
        verbose=1,
    )
    return VotingClassifier(
        estimators=[("hgb", hgb), ("extra", extra)],
        voting="soft",
        weights=[2.0, 1.0],
        n_jobs=1,
    )


def _cwe_class_weight_map(y_cwe: np.ndarray, num_classes: int) -> Dict[int, float]:
    counts = np.bincount(y_cwe, minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights = np.clip(weights, 0.35, 18.0)
    return {index: float(weight) for index, weight in enumerate(weights)}


def _build_cwe_ensemble(class_weight_map: Dict[int, float], random_state: int) -> VotingClassifier:
    forest = RandomForestClassifier(
        n_estimators=520,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight_map,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=True,
        verbose=1,
    )
    extra = ExtraTreesClassifier(
        n_estimators=720,
        random_state=random_state + 17,
        n_jobs=-1,
        class_weight=class_weight_map,
        max_features="sqrt",
        min_samples_leaf=1,
        bootstrap=False,
        verbose=1,
    )
    return VotingClassifier(
        estimators=[("rf", forest), ("extra", extra)],
        voting="soft",
        weights=[1.0, 1.2],
        n_jobs=1,
    )


def _train_tabular_ensemble(
    X: np.ndarray,
    y_label: np.ndarray,
    cwe_ids: Sequence[str],
    force_retrain: bool,
) -> Tuple[object, float, object, List[str]]:
    label_path = MODEL_DIR / LABEL_MODEL_NAME
    cwe_path = MODEL_DIR / CWE_MODEL_NAME
    if not force_retrain and label_path.exists() and cwe_path.exists():
        label_bundle = joblib.load(label_path)
        cwe_bundle = joblib.load(cwe_path)
        return (
            label_bundle["model"],
            float(label_bundle["threshold"]),
            cwe_bundle["model"],
            list(cwe_bundle["classes"]),
        )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X,
        y_label,
        test_size=0.2,
        random_state=42,
        stratify=y_label,
    )
    label_model = _build_label_ensemble(random_state=42)
    label_model.fit(X_tr, y_tr)
    threshold = _best_threshold(y_val, label_model.predict_proba(X_val)[:, 1])
    label_model = _build_label_ensemble(random_state=42)
    label_model.fit(X, y_label)

    positive_mask = y_label == 1
    positive_cwe_ids = [cwe_ids[i] for i in range(len(cwe_ids)) if positive_mask[i]]
    classes = sorted(set(positive_cwe_ids))
    mapping = {name: index for index, name in enumerate(classes)}
    y_cwe = np.asarray([mapping[cwe] for cwe in positive_cwe_ids], dtype=np.int32)
    class_weight_map = _cwe_class_weight_map(y_cwe, len(classes))

    cwe_model = _build_cwe_ensemble(class_weight_map=class_weight_map, random_state=42)
    cwe_model.fit(X[positive_mask], y_cwe)

    joblib.dump(
        {
            "model": label_model,
            "threshold": threshold,
            "feature_columns": get_feature_columns(),
            "model_family": "hist_gradient_boosting_extra_trees_vote",
        },
        label_path,
    )
    joblib.dump(
        {
            "model": cwe_model,
            "feature_columns": get_feature_columns(),
            "classes": classes,
            "model_family": "random_forest_extra_trees_vote",
            "class_weight_map": class_weight_map,
        },
        cwe_path,
    )

    return label_model, threshold, cwe_model, classes


def _build_neural_model(
    tabular_dim: int,
    num_cwe_classes: int,
    dropout: float,
    byte_embedding_dim: int,
) -> ByteMetaMultiTaskNet:
    return ByteMetaMultiTaskNet(
        tabular_dim=tabular_dim,
        num_cwe_classes=num_cwe_classes,
        dropout=dropout,
        byte_embedding_dim=byte_embedding_dim,
    )


def _train_neural_model(
    X_byte: np.ndarray,
    X_tabular: np.ndarray,
    y_label: np.ndarray,
    y_cwe: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    num_cwe_classes: int,
    epochs: int,
    batch_size: int,
    lr: float,
    dropout: float,
    byte_embedding_dim: int,
    patience: int,
    cwe_loss_weight: float,
) -> Tuple[ByteMetaMultiTaskNet, TabularNormalizer, Dict[str, float], np.ndarray, np.ndarray]:
    normalizer = fit_tabular_normalizer(X_tabular[train_idx])
    X_train_tab = apply_tabular_normalizer(X_tabular[train_idx], normalizer)
    X_val_tab = apply_tabular_normalizer(X_tabular[val_idx], normalizer)

    model = _build_neural_model(
        tabular_dim=X_train_tab.shape[1],
        num_cwe_classes=num_cwe_classes,
        dropout=dropout,
        byte_embedding_dim=byte_embedding_dim,
    ).to(DEVICE)
    pos_count = float(y_label[train_idx].sum())
    neg_count = float(len(train_idx) - pos_count)
    pos_weight = torch.tensor([neg_count / max(pos_count, 1.0)], dtype=torch.float32, device=DEVICE)
    label_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    cwe_weights = build_cwe_class_weights(y_cwe[train_idx], num_cwe_classes).to(DEVICE)
    cwe_criterion = torch.nn.CrossEntropyLoss(weight=cwe_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_state = None
    best_score = -1.0
    best_metrics: Dict[str, float] = {}
    epochs_without_improvement = 0

    train_order = np.arange(len(train_idx))

    for epoch in range(1, epochs + 1):
        model.train()
        np.random.default_rng(42 + epoch).shuffle(train_order)
        total_loss = 0.0
        total_label_loss = 0.0
        total_cwe_loss = 0.0
        total_batches = int(np.ceil(len(train_order) / batch_size))
        with tqdm(total=total_batches, desc=f"Epoch {epoch}/{epochs}", unit="batch") as progress:
            for batch_no, start in enumerate(range(0, len(train_order), batch_size), 1):
                batch_positions = train_order[start : start + batch_size]
                batch_indices = train_idx[batch_positions]
                byte_batch = torch.from_numpy(X_byte[batch_indices]).to(DEVICE)
                tab_batch = torch.from_numpy(X_train_tab[batch_positions]).to(DEVICE)
                label_batch = torch.from_numpy(y_label[batch_indices].astype(np.float32)).to(DEVICE)
                cwe_batch = torch.from_numpy(y_cwe[batch_indices].astype(np.int64)).to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                label_logits, cwe_logits = model(byte_batch, tab_batch)
                label_loss = label_criterion(label_logits, label_batch)
                positive_mask = cwe_batch >= 0
                if positive_mask.any():
                    cwe_loss = cwe_criterion(cwe_logits[positive_mask], cwe_batch[positive_mask])
                else:
                    cwe_loss = torch.zeros((), device=DEVICE)
                loss = label_loss + cwe_loss_weight * cwe_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

                total_loss += float(loss.item())
                total_label_loss += float(label_loss.item())
                total_cwe_loss += float(cwe_loss.item())
                progress.set_postfix_str(f"loss={total_loss / batch_no:.4f}, label={total_label_loss / batch_no:.4f}, cwe={total_cwe_loss / batch_no:.4f}")
                progress.update(1)

        neural_label_probs, neural_cwe_probs = predict_multitask(
            model,
            X_byte[val_idx],
            X_val_tab,
            batch_size=batch_size,
            device=DEVICE,
            desc=f"Validate neural epoch {epoch}",
        )
        neural_threshold = _best_threshold(y_label[val_idx], neural_label_probs)
        neural_label_pred = (neural_label_probs >= neural_threshold).astype(int)
        neural_label_f1 = f1_score(y_label[val_idx], neural_label_pred)
        val_positive_mask = y_label[val_idx] == 1
        if val_positive_mask.any():
            neural_cwe_pred = neural_cwe_probs[val_positive_mask].argmax(axis=1)
            neural_cwe_macro = f1_score(
                y_cwe[val_idx][val_positive_mask],
                neural_cwe_pred,
                average="macro",
                labels=list(range(num_cwe_classes)),
                zero_division=0,
            )
        else:
            neural_cwe_macro = 0.0

        composite_score = neural_label_f1 + 0.45 * neural_cwe_macro
        scheduler.step(composite_score)
        if composite_score > best_score:
            best_score = composite_score
            best_state = {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "label_f1": float(neural_label_f1),
                "cwe_macro_f1": float(neural_cwe_macro),
                "threshold": float(neural_threshold),
            }
            best_metrics = {
                "label_f1": float(neural_label_f1),
                "cwe_macro_f1": float(neural_cwe_macro),
                "threshold": float(neural_threshold),
            }

            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"note: neural early stopping at epoch {epoch} after {patience} stale epochs.")
                break

    assert best_state is not None
    model.load_state_dict(best_state["state_dict"])
    return model, normalizer, best_metrics, X_train_tab, X_val_tab


def main() -> None:
    args = _parse_args()
    ensure_model_dir(MODEL_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(TRAIN_CSV)

    with tqdm(total=5, desc="Training pipeline", unit="stage") as pipeline:
        pipeline.set_postfix_str("tabular cache")
        tabular_cache = _load_or_build_tabular_cache(rows)
        X = np.asarray(tabular_cache["X"], dtype=np.float32)
        y_label = np.asarray(tabular_cache["y_label"], dtype=np.int32)
        cwe_ids = list(tabular_cache["cwe_ids"])
        feature_columns = list(tabular_cache["feature_columns"])
        joblib.dump(tabular_cache, MODEL_DIR / TRAIN_CACHE_NAME)
        pipeline.update(1)

        pipeline.set_postfix_str("byte cache")
        byte_cache = _load_or_build_byte_cache(rows, args.byte_length)
        X_byte = np.asarray(byte_cache["X_byte"], dtype=np.uint8)
        joblib.dump(byte_cache, MODEL_DIR / TRAIN_BYTE_CACHE_NAME)
        pipeline.update(1)

        pipeline.set_postfix_str("tabular ensemble")
        label_model, tree_threshold, cwe_model, cwe_classes = _train_tabular_ensemble(
            X,
            y_label,
            cwe_ids,
            force_retrain=args.retrain_tree,
        )
        positive_mask = y_label == 1
        cwe_mapping = {name: index for index, name in enumerate(cwe_classes)}
        y_cwe = np.full(len(cwe_ids), -1, dtype=np.int32)
        for index, cwe_id in enumerate(cwe_ids):
            if cwe_id:
                y_cwe[index] = cwe_mapping[cwe_id]
        pipeline.update(1)

        pipeline.set_postfix_str("neural multitask")
        train_idx, val_idx = _split_indices(y_label, cwe_ids)
        neural_model, normalizer, neural_metrics, X_train_tab, X_val_tab = _train_neural_model(
            X_byte=X_byte,
            X_tabular=X,
            y_label=y_label,
            y_cwe=y_cwe,
            train_idx=train_idx,
            val_idx=val_idx,
            num_cwe_classes=len(cwe_classes),
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            dropout=args.dropout,
            byte_embedding_dim=args.byte_embedding_dim,
            patience=args.patience,
            cwe_loss_weight=args.cwe_loss_weight,
        )
        pipeline.update(1)

        pipeline.set_postfix_str("fusion save")
        tree_label_probs = _aligned_positive_probability(label_model, X[val_idx])
        tree_cwe_probs = _aligned_cwe_probability(cwe_model, X[val_idx][y_label[val_idx] == 1], len(cwe_classes))
        neural_label_probs, neural_cwe_probs = predict_multitask(
            neural_model,
            X_byte[val_idx],
            X_val_tab,
            batch_size=args.batch_size,
            device=DEVICE,
            desc="Final fusion validation",
        )
        neural_label_weight, fusion_threshold, fused_label_f1 = _search_binary_fusion(
            y_label[val_idx],
            tree_label_probs,
            neural_label_probs,
        )
        val_positive_mask = y_label[val_idx] == 1
        if val_positive_mask.any():
            neural_cwe_weight, fused_cwe_macro = _search_cwe_fusion(
                y_cwe[val_idx][val_positive_mask],
                tree_cwe_probs,
                neural_cwe_probs[val_positive_mask],
            )
        else:
            neural_cwe_weight, fused_cwe_macro = 1.0, 0.0

        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in neural_model.state_dict().items()},
                "model_config": {
                    "tabular_dim": int(X.shape[1]),
                    "num_cwe_classes": int(len(cwe_classes)),
                    "dropout": float(args.dropout),
                    "byte_embedding_dim": int(args.byte_embedding_dim),
                },
                "normalizer": {
                    "mean": torch.from_numpy(normalizer.mean),
                    "std": torch.from_numpy(normalizer.std),
                },
                "feature_columns": feature_columns,
                "cwe_classes": cwe_classes,
                "byte_length": int(args.byte_length),
                "metrics": neural_metrics,
                "model_version": "v1.2",
            },
            MODEL_DIR / NEURAL_BUNDLE_NAME,
        )

        write_json(
            MODEL_DIR / FEATURE_COLUMNS_NAME,
            feature_columns,
        )
        write_json(
            MODEL_DIR / CWE_MAPPING_NAME,
            {
                "classes": cwe_classes,
                "class_to_index": {name: index for index, name in enumerate(cwe_classes)},
            },
        )
        write_json(
            MODEL_DIR / FUSION_CONFIG_NAME,
            {
                "model_version": "v1.2",
                "device": str(DEVICE),
                "byte_length": int(args.byte_length),
                "batch_size": int(args.batch_size),
                "epochs": int(args.epochs),
                "tree_threshold": float(tree_threshold),
                "fusion_threshold": float(fusion_threshold),
                "tree_label_weight": float(1.0 - neural_label_weight),
                "neural_label_weight": float(neural_label_weight),
                "tree_cwe_weight": float(1.0 - neural_cwe_weight),
                "neural_cwe_weight": float(neural_cwe_weight),
                "neural_label_f1": float(neural_metrics["label_f1"]),
                "neural_cwe_macro_f1": float(neural_metrics["cwe_macro_f1"]),
                "fused_label_f1": float(fused_label_f1),
                "fused_cwe_macro_f1": float(fused_cwe_macro),
                "cwe_loss_weight": float(args.cwe_loss_weight),
                "byte_embedding_dim": int(args.byte_embedding_dim),
                "patience": int(args.patience),
            },
        )
        joblib.dump(
            {
                "label_model_file": LABEL_MODEL_NAME,
                "cwe_model_file": CWE_MODEL_NAME,
                "feature_columns": feature_columns,
                "cwe_classes": cwe_classes,
                "tree_threshold": float(tree_threshold),
                "tree_model_source": "trained_v1.2",
                "model_version": "v1.2",
            },
            MODEL_DIR / TABULAR_BUNDLE_NAME,
        )
        pipeline.update(1)

    print(f"saved tree bundle to {MODEL_DIR / TABULAR_BUNDLE_NAME}")
    print(f"saved neural bundle to {MODEL_DIR / NEURAL_BUNDLE_NAME}")
    print(f"saved fusion config to {MODEL_DIR / FUSION_CONFIG_NAME}")
    print(f"submission target: {OUTPUT_DIR / SUBMISSION_NAME}")
    print(f"device: {DEVICE}")


if __name__ == "__main__":
    main()
