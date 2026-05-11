"""OOF error analysis for binary vulnerability detection (v2.5)."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from byte_features import DEFAULT_BYTE_LENGTH
from dataset import read_csv_rows
from features import get_feature_columns
from models import (
    CWE_MAPPING_NAME, FEATURE_COLUMNS_NAME, FUSION_CONFIG_NAME,
    seed_label_model, seed_cwe_model, seed_neural_bundle,
    ensure_model_dir,
)
from nn_models import (
    ByteMetaMultiTaskNet, TabularNormalizer,
    apply_tabular_normalizer, predict_multitask,
)
from train import (
    _aligned_positive_probability, _aligned_cwe_probability,
    _build_label_ensemble, _build_cwe_gbdt, _cwe_class_weight_map,
    _split_indices, _load_or_build_tabular_cache, _load_or_build_byte_cache,
    DEVICE,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "模型"
TRAIN_CSV = ROOT / "train.csv"
PSEUDO_CSV = ROOT / "pseudo_train_v2.5.csv"
OUTPUT_REPORT = MODEL_DIR.resolve() / "oof_report_v2.5.json"
TRAIN_ROWS = read_csv_rows(TRAIN_CSV)
PSEUDO_ROWS = read_csv_rows(PSEUDO_CSV) if PSEUDO_CSV.exists() else []
ALL_ROWS = TRAIN_ROWS + PSEUDO_ROWS


def _build_data():
    """Load or build cached training features (train + pseudo to match cache)."""
    tab_cache = _load_or_build_tabular_cache(ALL_ROWS)
    X = np.asarray(tab_cache["X"], dtype=np.float32)
    y_label = np.asarray(tab_cache["y_label"], dtype=np.int32)
    cwe_ids_list = list(tab_cache["cwe_ids"])
    feature_cols = list(tab_cache.get("feature_columns", get_feature_columns()))

    byte_cache = _load_or_build_byte_cache(ALL_ROWS, DEFAULT_BYTE_LENGTH)
    X_byte = np.asarray(byte_cache["X_byte"], dtype=np.uint8)

    # CWE mapping from training labels only (same as train.py)
    train_pos_mask = (y_label == 1) & (np.arange(len(y_label)) < len(TRAIN_ROWS))
    train_pos_cwe = [cwe_ids_list[i] for i in range(min(len(TRAIN_ROWS), len(y_label))) if y_label[i] == 1]
    cwe_classes = sorted(set(train_pos_cwe))
    cwe_mapping = {name: idx for idx, name in enumerate(cwe_classes)}
    y_cwe = np.full(len(cwe_ids_list), -1, dtype=np.int32)
    for i, cid in enumerate(cwe_ids_list):
        if cid and cid in cwe_mapping:
            y_cwe[i] = cwe_mapping[cid]

    return X, X_byte, y_label, y_cwe, cwe_ids_list, cwe_classes, cwe_mapping, feature_cols


def _tabular_oof_predict(X, y_label, cwe_ids_list, train_idx, val_idx, cwe_classes, seed):
    """Train tabular models on train_idx only, predict on val_idx."""
    X_tr, X_val = X[train_idx], X[val_idx]
    y_tr_label, y_val_label = y_label[train_idx], y_label[val_idx]

    # Label model on train split only
    label_model = _build_label_ensemble(random_state=seed)
    label_model.fit(X_tr, y_tr_label)
    tree_lp = _aligned_positive_probability(label_model, X_val)

    # CWE model on train split only
    pos_tr = y_tr_label == 1
    cwe_tr = [cwe_ids_list[int(i)] for i in train_idx[pos_tr]]
    cwe_mapping_local = {name: idx for idx, name in enumerate(cwe_classes)}
    y_cwe_tr = np.array([cwe_mapping_local.get(c, 0) for c in cwe_tr], dtype=np.int32)

    class_weight_map = _cwe_class_weight_map(y_cwe_tr, len(cwe_classes))
    cwe_model = _build_cwe_gbdt(len(cwe_classes), class_weight_map, random_state=seed)
    cwe_model.fit(X_tr[pos_tr], y_cwe_tr)
    tree_cp = _aligned_cwe_probability(cwe_model, X_val, len(cwe_classes))

    return tree_lp, tree_cp


def run_oof():
    print("=== OOF Analysis v2.5 ===")
    print("Loading data...")
    X, X_byte, y_label, y_cwe, cwe_ids_list, cwe_classes, cwe_mapping, feature_cols = _build_data()

    # Find latest trained fusion config (v2.5 not trained yet, use v2.4)
    config_path = MODEL_DIR.resolve() / FUSION_CONFIG_NAME
    if not config_path.exists():
        config_path = MODEL_DIR.resolve() / "fusion_config_v2.4.json"
    fusion_config = json.load(open(config_path))
    print(f"Using config: {config_path.name}")
    seeds = fusion_config.get("seeds", [42, 123, 202])
    num_cwe = len(cwe_classes)

    all_binary_ids = []
    all_val_y_label = []
    all_val_y_cwe = []
    all_tree_lp = []
    all_tree_cp = []
    all_neural_lp = []
    all_neural_cp = []

    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        train_idx, val_idx = _split_indices(y_label, cwe_ids_list, rng_seed=seed)
        print(f"  val samples: {len(val_idx)}")

        # Tabular OOF
        tree_lp, tree_cp = _tabular_oof_predict(
            X, y_label, cwe_ids_list, train_idx, val_idx, cwe_classes, seed)
        print(f"  tabular done: label_pos={tree_lp.mean():.3f}")

        # Neural OOF (model was trained on train_idx only)
        nb_path = MODEL_DIR.resolve() / seed_neural_bundle(seed)
        if not nb_path.exists():
            nb_path = MODEL_DIR.resolve() / f"neural_bundle_v2.4_seed{seed}.pt"
        nb = torch.load(nb_path, map_location=DEVICE, weights_only=False)
        neural_model = ByteMetaMultiTaskNet(**nb["model_config"]).to(DEVICE)
        neural_model.load_state_dict(nb["state_dict"])
        neural_model.eval()
        normalizer = TabularNormalizer(
            mean=nb["normalizer"]["mean"].cpu().numpy().astype(np.float32),
            std=nb["normalizer"]["std"].cpu().numpy().astype(np.float32),
        )
        X_val_tab = apply_tabular_normalizer(X[val_idx], normalizer)
        neural_lp, neural_cp = predict_multitask(
            neural_model, X_byte[val_idx], X_val_tab,
            batch_size=128, device=DEVICE, desc=f"Neural OOF s={seed}")
        print(f"  neural done: label_pos={neural_lp.mean():.3f}")

        # Collect
        val_binary_ids = [ALL_ROWS[i]["binary_id"] for i in val_idx]
        all_binary_ids.extend(val_binary_ids)
        all_val_y_label.append(y_label[val_idx])
        all_val_y_cwe.append(y_cwe[val_idx])
        all_tree_lp.append(tree_lp)
        all_tree_cp.append(tree_cp)
        all_neural_lp.append(neural_lp)
        all_neural_cp.append(neural_cp)

    # Concatenate
    oof_y_label = np.concatenate(all_val_y_label)
    oof_y_cwe = np.concatenate(all_val_y_cwe)
    oof_tree_lp = np.concatenate(all_tree_lp)
    oof_tree_cp = np.concatenate(all_tree_cp)
    oof_neural_lp = np.concatenate(all_neural_lp)
    oof_neural_cp = np.concatenate(all_neural_cp)

    glw = float(fusion_config["scalar_neural_cwe_weight"])
    print(f"\nTotal OOF samples: {len(oof_y_label)} (label=1: {oof_y_label.sum()})")
    print(f"Global neural_cwe_weight: {glw:.4f}")

    # === Per-class analysis ===
    pos_mask = oof_y_label == 1
    pos_y_cwe = oof_y_cwe[pos_mask]

    # Fused CWE predictions (global weight)
    fused_cp = glw * oof_neural_cp + (1 - glw) * oof_tree_cp
    # Label fusion
    gllw = float(fusion_config["scalar_neural_label_weight"])
    fused_lp = gllw * oof_neural_lp + (1 - gllw) * oof_tree_lp
    label_thresh = float(fusion_config.get("fusion_threshold", 0.5))
    fused_label_pred = (fused_lp >= label_thresh).astype(int)

    # Per-class metrics
    class_metrics = []
    for c in range(num_cwe):
        c_mask = pos_y_cwe == c
        support = int(c_mask.sum())
        pred_c = fused_cp[pos_mask].argmax(axis=1) == c
        c_pred_mask = (fused_label_pred[pos_mask]) & (fused_cp[pos_mask].argmax(axis=1) == c)
        # For recall, use ground truth label=1 AND cwe=c
        true_c = pos_mask.copy()
        true_c[pos_mask] = c_mask
        tp_c = (fused_label_pred == 1) & (fused_cp.argmax(axis=1) == c) & (oof_y_cwe == c)
        prec_c = tp_c.sum() / max(c_pred_mask.sum(), 1)
        rec_c = (tp_c.sum()) / max(support, 1)
        f1_c = 2 * prec_c * rec_c / max(prec_c + rec_c, 1e-12)
        # Per-model F1
        tree_pred = oof_tree_cp[pos_mask].argmax(axis=1)
        neural_pred = oof_neural_cp[pos_mask].argmax(axis=1)
        tree_f1 = f1_score(c_mask, tree_pred == c, zero_division=0)
        neural_f1 = f1_score(c_mask, neural_pred == c, zero_division=0)
        class_metrics.append({
            "class": cwe_classes[c], "support": support,
            "precision": round(float(prec_c), 4), "recall": round(float(rec_c), 4),
            "f1": round(float(f1_c), 4),
            "tree_f1": round(float(tree_f1), 4), "neural_f1": round(float(neural_f1), 4),
        })

    # Sort by F1 (worst first)
    class_metrics.sort(key=lambda x: x["f1"])
    print("\n=== Bottom 10 CWE classes ===")
    for m in class_metrics[:10]:
        better = "neural" if m["neural_f1"] > m["tree_f1"] else "tree"
        print(f"  {m['class']}: F1={m['f1']:.4f} sup={m['support']} (tree={m['tree_f1']:.4f} neural={m['neural_f1']:.4f}) [{better}]")

    print("\n=== Top 10 CWE classes ===")
    for m in class_metrics[-10:]:
        better = "neural" if m["neural_f1"] > m["tree_f1"] else "tree"
        print(f"  {m['class']}: F1={m['f1']:.4f} sup={m['support']} [{better}]")

    # Confusion pairs
    conf_pairs = Counter()
    for i in range(len(pos_mask)):
        if pos_mask[i] and fused_label_pred[i] == 1:
            true_c = int(oof_y_cwe[i])
            pred_c = int(fused_cp[i].argmax())
            if true_c != pred_c:
                conf_pairs[(cwe_classes[true_c], cwe_classes[pred_c])] += 1

    print("\n=== Top 20 Confusion Pairs ===")
    for (true_c, pred_c), count in conf_pairs.most_common(20):
        print(f"  {true_c} -> {pred_c}: {count}")

    # Summary stats
    valid = [m for m in class_metrics if m["support"] > 0]
    macro_f1 = np.mean([m["f1"] for m in valid])
    macro_prec = np.mean([m["precision"] for m in valid])
    macro_rec = np.mean([m["recall"] for m in valid])
    neural_better = sum(1 for m in valid if m["neural_f1"] > m["tree_f1"])
    tree_better = sum(1 for m in valid if m["tree_f1"] > m["neural_f1"])

    report = {
        "model_version": "v2.5-oof",
        "num_oof_samples": int(len(oof_y_label)),
        "num_positive": int(oof_y_label.sum()),
        "global_neural_cwe_weight": float(glw),
        "macro_f1": round(float(macro_f1), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "neural_better_classes": neural_better,
        "tree_better_classes": tree_better,
        "per_class": class_metrics,
        "top_confusions": [{"true": t, "pred": p, "count": c} for (t, p), c in conf_pairs.most_common(30)],
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {OUTPUT_REPORT}")
    print(f"Macro-F1 (OOF, 87-class): {macro_f1:.4f}")
    print(f"Neural better: {neural_better}, Tree better: {tree_better}, Equal: {num_cwe - neural_better - tree_better}")


if __name__ == "__main__":
    run_oof()
