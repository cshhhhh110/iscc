"""v3.1 ablation: isolate ngram and meta-model contributions from cached OOF assets.

Usage: python ablation_v3.1.py [--ablation-path models/ablation_v3.1.joblib]

Reads OOF predictions + ngram features saved during training, then compares:

  1. Baseline:        argmax on OOF CWE probs (pure ensemble, no meta-model)
  2. Meta only:       LightGBM on base_X (ensemble preds + file_size, NO ngram)
  3. Full v3.1:       LightGBM on base_X + ngram_X (full meta-model)

Ngram contribution = f1_3 - f1_2  (what segment features add beyond meta correction)
Meta contribution  = f1_2 - f1_1  (what meta-model adds beyond raw argmax)
Total gain          = f1_3 - f1_1  (combined v3.1 improvement over baseline)

All 3 run in seconds since the heavy training is already cached.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

META_PARAMS = dict(
    objective="multiclass", num_leaves=15, max_depth=4,
    learning_rate=0.05, n_estimators=200,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, min_child_samples=20,
    random_state=42, n_jobs=4, verbose=-1,
)


def _cwe_macro_f1(y_true, y_pred):
    """Macro-F1 over CWE classes, excluding label=0 samples."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _train_and_eval(X_train, y_train, X_eval, y_eval, n_classes):
    model = LGBMClassifier(num_class=n_classes, **META_PARAMS)
    model.fit(X_train, y_train)
    pred = model.predict(X_eval)
    return _cwe_macro_f1(y_eval, pred), model


def main():
    parser = argparse.ArgumentParser(description="v3.1 ablation analysis")
    parser.add_argument("--ablation-path", default="models/ablation_v3.1.joblib")
    args = parser.parse_args()

    path = Path(args.ablation_path)
    if not path.exists():
        print(f"Ablation file not found: {path}")
        print("Run v3.1 training first to generate it.")
        return

    data = joblib.load(path)
    print("=== v3.1 Ablation Analysis ===")
    print(f"Loaded ablation assets: {list(data.keys())}")
    print(f"  OOF samples: {data['has_oof'].sum()}/{len(data['y_label'])}")
    print(f"  CWE classes: {len(data['cwe_classes'])}")

    # Extract data
    has_oof = data["has_oof"]
    n_orig = data["n_orig"]
    y_label = data["y_label"]
    y_cwe = data["y_cwe"]
    n_classes = len(data["cwe_classes"])

    # Filter: OOF + original + label=1
    pos_mask = np.zeros(len(y_label), dtype=bool)
    pos_mask[:n_orig] = True
    pos_mask &= has_oof
    pos_mask &= (y_label == 1)

    # Hold-out split: 80% train / 20% eval for ablation
    rng = np.random.RandomState(42)
    idx = rng.permutation(pos_mask.sum())
    n_train = int(len(idx) * 0.8)
    train_pos = np.where(pos_mask)[0][idx[:n_train]]
    eval_pos = np.where(pos_mask)[0][idx[n_train:]]

    print(f"  Ablation train: {len(train_pos)}, eval: {len(eval_pos)}")

    # Build feature sets
    base_X = np.hstack([
        data["oof_label_probs"].reshape(-1, 1),
        data["oof_cwe_probs"],
        data["file_sizes"].reshape(-1, 1),
    ]).astype(np.float32)

    ngram_X = data["ngram_matrix"].astype(np.float32)

    # 1. Baseline: argmax on raw OOF CWE probs (no meta-model at all)
    print("\n--- 1. Baseline (pure ensemble argmax, no meta) ---")
    y_eval = y_cwe[eval_pos]
    oof_cp = data["oof_cwe_probs"][eval_pos]
    y_pred_1 = oof_cp.argmax(axis=1)
    f1_1 = _cwe_macro_f1(y_eval, y_pred_1)
    print(f"  CWE Macro-F1: {f1_1:.4f}")

    # 2. Meta only: LightGBM on base_X (no ngram features)
    print("\n--- 2. Meta only (ensemble + file_size, NO ngram) ---")
    f1_2, model_2 = _train_and_eval(base_X[train_pos], y_cwe[train_pos],
                                      base_X[eval_pos], y_eval, n_classes)
    print(f"  CWE Macro-F1: {f1_2:.4f}  (Δ vs baseline: {f1_2 - f1_1:+.4f})")

    # 3. Full v3.1: LightGBM on base_X + ngram_X
    print("\n--- 3. Full v3.1 (ensemble + file_size + ngram) ---")
    X_full = np.hstack([base_X, ngram_X]).astype(np.float32)
    f1_3, model_3 = _train_and_eval(X_full[train_pos], y_cwe[train_pos],
                                      X_full[eval_pos], y_eval, n_classes)
    print(f"  CWE Macro-F1: {f1_3:.4f}  (Δ vs baseline: {f1_3 - f1_1:+.4f})")

    print("\n=== Summary ===")
    print(f"  Baseline (pure ensemble):    {f1_1:.4f}")
    print(f"  Meta only (no ngram):        {f1_2:.4f}  (meta contribution:    {f1_2 - f1_1:+.4f})")
    print(f"  Full v3.1 (ngram + meta):    {f1_3:.4f}  (ngram contribution:   {f1_3 - f1_2:+.4f})")
    print(f"  Total gain vs baseline:      {f1_3 - f1_1:+.4f}")


if __name__ == "__main__":
    main()
