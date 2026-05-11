from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from common import (
    DEFAULT_SEED,
    ID_COL,
    LABEL_COL,
    append_action_log,
    classification_summary,
    encode_labels,
    ensure_feature_columns,
    make_feature_frame,
    read_table,
    validate_prediction_frame,
)


def build_hgb(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    learning_rate=0.03,
                    max_iter=420,
                    max_leaf_nodes=63,
                    max_depth=None,
                    min_samples_leaf=12,
                    l2_regularization=0.1,
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=25,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_extra_trees(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    criterion="gini",
                    max_features="sqrt",
                    min_samples_leaf=2,
                    min_samples_split=2,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_logistic(seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.5,
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=4000,
                    n_jobs=-1,
                    random_state=seed,
                ),
            ),
        ]
    )


def build_lda(seed: int | None = None) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
    )


def build_model_specs():
    return [
        ("hgb", build_hgb),
        ("et", build_extra_trees),
        ("lr", build_logistic),
        ("lda", build_lda),
    ]


def stack_probabilities(probabilities_by_model: dict[str, np.ndarray], model_names: list[str]) -> np.ndarray:
    return np.concatenate([probabilities_by_model[name] for name in model_names], axis=1)


def build_meta_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    multi_class="multinomial",
                    max_iter=4000,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train stacked ensemble for the competition.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle.joblib")
    parser.add_argument("--submission-path", type=Path, default=default_root / "提交结果" / "submission.csv")
    parser.add_argument("--report-path", type=Path, default=default_root / "模型" / "validation_report.json")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file
    sample_path = data_dir / args.sample_file

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    args.submission_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)

    append_action_log(args.action_log, "Training started.")
    train_df = read_table(train_path)
    test_df = read_table(test_path)
    sample_df = read_table(sample_path)
    feature_cols = ensure_feature_columns(train_df, test_df)

    if LABEL_COL not in train_df.columns:
        raise ValueError(f"{train_path} does not contain label column {LABEL_COL!r}")
    if ID_COL not in train_df.columns or ID_COL not in test_df.columns:
        raise ValueError("train/test data must contain id column")

    x_train = make_feature_frame(train_df, feature_cols)
    x_test = make_feature_frame(test_df, feature_cols)
    label_encoder, y = encode_labels(train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()

    append_action_log(
        args.action_log,
        f"Loaded data: train_rows={len(train_df)}, test_rows={len(test_df)}, "
        f"features={len(feature_cols)}, labels={len(label_names)}.",
    )

    model_specs = build_model_specs()
    model_names = [name for name, _ in model_specs]
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    n_classes = len(label_names)

    oof_probs = {name: np.zeros((len(train_df), n_classes), dtype=np.float32) for name in model_names}

    for fold_idx, (tr_idx, va_idx) in enumerate(
        tqdm(cv.split(x_train, y), total=args.folds, desc="CV folds", unit="fold", dynamic_ncols=True),
        start=1,
    ):
        fold_seed = args.seed + fold_idx * 1000
        x_tr = x_train.iloc[tr_idx]
        y_tr = y[tr_idx]
        x_va = x_train.iloc[va_idx]

        for model_offset, (name, builder) in enumerate(model_specs, start=1):
            model = builder(fold_seed + model_offset)
            model.fit(x_tr, y_tr)
            oof_probs[name][va_idx] = model.predict_proba(x_va).astype(np.float32)

        fold_avg = np.mean([oof_probs[name][va_idx] for name in model_names], axis=0)
        fold_pred = np.argmax(fold_avg, axis=1)
        fold_macro_f1 = classification_summary(y[va_idx], fold_pred, label_names)["macro_f1"]
        tqdm.write(f"Fold {fold_idx}/{args.folds} collected OOF predictions: macro_f1={fold_macro_f1:.6f}")

    stack_train = stack_probabilities(oof_probs, model_names)
    meta_model = build_meta_model()
    meta_model.fit(stack_train, y)
    meta_pred = meta_model.predict(stack_train)

    base_reports = {
        name: classification_summary(y, np.argmax(oof_probs[name], axis=1), label_names)
        for name in model_names
    }
    meta_report = classification_summary(y, meta_pred, label_names)

    base_models: dict[str, Pipeline] = {}
    for model_offset, (name, builder) in enumerate(model_specs, start=1):
        full_seed = args.seed + 10000 + model_offset
        model = builder(full_seed)
        model.fit(x_train, y)
        base_models[name] = model

    stack_test = stack_probabilities(
        {name: base_models[name].predict_proba(x_test).astype(np.float32) for name in model_names},
        model_names,
    )
    test_probs = meta_model.predict_proba(stack_test)
    test_pred = np.argmax(test_probs, axis=1)
    test_labels = label_encoder.inverse_transform(test_pred)

    report = {
        **meta_report,
        "seed": args.seed,
        "folds": args.folds,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "label_count": int(len(label_names)),
        "label_names": label_names,
        "base_models": model_names,
        "meta_model": "logistic_regression",
        "base_model_reports": base_reports,
    }

    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: test_labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))
    submission.to_csv(args.submission_path, index=False, encoding="utf-8")

    bundle = {
        "schema_version": 2,
        "python_version": sys.version,
        "seed": args.seed,
        "folds": args.folds,
        "feature_columns": feature_cols,
        "label_names": label_names,
        "base_model_names": model_names,
        "base_models": base_models,
        "meta_model": meta_model,
        "validation": report,
    }
    joblib.dump(bundle, args.model_path, compress=3)

    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"Training completed: meta_macro_f1={report['macro_f1']:.6f}, "
        f"accuracy={report['accuracy']:.6f}, submission={args.submission_path}, model={args.model_path}.",
    )
    print(json.dumps({"macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}, indent=2))
    print(f"Wrote submission: {args.submission_path}")
    print(f"Wrote model: {args.model_path}")
    print(f"Wrote report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
