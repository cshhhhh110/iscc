from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from common import (
    ANOMALY_TYPES,
    BINARY_CLASSES,
    DEFAULT_SEED,
    FEATURE_CONFIG,
    append_action_log,
    best_threshold_for_f1,
    decode_predictions,
    estimate_training_weights,
    evaluate_predictions,
    flatten_line_labels_by_type,
    flatten_line_scores_for_type,
    make_doc_feature_matrix,
    make_line_feature_matrix,
    make_sample_weight,
    parse_documents_batch,
    predict_raw_scores,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_json,
    write_submission,
)


def build_sgd(seed: int, alpha: float) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        fit_intercept=True,
        learning_rate="optimal",
        average=True,
        shuffle=False,
        random_state=seed,
    )


def partial_fit_binary(model: SGDClassifier, x_batch, y_batch, sample_weight) -> None:
    if hasattr(model, "classes_"):
        model.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
    else:
        model.partial_fit(x_batch, y_batch, classes=BINARY_CLASSES, sample_weight=sample_weight)


def train_one_epoch(
    docs,
    doc_model: SGDClassifier,
    type_models: dict[str, SGDClassifier],
    doc_weights: tuple[float, float],
    type_weights: dict[str, tuple[float, float]],
    batch_size: int,
    rng: np.random.Generator,
    desc: str,
) -> None:
    order = np.arange(len(docs))
    rng.shuffle(order)
    ordered_docs = [docs[int(i)] for i in order]
    batches = [ordered_docs[start : start + batch_size] for start in range(0, len(ordered_docs), batch_size)]
    for batch_docs in tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
        parsed = parse_documents_batch(batch_docs)
        doc_matrix, y_doc = make_doc_feature_matrix(parsed)
        partial_fit_binary(doc_model, doc_matrix, y_doc, make_sample_weight(y_doc, doc_weights))

        line_matrix, line_targets, _ = make_line_feature_matrix(parsed, include_targets=True)
        assert line_targets is not None
        for label, model in type_models.items():
            y_type = line_targets[label]
            partial_fit_binary(model, line_matrix, y_type, make_sample_weight(y_type, type_weights[label]))


def fold_early_score(val_docs, doc_probs, line_scores_by_doc, val_line_labels) -> dict:
    y_doc = np.array([doc.has_anomaly for doc in val_docs], dtype=np.int32)
    _, doc_f1 = best_threshold_for_f1(y_doc, doc_probs, average="macro")
    line_f1s = {}
    for type_idx, label in enumerate(ANOMALY_TYPES):
        probs = flatten_line_scores_for_type(line_scores_by_doc, type_idx)
        _, score = best_threshold_for_f1(val_line_labels[label], probs, average="binary")
        line_f1s[label] = score
    mean_line_f1 = float(np.mean(list(line_f1s.values()))) if line_f1s else 0.0
    blended = 0.35 * doc_f1 + 0.65 * mean_line_f1
    return {"score": blended, "doc_f1": doc_f1, "mean_line_f1": mean_line_f1}


def train_fold(train_docs, val_docs, args, fold_idx: int):
    seed = args.seed + fold_idx * 1000
    rng = np.random.default_rng(seed)
    doc_model = build_sgd(seed, alpha=args.doc_alpha)
    type_models = {
        label: build_sgd(seed + 10 + idx, alpha=args.line_alpha)
        for idx, label in enumerate(ANOMALY_TYPES)
    }
    doc_weights, type_weights = estimate_training_weights(train_docs)
    val_line_labels = flatten_line_labels_by_type(val_docs)

    best_score = -1.0
    best_epoch = 0
    best_doc_model = None
    best_type_models = None
    patience_left = args.patience
    epoch_reports = []

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            train_docs,
            doc_model,
            type_models,
            doc_weights,
            type_weights,
            args.batch_size,
            rng,
            desc=f"fold {fold_idx} epoch {epoch}/{args.epochs}",
        )
        doc_probs, line_scores = predict_raw_scores(
            val_docs,
            doc_model,
            type_models,
            batch_size=args.batch_size,
            desc=f"fold {fold_idx} validate epoch {epoch}",
        )
        score_info = fold_early_score(val_docs, doc_probs, line_scores, val_line_labels)
        score_info["epoch"] = epoch
        epoch_reports.append(score_info)

        if score_info["score"] > best_score + 1e-5:
            best_score = score_info["score"]
            best_epoch = epoch
            best_doc_model = copy.deepcopy(doc_model)
            best_type_models = {label: copy.deepcopy(model) for label, model in type_models.items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    assert best_doc_model is not None and best_type_models is not None
    doc_probs, line_scores = predict_raw_scores(
        val_docs,
        best_doc_model,
        best_type_models,
        batch_size=args.batch_size,
        desc=f"fold {fold_idx} best validate",
    )
    return {
        "doc_model": best_doc_model,
        "type_models": best_type_models,
        "doc_probs": doc_probs,
        "line_scores": line_scores,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "epoch_reports": epoch_reports,
    }


def tune_thresholds(train_docs, oof_doc_probs, oof_line_scores) -> tuple[dict, dict]:
    y_doc = np.array([doc.has_anomaly for doc in train_docs], dtype=np.int32)
    doc_threshold, doc_f1 = best_threshold_for_f1(y_doc, oof_doc_probs, average="macro")
    line_labels = flatten_line_labels_by_type(train_docs)
    type_thresholds = {}
    type_reports = {}
    for type_idx, label in enumerate(ANOMALY_TYPES):
        probs = flatten_line_scores_for_type(oof_line_scores, type_idx)
        threshold, line_f1 = best_threshold_for_f1(line_labels[label], probs, average="binary")
        type_thresholds[label] = float(threshold)
        type_reports[label] = {"threshold": float(threshold), "line_f1": float(line_f1)}

    thresholds = {
        "doc_threshold": float(doc_threshold),
        "type_thresholds": type_thresholds,
        "smoothing_window": 3,
        "gap_merge": 1,
        "min_span_len": 2,
        "max_span_len": 12,
        "max_spans": 2,
    }
    report = {"doc_threshold": float(doc_threshold), "doc_f1": float(doc_f1), "type_thresholds": type_reports}
    return thresholds, report


def train_full_model(train_docs, args, epochs: int):
    seed = args.seed + 90000
    rng = np.random.default_rng(seed)
    doc_model = build_sgd(seed, alpha=args.doc_alpha)
    type_models = {
        label: build_sgd(seed + 10 + idx, alpha=args.line_alpha)
        for idx, label in enumerate(ANOMALY_TYPES)
    }
    doc_weights, type_weights = estimate_training_weights(train_docs)
    for epoch in range(1, epochs + 1):
        train_one_epoch(
            train_docs,
            doc_model,
            type_models,
            doc_weights,
            type_weights,
            args.batch_size,
            rng,
            desc=f"full train epoch {epoch}/{epochs}",
        )
    return doc_model, type_models


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train line-level multilabel baseline for ISCC log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle.joblib")
    parser.add_argument("--submission-path", type=Path, default=default_root / "提交结果" / "submission.csv")
    parser.add_argument("--report-path", type=Path, default=default_root / "模型" / "validation_report.json")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--doc-alpha", type=float, default=2e-6)
    parser.add_argument("--line-alpha", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-limit", type=int, default=0, help="Optional smoke-test row limit; 0 means full data.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file
    sample_path = data_dir / args.sample_file

    append_action_log(args.action_log, "Training started for system log anomaly detection.")
    train_docs = read_documents(train_path, expect_labels=True)
    test_docs = read_documents(test_path, expect_labels=False)
    if args.dev_limit > 0:
        train_docs = train_docs[: args.dev_limit]

    if args.folds < 2:
        raise ValueError("--folds must be at least 2 for OOF threshold search")

    y = np.array([doc.has_anomaly for doc in train_docs], dtype=np.int32)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_doc_probs = np.zeros(len(train_docs), dtype=np.float32)
    oof_line_scores: list[np.ndarray | None] = [None] * len(train_docs)
    fold_reports = []
    best_epochs = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        tqdm(cv.split(np.zeros(len(train_docs)), y), total=args.folds, desc="CV folds", unit="fold", dynamic_ncols=True),
        start=1,
    ):
        train_fold_docs = [train_docs[int(i)] for i in tr_idx]
        val_fold_docs = [train_docs[int(i)] for i in va_idx]
        result = train_fold(train_fold_docs, val_fold_docs, args, fold_idx)
        best_epochs.append(result["best_epoch"])
        oof_doc_probs[va_idx] = result["doc_probs"]
        for local_pos, global_idx in enumerate(va_idx):
            oof_line_scores[int(global_idx)] = result["line_scores"][local_pos]
        fold_reports.append(
            {
                "fold": fold_idx,
                "train_rows": len(train_fold_docs),
                "valid_rows": len(val_fold_docs),
                "best_epoch": int(result["best_epoch"]),
                "best_early_score": float(result["best_score"]),
                "epoch_reports": result["epoch_reports"],
            }
        )
        append_action_log(args.action_log, f"Fold {fold_idx} completed: best_epoch={result['best_epoch']}.")

    if any(scores is None for scores in oof_line_scores):
        raise RuntimeError("OOF line score collection failed.")
    oof_line_scores_final = [scores for scores in oof_line_scores if scores is not None]

    thresholds, threshold_report = tune_thresholds(train_docs, oof_doc_probs, oof_line_scores_final)
    oof_predictions = decode_predictions(train_docs, oof_doc_probs, oof_line_scores_final, thresholds)
    oof_metrics = evaluate_predictions(train_docs, oof_predictions)
    final_epochs = max(1, min(args.epochs, int(round(float(np.median(best_epochs))))))

    append_action_log(
        args.action_log,
        f"OOF score={oof_metrics['score']:.6f}, final_epochs={final_epochs}, doc_threshold={thresholds['doc_threshold']:.4f}.",
    )

    doc_model, type_models = train_full_model(train_docs, args, epochs=final_epochs)
    bundle = {
        "version": 1,
        "seed": args.seed,
        "feature_config": FEATURE_CONFIG,
        "labels": ANOMALY_TYPES,
        "doc_model": doc_model,
        "type_models": type_models,
        "thresholds": thresholds,
        "final_epochs": final_epochs,
        "oof_metrics": oof_metrics,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)

    test_doc_probs, test_line_scores = predict_raw_scores(
        test_docs,
        doc_model,
        type_models,
        batch_size=args.batch_size,
        desc="predict test",
    )
    test_predictions = decode_predictions(test_docs, test_doc_probs, test_line_scores, thresholds)
    write_submission(args.submission_path, test_predictions)
    sample_columns = read_sample_columns(sample_path)
    validation_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    report = {
        "train_rows": len(train_docs),
        "test_rows": len(test_docs),
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "best_epochs": best_epochs,
        "final_epochs": final_epochs,
        "threshold_report": threshold_report,
        "thresholds": thresholds,
        "oof_metrics": oof_metrics,
        "fold_reports": fold_reports,
        "submission_validation": validation_info,
    }
    write_json(args.report_path, report)
    append_action_log(args.action_log, f"Training completed: model={args.model_path}, submission={args.submission_path}.")
    print("Training completed")
    print(f"OOF score: {oof_metrics['score']:.6f}")
    print(f"OOF detect F1: {oof_metrics['f1_detect']:.6f}")
    print(f"OOF loc IoU: {oof_metrics['iou_loc']:.6f}")
    print(f"OOF type F1: {oof_metrics['f1_type']:.6f}")
    print(f"Model: {args.model_path}")
    print(f"Submission: {args.submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
