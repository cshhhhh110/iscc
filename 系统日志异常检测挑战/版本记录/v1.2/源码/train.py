from __future__ import annotations

import argparse
import copy
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from common import (
    ANOMALY_TYPES,
    BINARY_CLASSES,
    DEFAULT_SEED,
    FEATURE_CONFIG,
    TYPE_SPAN_LENGTHS,
    append_action_log,
    best_threshold_for_f1,
    decode_predictions,
    evaluate_predictions,
    flatten_line_labels_by_type,
    flatten_line_scores_for_type,
    make_doc_feature_matrix,
    make_line_feature_matrix,
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


def calibrate_model(model: SGDClassifier, x_val, y_val) -> CalibratedClassifierCV:
    cal = CalibratedClassifierCV(model, method="isotonic", cv="prefit")
    cal.fit(x_val, y_val)
    return cal


def partial_fit_binary(model: SGDClassifier, x_batch, y_batch, sample_weight) -> None:
    if hasattr(model, "classes_"):
        model.partial_fit(x_batch, y_batch, sample_weight=sample_weight)
    else:
        model.partial_fit(x_batch, y_batch, classes=BINARY_CLASSES, sample_weight=sample_weight)


def make_sample_weight(y: np.ndarray, weights: tuple[float, float]) -> np.ndarray:
    neg_weight, pos_weight = weights
    return np.where(y == 1, pos_weight, neg_weight).astype(np.float32)


def class_weights_from_counts(pos: int, neg: int, cap: float) -> tuple[float, float]:
    if pos <= 0 or neg <= 0:
        return 1.0, 1.0
    pos_weight = max(1.0, np.sqrt(neg / pos))
    neg_weight = max(1.0, np.sqrt(pos / neg))
    return min(neg_weight, cap), min(pos_weight, cap)


def estimate_training_weights(docs):
    doc_pos = sum(doc.has_anomaly for doc in docs)
    doc_neg = len(docs) - doc_pos
    doc_weights = class_weights_from_counts(doc_pos, doc_neg, cap=5.0)

    total_lines = sum(len(doc.lines) for doc in docs)
    type_pos = {label: 0 for label in ANOMALY_TYPES}
    for doc in docs:
        seen = {label: set() for label in ANOMALY_TYPES}
        n_lines = len(doc.lines)
        for span in doc.spans:
            if span.label not in seen or n_lines == 0:
                continue
            start = max(0, min(n_lines - 1, span.start))
            end = max(0, min(n_lines - 1, span.end))
            if start <= end:
                seen[span.label].update(range(start, end + 1))
        for label in ANOMALY_TYPES:
            type_pos[label] += len(seen[label])

    type_weights = {
        label: class_weights_from_counts(pos, total_lines - pos, cap=25.0)
        for label, pos in type_pos.items()
    }
    return doc_weights, type_weights


def train_one_epoch(docs, doc_model, type_models, doc_weights, type_weights, batch_size, rng, desc):
    order = np.arange(len(docs))
    rng.shuffle(order)
    ordered_docs = [docs[int(i)] for i in order]
    batches = [ordered_docs[start: start + batch_size] for start in range(0, len(ordered_docs), batch_size)]
    for batch_docs in tqdm(batches, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
        parsed = parse_documents_batch(batch_docs)
        doc_matrix, y_doc = make_doc_feature_matrix(parsed)
        partial_fit_binary(doc_model, doc_matrix, y_doc, make_sample_weight(y_doc, doc_weights))

        line_matrix, line_targets, _ = make_line_feature_matrix(parsed, include_targets=True)
        for label, model in type_models.items():
            y_type = line_targets[label]
            partial_fit_binary(model, line_matrix, y_type, make_sample_weight(y_type, type_weights[label]))


def train_fold_sgd(train_docs, val_docs, args, fold_idx: int):
    seed = args.seed + fold_idx * 1000
    rng = np.random.default_rng(seed)

    doc_model = build_sgd(seed, alpha=args.doc_alpha)
    type_models = {
        label: build_sgd(seed + 10 + idx, alpha=args.line_alpha)
        for idx, label in enumerate(ANOMALY_TYPES)
    }
    doc_weights, type_weights = estimate_training_weights(train_docs)

    best_score = -1.0
    best_epoch = 0
    best_doc_model = None
    best_type_models = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_one_epoch(
            train_docs, doc_model, type_models, doc_weights, type_weights,
            args.batch_size, rng,
            desc=f"fold {fold_idx} epoch {epoch}/{args.epochs}",
        )
        doc_probs, line_scores = predict_raw_scores(
            val_docs, doc_model, type_models,
            batch_size=args.batch_size,
            desc=f"fold {fold_idx} validate epoch {epoch}",
        )
        y_val_doc = np.array([d.has_anomaly for d in val_docs], dtype=np.int32)
        _, doc_f1 = best_threshold_for_f1(y_val_doc, doc_probs, average="macro")

        val_line_labels = flatten_line_labels_by_type(val_docs)
        line_f1s = {}
        for type_idx, label in enumerate(ANOMALY_TYPES):
            probs = flatten_line_scores_for_type(line_scores, type_idx)
            _, score = best_threshold_for_f1(val_line_labels[label], probs, average="binary")
            line_f1s[label] = score
        mean_line_f1 = float(np.mean(list(line_f1s.values()))) if line_f1s else 0.0
        blended = 0.35 * doc_f1 + 0.65 * mean_line_f1

        if blended > best_score + 1e-5:
            best_score = blended
            best_epoch = epoch
            best_doc_model = copy.deepcopy(doc_model)
            best_type_models = {label: copy.deepcopy(model) for label, model in type_models.items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    doc_probs, line_scores = predict_raw_scores(
        val_docs, best_doc_model, best_type_models,
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
    }


def train_full_model_sgd(train_docs, args, epochs: int):
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
            train_docs, doc_model, type_models, doc_weights, type_weights,
            args.batch_size, rng,
            desc=f"full train epoch {epoch}/{epochs}",
        )
    return doc_model, type_models


def tune_thresholds_expanded(train_docs, oof_doc_probs, oof_line_scores) -> tuple[dict, dict]:
    """Three-stage threshold search: coarse random → local refine → final fine-tune."""
    y_doc = np.array([doc.has_anomaly for doc in train_docs], dtype=np.int32)
    line_labels = flatten_line_labels_by_type(train_docs)

    base_type_thresholds: dict[str, float] = {}
    type_reports: dict[str, dict] = {}
    for type_idx, label in enumerate(ANOMALY_TYPES):
        probs = flatten_line_scores_for_type(oof_line_scores, type_idx)
        threshold, line_f1 = best_threshold_for_f1(line_labels[label], probs, average="binary")
        base_type_thresholds[label] = float(threshold)
        type_reports[label] = {"threshold": float(threshold), "line_f1": float(line_f1)}

    base_doc_threshold, base_doc_f1 = best_threshold_for_f1(y_doc, oof_doc_probs, average="macro")

    if len(train_docs) > 5000:
        rng = np.random.default_rng(DEFAULT_SEED)
        pos_idx = np.flatnonzero(y_doc == 1)
        neg_idx = np.flatnonzero(y_doc == 0)
        pos_take = min(2500, len(pos_idx))
        neg_take = min(2500, len(neg_idx))
        sampled = np.concatenate([
            rng.choice(pos_idx, size=pos_take, replace=False) if pos_take else np.zeros(0, dtype=np.int64),
            rng.choice(neg_idx, size=neg_take, replace=False) if neg_take else np.zeros(0, dtype=np.int64),
        ])
        search_indices = np.sort(sampled.astype(np.int64, copy=False))
    else:
        search_indices = np.arange(len(train_docs), dtype=np.int64)

    search_docs = [train_docs[int(i)] for i in search_indices]
    search_doc_probs = oof_doc_probs[search_indices]
    search_line_scores = [oof_line_scores[int(i)] for i in search_indices]

    decoder_grid = [
        (1, 0, 2, 8, 2), (1, 0, 2, 10, 2), (1, 1, 2, 10, 2), (1, 1, 2, 12, 2),
        (3, 0, 2, 8, 2), (3, 0, 2, 10, 2), (3, 1, 2, 10, 2), (3, 1, 2, 12, 2),
        (3, 1, 3, 10, 2), (3, 1, 3, 12, 2), (3, 2, 3, 10, 2), (3, 2, 3, 12, 2),
        (5, 1, 3, 10, 2), (5, 1, 3, 12, 2), (5, 2, 3, 12, 2),
        (3, 1, 2, 10, 1), (3, 1, 3, 12, 1), (5, 2, 3, 12, 1),
    ]

    rng = np.random.default_rng(DEFAULT_SEED + 777)

    def make_config(doc_thr, type_thrs, sw, gm, msl, mxl, ms):
        return {
            "doc_threshold": float(doc_thr),
            "type_thresholds": {label: float(type_thrs[label]) for label in ANOMALY_TYPES},
            "type_threshold_offset": 0.0,
            "type_span_lengths": TYPE_SPAN_LENGTHS,
            "smoothing_window": int(sw),
            "gap_merge": int(gm),
            "min_span_len": int(msl),
            "max_span_len": int(mxl),
            "max_spans": int(ms),
        }

    def random_type_thresholds():
        return {label: float(np.clip(rng.uniform(0.1, 0.98), 0.02, 0.98)) for label in ANOMALY_TYPES}

    def random_decoder():
        return decoder_grid[rng.integers(0, len(decoder_grid))]

    best_config = None
    best_score = -1.0
    all_results: list[tuple[float, dict]] = []

    for _ in tqdm(range(200), desc="search stage1", unit="trial", dynamic_ncols=True):
        doc_thr = float(np.clip(rng.uniform(0.1, 0.9), 0.02, 0.98))
        type_thrs = random_type_thresholds()
        sw, gm, msl, mxl, ms = random_decoder()
        cfg = make_config(doc_thr, type_thrs, sw, gm, msl, mxl, ms)
        predictions = decode_predictions(search_docs, search_doc_probs, search_line_scores, cfg)
        metrics = evaluate_predictions(search_docs, predictions)
        all_results.append((metrics["score"], cfg))
        if metrics["score"] > best_score + 1e-12:
            best_score = metrics["score"]
            best_config = cfg

    all_results.sort(key=lambda x: -x[0])
    top_configs = [cfg for _, cfg in all_results[:10]]

    for _ in tqdm(range(100), desc="search stage2", unit="trial", dynamic_ncols=True):
        base = top_configs[rng.integers(0, len(top_configs))]
        doc_thr = float(np.clip(base["doc_threshold"] + rng.uniform(-0.04, 0.04), 0.02, 0.98))
        type_thrs = {
            label: float(np.clip(base["type_thresholds"][label] + rng.uniform(-0.04, 0.04), 0.02, 0.98))
            for label in ANOMALY_TYPES
        }
        sw = int(np.clip(base["smoothing_window"] + rng.integers(-1, 2), 1, 5))
        gm = int(np.clip(base["gap_merge"] + rng.integers(-1, 2), 0, 3))
        msl = int(np.clip(base["min_span_len"] + rng.integers(-1, 2), 2, 5))
        mxl = int(np.clip(base["max_span_len"] + rng.integers(-1, 2), msl, 15))
        ms = int(np.clip(base["max_spans"] + rng.integers(0, 2), 1, 3))
        cfg = make_config(doc_thr, type_thrs, sw, gm, msl, mxl, ms)
        predictions = decode_predictions(search_docs, search_doc_probs, search_line_scores, cfg)
        metrics = evaluate_predictions(search_docs, predictions)
        if metrics["score"] > best_score + 1e-12:
            best_score = metrics["score"]
            best_config = cfg

    for _ in tqdm(range(50), desc="search stage3", unit="trial", dynamic_ncols=True):
        doc_thr = float(np.clip(best_config["doc_threshold"] + rng.uniform(-0.015, 0.015), 0.02, 0.98))
        type_thrs = {
            label: float(np.clip(best_config["type_thresholds"][label] + rng.uniform(-0.015, 0.015), 0.02, 0.98))
            for label in ANOMALY_TYPES
        }
        sw = int(np.clip(best_config["smoothing_window"] + rng.integers(-1, 2), 1, 5))
        gm = int(np.clip(best_config["gap_merge"] + rng.integers(0, 2), 0, 3))
        msl = int(np.clip(best_config["min_span_len"] + rng.integers(0, 2), 2, 5))
        mxl = int(np.clip(best_config["max_span_len"] + rng.integers(0, 2), msl, 15))
        ms = int(np.clip(best_config["max_spans"] + rng.integers(0, 2), 1, 3))
        cfg = make_config(doc_thr, type_thrs, sw, gm, msl, mxl, ms)
        predictions = decode_predictions(search_docs, search_doc_probs, search_line_scores, cfg)
        metrics = evaluate_predictions(search_docs, predictions)
        if metrics["score"] > best_score + 1e-12:
            best_score = metrics["score"]
            best_config = cfg

    if best_config is None:
        raise RuntimeError("Decoder threshold search failed.")

    full_predictions = decode_predictions(train_docs, oof_doc_probs, oof_line_scores, best_config)
    full_metrics = evaluate_predictions(train_docs, full_predictions)

    report = {
        "base_doc_threshold": float(base_doc_threshold),
        "base_doc_f1": float(base_doc_f1),
        "type_thresholds": type_reports,
        "decoder_search": {
            "search_rows": int(len(search_docs)),
            "total_trials": 350,
            "best_score": float(best_score),
            "best_metrics": evaluate_predictions(search_docs, decode_predictions(
                search_docs, search_doc_probs, search_line_scores, best_config
            )),
            "full_metrics": full_metrics,
        },
        "selected": best_config,
    }
    return best_config, report


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
    parser.add_argument("--dev-limit", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_path = data_dir / args.train_file
    test_path = data_dir / args.test_file
    sample_path = data_dir / args.sample_file

    append_action_log(args.action_log, "Training started for system log anomaly detection (v1.2 SGD+Calibration).")
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
        result = train_fold_sgd(train_fold_docs, val_fold_docs, args, fold_idx)
        best_epochs.append(result["best_epoch"])

        # Calibrate on validation set
        val_parsed = parse_documents_batch(val_fold_docs)
        val_doc_matrix, val_y_doc = make_doc_feature_matrix(val_parsed)
        doc_model_cal = calibrate_model(result["doc_model"], val_doc_matrix, val_y_doc)

        val_line_matrix, val_line_targets, _ = make_line_feature_matrix(val_parsed, include_targets=True)
        type_models_cal = {}
        for label in ANOMALY_TYPES:
            type_models_cal[label] = calibrate_model(
                result["type_models"][label], val_line_matrix, val_line_targets[label]
            )

        # Re-predict with calibrated models
        doc_probs, line_scores = predict_raw_scores(
            val_fold_docs, doc_model_cal, type_models_cal,
            batch_size=args.batch_size,
            desc=f"fold {fold_idx} cal predict",
        )
        oof_doc_probs[va_idx] = doc_probs
        for local_pos, global_idx in enumerate(va_idx):
            oof_line_scores[int(global_idx)] = line_scores[local_pos]

        fold_reports.append({
            "fold": fold_idx,
            "train_rows": len(train_fold_docs),
            "valid_rows": len(val_fold_docs),
            "best_epoch": int(result["best_epoch"]),
            "best_early_score": float(result["best_score"]),
        })
        append_action_log(args.action_log, f"Fold {fold_idx} completed: best_epoch={result['best_epoch']}.")

    if any(scores is None for scores in oof_line_scores):
        raise RuntimeError("OOF line score collection failed.")
    oof_line_scores_final = [scores for scores in oof_line_scores if scores is not None]

    thresholds, threshold_report = tune_thresholds_expanded(train_docs, oof_doc_probs, oof_line_scores_final)
    oof_predictions = decode_predictions(train_docs, oof_doc_probs, oof_line_scores_final, thresholds)
    oof_metrics = evaluate_predictions(train_docs, oof_predictions)
    final_epochs = max(1, min(args.epochs, int(round(float(np.median(best_epochs))))))

    append_action_log(
        args.action_log,
        f"OOF score={oof_metrics['score']:.6f}, final_epochs={final_epochs}, doc_threshold={thresholds['doc_threshold']:.4f}.",
    )

    doc_model, type_models = train_full_model_sgd(train_docs, args, epochs=final_epochs)
    bundle = {
        "version": 2,
        "model": "SGD+Calibrated",
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
        test_docs, doc_model, type_models,
        batch_size=args.batch_size,
        desc="predict test",
    )
    test_predictions = decode_predictions(test_docs, test_doc_probs, test_line_scores, thresholds)
    write_submission(args.submission_path, test_predictions)
    sample_columns = read_sample_columns(sample_path)
    validation_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    report = {
        "version": 2,
        "model": "SGD+Calibrated",
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
