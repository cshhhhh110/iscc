"""Train BiLSTM sequence labeling model for log anomaly detection.

Uses pre-built dense features (from build_dense_features.py).
5-fold CV → OOF scoring → full retrain → submission.
"""

from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    ANOMALY_TYPES,
    DEFAULT_SEED,
    Prediction,
    TYPE_SPAN_LENGTHS,
    append_action_log,
    evaluate_predictions,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_json,
    write_submission,
)
from model_nn_v2 import LABEL_TO_TYPE, NUM_LABELS, O_LABEL, LogBiLSTM, collate_batch


class DocDataset(Dataset):
    """Dataset wrapper for dense document features."""

    def __init__(self, data: list[dict]):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]


def _compute_class_weights(dense_train: list[dict]) -> tuple[torch.Tensor, float]:
    """Compute per-line class weights and doc positive weight."""
    all_labels = np.concatenate([item["labels"] for item in dense_train])
    counts = np.bincount(all_labels, minlength=NUM_LABELS).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    weights = torch.from_numpy(weights.astype(np.float32))

    has_anom = np.array([item["has_anomaly"] for item in dense_train], dtype=np.float32)
    pos = has_anom.sum()
    neg = len(has_anom) - pos
    doc_pos_weight = float(neg / max(1, pos))

    return weights, doc_pos_weight


def _model_predictions(model: LogBiLSTM, loader: DataLoader, device: torch.device) -> tuple[list[dict], float]:
    """Generate predictions from model. Returns (prediction_dicts, val_loss)."""
    model.eval()
    all_preds: list[dict] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]

            line_logits, doc_logits = model(features, mask)
            line_preds = torch.argmax(line_logits, dim=-1) * mask.long()
            doc_probs = torch.sigmoid(doc_logits).squeeze(-1)

            if "labels" in batch:
                labels = batch["labels"].to(device)
                has_anom = batch["has_anomaly"].to(device)
                line_loss = F.cross_entropy(
                    line_logits.permute(0, 2, 1), labels, reduction="sum"
                ) / mask.sum().float()
                doc_loss = F.binary_cross_entropy_with_logits(
                    doc_logits.squeeze(-1), has_anom, reduction="mean"
                )
                total_loss += (line_loss + 0.5 * doc_loss).item()
                n_batches += 1

            preds_batch = _decode_batch(doc_ids, line_preds, doc_probs, mask)
            all_preds.extend(preds_batch)

    avg_loss = total_loss / max(1, n_batches)
    return all_preds, avg_loss


def _decode_batch(doc_ids, line_preds, doc_probs, mask) -> list[Prediction]:
    """Convert model outputs to Prediction objects."""
    results: list[Prediction] = []
    for i, doc_id in enumerate(doc_ids):
        n_valid = int(mask[i].sum())
        pred_labels = line_preds[i, :n_valid].cpu().numpy().astype(np.int64)
        doc_p = float(doc_probs[i])

        spans = _labels_to_spans(pred_labels)

        if not spans:
            results.append(Prediction(
                doc_id=str(doc_id), has_anomaly=0,
                primary_start_idx=-1, primary_end_idx=-1,
                primary_anomaly_type="none", all_spans="",
            ))
        else:
            primary = spans[0]
            all_spans = ";".join(f"{s}|{e}|{t}" for s, e, t in spans)
            results.append(Prediction(
                doc_id=str(doc_id), has_anomaly=1,
                primary_start_idx=primary[0], primary_end_idx=primary[1],
                primary_anomaly_type=primary[2], all_spans=all_spans,
            ))
    return results


def _labels_to_spans(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Convert per-line label array to list of (start, end, type) spans."""
    n = len(labels)
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        if labels[i] == O_LABEL:
            i += 1
            continue
        label_id = labels[i]
        start = i
        while i < n and labels[i] == label_id:
            i += 1
        end = i - 1
        if label_id in LABEL_TO_TYPE:
            spans.append((int(start), int(end), LABEL_TO_TYPE[label_id]))
    return spans


def train_epoch(
    model: LogBiLSTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    class_weights: torch.Tensor,
    doc_pos_weight: float,
    device: torch.device,
    grad_clip: float,
) -> float:
    """Train one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)
        has_anom = batch["has_anomaly"].to(device)

        line_logits, doc_logits = model(features, mask)

        line_loss = F.cross_entropy(
            line_logits.permute(0, 2, 1), labels, weight=class_weights.to(device),
            reduction="sum",
        ) / mask.sum().float()

        doc_loss = F.binary_cross_entropy_with_logits(
            doc_logits.squeeze(-1), has_anom,
            pos_weight=torch.tensor(doc_pos_weight, device=device),
            reduction="mean",
        )

        loss = line_loss + 0.5 * doc_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


def train_fold(
    model: LogBiLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    fold_idx: int,
    device: torch.device,
    class_weights: torch.Tensor,
    doc_pos_weight: float,
) -> tuple[LogBiLSTM, list[Prediction], int, float]:
    """Train for one fold. Returns (best_model, val_predictions, best_epoch, best_score)."""
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        # Compute class weights fresh (in case of class imbalance changes)
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            class_weights, doc_pos_weight, device, args.grad_clip,
        )

        val_preds, val_loss = _model_predictions(model, val_loader, device)

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    # Load best state for final validation
    model.load_state_dict(best_state)
    val_preds, _ = _model_predictions(model, val_loader, device)
    return model, val_preds, best_epoch, best_val_loss


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BiLSTM v2 for log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--action-log", type=Path, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if args.model_path is None:
        args.model_path = data_dir / "模型" / "model_bundle_nn_v2.joblib"
    if args.submission_path is None:
        args.submission_path = data_dir / "提交结果" / "submission_nn_v2.csv"
    if args.report_path is None:
        args.report_path = data_dir / "模型" / "validation_report_nn_v2.json"
    if args.action_log is None:
        args.action_log = data_dir / "ACTION_LOG.md"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dense features
    cache_dir = data_dir / "缓存"
    dense_train_path = cache_dir / "dense_train.joblib"
    dense_test_path = cache_dir / "dense_test.joblib"

    if not dense_train_path.exists():
        print(f"Dense features not found at {dense_train_path}")
        print("Run 'python 源码/build_dense_features.py' first.")
        return 1

    print("Loading dense features...")
    dense_train = joblib.load(dense_train_path)
    dense_test = joblib.load(dense_test_path)
    print(f"  Train: {len(dense_train)} docs, Test: {len(dense_test)} docs")

    # Load original docs for validation
    train_docs = read_documents(data_dir / args.train_file, expect_labels=True)
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)

    # Verify alignment
    for i, (feat, doc) in enumerate(zip(dense_train, train_docs)):
        if str(feat["doc_id"]) != str(doc.doc_id):
            raise ValueError(f"Train order mismatch at {i}: {feat['doc_id']} vs {doc.doc_id}")
    for i, (feat, doc) in enumerate(zip(dense_test, test_docs)):
        if str(feat["doc_id"]) != str(doc.doc_id):
            raise ValueError(f"Test order mismatch at {i}: {feat['doc_id']} vs {doc.doc_id}")
    print("  Document order verified")

    input_dim = dense_train[0]["features"].shape[1]
    print(f"  Input dim: {input_dim}")

    append_action_log(args.action_log, "Training started (BiLSTM v2).")

    # Compute global class weights
    class_weights, doc_pos_weight = _compute_class_weights(dense_train)
    print(f"  Class weights: {class_weights.tolist()}")
    print(f"  Doc pos weight: {doc_pos_weight:.3f}")

    # 5-fold CV
    y = np.array([item["has_anomaly"] for item in dense_train], dtype=np.int32)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_predictions: list[Prediction | None] = [None] * len(train_docs)
    fold_reports: list[dict] = []
    best_epochs: list[int] = []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        tqdm(cv.split(np.zeros(len(dense_train)), y), total=args.folds, desc="CV folds", unit="fold", dynamic_ncols=True),
        start=1,
    ):
        train_data = [dense_train[int(i)] for i in tr_idx]
        val_data = [dense_train[int(i)] for i in va_idx]

        train_ds = DocDataset(train_data)
        val_ds = DocDataset(val_data)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=collate_batch, num_workers=0, drop_last=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_batch, num_workers=0,
        )

        model = LogBiLSTM(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_labels=NUM_LABELS,
            dropout=args.dropout,
            num_lstm_layers=args.lstm_layers,
        ).to(device)

        torch.manual_seed(args.seed + fold_idx * 1000)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed + fold_idx * 1000)

        model, val_preds, best_epoch, best_loss = train_fold(
            model, train_loader, val_loader, args, fold_idx, device,
            class_weights, doc_pos_weight,
        )

        best_epochs.append(best_epoch)
        for local_pos, global_idx in enumerate(va_idx):
            oof_predictions[int(global_idx)] = val_preds[local_pos]

        # Score on validation fold
        val_docs = [train_docs[int(i)] for i in va_idx]
        val_metrics = evaluate_predictions(val_docs, val_preds)
        fold_reports.append({
            "fold": fold_idx,
            "train_rows": len(train_data),
            "valid_rows": len(val_data),
            "best_epoch": best_epoch,
            "val_loss": round(best_loss, 6),
            "val_score": round(val_metrics["score"], 6),
        })
        append_action_log(
            args.action_log,
            f"NN Fold {fold_idx}: best_epoch={best_epoch}, val_score={val_metrics['score']:.6f}.",
        )

    # OOF scoring
    if any(p is None for p in oof_predictions):
        raise RuntimeError("OOF prediction collection failed.")
    oof_preds = [p for p in oof_predictions if p is not None]
    oof_metrics = evaluate_predictions(train_docs, oof_preds)
    final_epochs = max(1, min(args.epochs, int(round(float(np.median(best_epochs))))))

    print(f"\nOOF score: {oof_metrics['score']:.6f}")
    print(f"  F1_detect: {oof_metrics['f1_detect']:.6f}")
    print(f"  IoU_loc:   {oof_metrics['iou_loc']:.6f}")
    print(f"  F1_type:   {oof_metrics['f1_type']:.6f}")
    print(f"  Final epochs: {final_epochs}")

    append_action_log(
        args.action_log,
        f"NN OOF score={oof_metrics['score']:.6f}, final_epochs={final_epochs}.",
    )

    # Full retrain
    print("\n=== Full retrain ===")
    full_ds = DocDataset(dense_train)
    full_loader = DataLoader(
        full_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_batch, num_workers=0,
    )

    full_model = LogBiLSTM(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_labels=NUM_LABELS, dropout=args.dropout,
        num_lstm_layers=args.lstm_layers,
    ).to(device)

    torch.manual_seed(args.seed + 90000)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + 90000)

    optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=final_epochs)

    for epoch in range(1, final_epochs + 1):
        train_loss = train_epoch(
            full_model, full_loader, optimizer, scheduler,
            class_weights, doc_pos_weight, device, args.grad_clip,
        )
        print(f"  Full epoch {epoch}/{final_epochs}: loss={train_loss:.6f}")

    # Predict test
    print("\n=== Predict test ===")
    test_ds = DocDataset(dense_test)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_batch, num_workers=0,
    )
    test_preds, _ = _model_predictions(full_model, test_loader, device)

    # Save submission
    write_submission(args.submission_path, test_preds)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    validation_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    # Save model bundle
    bundle = {
        "version": 4,
        "model": "BiLSTM_v2",
        "seed": args.seed,
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "num_labels": NUM_LABELS,
        "dropout": args.dropout,
        "lstm_layers": args.lstm_layers,
        "state_dict": {k: v.cpu().clone() for k, v in full_model.state_dict().items()},
        "final_epochs": final_epochs,
        "oof_metrics": oof_metrics,
        "fold_reports": fold_reports,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)

    # Save report
    report = {
        "version": 4,
        "model": "BiLSTM_v2",
        "train_rows": len(dense_train),
        "test_rows": len(dense_test),
        "folds": args.folds,
        "epochs": args.epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "lstm_layers": args.lstm_layers,
        "input_dim": input_dim,
        "final_epochs": final_epochs,
        "best_epochs": best_epochs,
        "oof_metrics": oof_metrics,
        "fold_reports": fold_reports,
        "submission_validation": validation_info,
    }
    write_json(args.report_path, report)

    append_action_log(args.action_log, f"NN Training completed: model={args.model_path}, submission={args.submission_path}.")
    print(f"\nDone! Model: {args.model_path}, Submission: {args.submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
