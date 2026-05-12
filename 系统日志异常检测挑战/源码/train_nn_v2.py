"""Train BiLSTM v2 for log anomaly detection — with AMP, checkpointing, and speed optimizations.

5-fold CV with per-fold checkpoints → OOF scoring → full retrain → submission.
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
    ANOMALY_TYPES, DEFAULT_SEED, Prediction,
    append_action_log, evaluate_predictions, read_documents,
    read_sample_columns, validate_submission_file, write_json, write_submission,
)
from model_nn_v2 import LABEL_TO_TYPE, NUM_LABELS, O_LABEL, LogBiLSTM, collate_batch


class DocDataset(Dataset):
    def __init__(self, data: list[dict]): self.data = data
    def __len__(self) -> int: return len(self.data)
    def __getitem__(self, idx: int) -> dict: return self.data[idx]


def _compute_class_weights(dense_train: list[dict]) -> tuple[torch.Tensor, float]:
    all_labels = np.concatenate([item["labels"] for item in dense_train])
    counts = np.bincount(all_labels, minlength=NUM_LABELS).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = torch.from_numpy((1.0 / counts / (1.0 / counts).mean()).astype(np.float32))
    has_anom = np.array([item["has_anomaly"] for item in dense_train], dtype=np.float32)
    pos, neg = float(has_anom.sum()), float(len(has_anom) - has_anom.sum())
    return weights, neg / max(1.0, pos)


def _make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int, persistent: bool = False) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_batch,
                      num_workers=num_workers, pin_memory=True, drop_last=False,
                      persistent_workers=persistent and num_workers > 0)


def _model_predictions(model: LogBiLSTM, loader: DataLoader, device: torch.device) -> tuple[list[dict], float]:
    model.eval()
    all_preds: list[dict] = []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            doc_ids = batch["doc_ids"]
            line_logits, doc_logits = model(features, mask)
            line_preds = torch.argmax(line_logits, dim=-1) * mask.long()
            doc_probs = torch.sigmoid(doc_logits).squeeze(-1)
            if "labels" in batch:
                labels = batch["labels"].to(device, non_blocking=True)
                has_anom = batch["has_anomaly"].to(device, non_blocking=True)
                line_loss = F.cross_entropy(line_logits.permute(0, 2, 1), labels, reduction="sum") / mask.sum().float()
                doc_loss = F.binary_cross_entropy_with_logits(doc_logits.squeeze(-1), has_anom, reduction="mean")
                total_loss += (line_loss + 0.5 * doc_loss).item(); n += 1
            all_preds.extend(_decode_batch(doc_ids, line_preds, doc_probs, mask))
    return all_preds, total_loss / max(1, n)


def _decode_batch(doc_ids, line_preds, doc_probs, mask) -> list[Prediction]:
    results: list[Prediction] = []
    for i, doc_id in enumerate(doc_ids):
        nv = int(mask[i].sum())
        pred_labels = line_preds[i, :nv].cpu().numpy().astype(np.int64)
        spans = _labels_to_spans(pred_labels)
        if not spans:
            results.append(Prediction(str(doc_id), 0, -1, -1, "none", ""))
        else:
            p = spans[0]
            results.append(Prediction(str(doc_id), 1, p[0], p[1], p[2],
                          ";".join(f"{s}|{e}|{t}" for s, e, t in spans)))
    return results


def _labels_to_spans(labels: np.ndarray) -> list[tuple[int, int, str]]:
    spans, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] == O_LABEL: i += 1; continue
        lid, s = labels[i], i
        while i < n and labels[i] == lid: i += 1
        if lid in LABEL_TO_TYPE: spans.append((int(s), int(i - 1), LABEL_TO_TYPE[lid]))
    return spans


# ---- Core training with AMP ----

def train_epoch(model, loader, optimizer, scheduler, class_weights, doc_pos_weight,
                device, grad_clip, scaler=None, use_amp=True):
    model.train()
    total_loss, n = 0.0, 0
    cw = class_weights.to(device)
    dpw = torch.tensor(doc_pos_weight, device=device)

    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        has_anom = batch["has_anomaly"].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            line_logits, doc_logits = model(features, mask)

            # Length-weighted line loss: long docs get higher weight
            doc_lens = mask.sum(dim=1).float().clamp(min=1)  # (B,)
            len_w = torch.ones_like(doc_lens)
            len_w[doc_lens > 100] = 3.0
            len_w[(doc_lens > 50) & (doc_lens <= 100)] = 1.5
            len_w = len_w / len_w.mean()  # normalize so avg weight = 1

            line_ce = F.cross_entropy(line_logits.permute(0, 2, 1), labels, weight=cw, reduction="none")  # (B, T)
            line_loss_per_doc = (line_ce * mask.float()).sum(dim=1) / doc_lens  # (B,)
            line_loss = (line_loss_per_doc * len_w).mean()

            doc_loss = F.binary_cross_entropy_with_logits(doc_logits.squeeze(-1), has_anom, pos_weight=dpw, reduction="mean")
            loss = line_loss + 0.5 * doc_loss

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(1, n)


def train_fold(model, train_loader, val_loader, args, fold_idx, device,
               class_weights, doc_pos_weight):
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    best_val_loss, best_epoch = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        train_epoch(model, train_loader, optimizer, scheduler,
                    class_weights, doc_pos_weight, device, args.grad_clip,
                    scaler=scaler, use_amp=args.use_amp)
        _, val_loss = _model_predictions(model, val_loader, device)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict()); patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0: break

    model.load_state_dict(best_state)
    val_preds, _ = _model_predictions(model, val_loader, device)
    return model, val_preds, best_epoch, best_val_loss


# ---- Main ----

def main() -> int:
    parser = argparse.ArgumentParser(description="Train BiLSTM v2 (AMP + checkpoints).")
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    args.use_amp = not args.no_amp and torch.cuda.is_available()

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
    print(f"Device: {device}, AMP: {args.use_amp}, workers: {args.num_workers}, batch: {args.batch_size}")

    cache_dir = data_dir / "缓存"
    dense_train_path = cache_dir / "dense_train.joblib"; dense_test_path = cache_dir / "dense_test.joblib"
    if not dense_train_path.exists():
        print(f"Dense features not found. Run build_dense_features.py first."); return 1

    print("Loading data...")
    dense_train = joblib.load(dense_train_path); dense_test = joblib.load(dense_test_path)
    train_docs = read_documents(data_dir / args.train_file, expect_labels=True)
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)

    for i, (f, d) in enumerate(zip(dense_train, train_docs)):
        if str(f["doc_id"]) != str(d.doc_id): raise ValueError(f"Train order mismatch at {i}")
    for i, (f, d) in enumerate(zip(dense_test, test_docs)):
        if str(f["doc_id"]) != str(d.doc_id): raise ValueError(f"Test order mismatch at {i}")
    print(f"  Train: {len(dense_train)}, Test: {len(dense_test)}")

    input_dim = dense_train[0]["features"].shape[1]
    class_weights, doc_pos_weight = _compute_class_weights(dense_train)
    print(f"  dim={input_dim}, doc_pw={doc_pos_weight:.2f}")

    append_action_log(args.action_log, f"Training started (BiLSTM v2, seed={args.seed}, amp={args.use_amp}).")

    # 5-fold CV with checkpoint resume
    prefix = Path(args.train_file).stem
    ckpt_path = cache_dir / f"train_ckpt_{prefix}_seed{args.seed}.joblib"
    y = np.array([item["has_anomaly"] for item in dense_train], dtype=np.int32)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    # Resume or start fresh
    if ckpt_path.exists():
        ckpt = joblib.load(ckpt_path)
        oof_predictions = ckpt["oof_predictions"]
        fold_reports = ckpt["fold_reports"]
        best_epochs = ckpt["best_epochs"]
        start_fold = ckpt["completed_folds"] + 1
        print(f"Resuming from checkpoint: {ckpt['completed_folds']}/{args.folds} folds done")
    else:
        oof_predictions = [None] * len(train_docs)
        fold_reports, best_epochs = [], []
        start_fold = 1

    fold_iter = list(cv.split(np.zeros(len(dense_train)), y))
    for fold_idx, (tr_idx, va_idx) in enumerate(fold_iter, start=1):
        if fold_idx < start_fold: continue

        print(f"  Fold {fold_idx}/{args.folds}: preparing data & model...", flush=True)
        train_data = [dense_train[int(i)] for i in tr_idx]
        val_data = [dense_train[int(i)] for i in va_idx]

        train_loader = _make_loader(DocDataset(train_data), args.batch_size, shuffle=True, num_workers=args.num_workers, persistent=True)
        val_loader = _make_loader(DocDataset(val_data), args.batch_size, shuffle=False, num_workers=args.num_workers)

        model = LogBiLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim,
                          num_labels=NUM_LABELS, dropout=args.dropout,
                          num_lstm_layers=args.lstm_layers).to(device)
        torch.manual_seed(args.seed + fold_idx * 1000)
        if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + fold_idx * 1000)

        model, val_preds, best_epoch, best_loss = train_fold(
            model, train_loader, val_loader, args, fold_idx, device, class_weights, doc_pos_weight)

        best_epochs.append(best_epoch)
        for lp, gi in enumerate(va_idx): oof_predictions[int(gi)] = val_preds[lp]

        val_docs = [train_docs[int(i)] for i in va_idx]
        val_metrics = evaluate_predictions(val_docs, val_preds)
        fold_reports.append({"fold": fold_idx, "train_rows": len(train_data), "valid_rows": len(val_data),
                            "best_epoch": best_epoch, "val_loss": round(best_loss, 6),
                            "val_score": round(val_metrics["score"], 6)})

        # Save checkpoint after each fold
        joblib.dump({"completed_folds": fold_idx, "oof_predictions": oof_predictions,
                      "fold_reports": fold_reports, "best_epochs": best_epochs},
                     ckpt_path, compress=3)
        append_action_log(args.action_log, f"Fold {fold_idx}: epoch={best_epoch}, score={val_metrics['score']:.6f}. ckpt saved.")

        # Clean up DataLoader workers
        train_loader._iterator = None; val_loader._iterator = None

    # OOF
    if any(p is None for p in oof_predictions): raise RuntimeError("OOF collection failed")
    oof_preds = [p for p in oof_predictions if p is not None]
    oof_metrics = evaluate_predictions(train_docs, oof_preds)
    final_epochs = max(1, min(args.epochs, int(round(float(np.median(best_epochs))))))
    print(f"\nOOF: score={oof_metrics['score']:.6f}, F1_d={oof_metrics['f1_detect']:.4f}, IoU={oof_metrics['iou_loc']:.4f}, F1_t={oof_metrics['f1_type']:.4f}")
    print(f"  final_epochs={final_epochs}, best_epochs={best_epochs}")

    append_action_log(args.action_log, f"OOF={oof_metrics['score']:.6f}, final_epochs={final_epochs}.")

    # Full retrain
    print("\n=== Full retrain ===")
    full_loader = _make_loader(DocDataset(dense_train), args.batch_size, shuffle=True, num_workers=args.num_workers, persistent=True)
    full_model = LogBiLSTM(input_dim=input_dim, hidden_dim=args.hidden_dim,
                           num_labels=NUM_LABELS, dropout=args.dropout,
                           num_lstm_layers=args.lstm_layers).to(device)
    torch.manual_seed(args.seed + 90000)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + 90000)

    optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=final_epochs)
    scaler = torch.amp.GradScaler("cuda") if args.use_amp else None

    for epoch in range(1, final_epochs + 1):
        loss = train_epoch(full_model, full_loader, optimizer, scheduler,
                          class_weights, doc_pos_weight, device, args.grad_clip,
                          scaler=scaler, use_amp=args.use_amp)
        print(f"  Full epoch {epoch}/{final_epochs}: loss={loss:.6f}")

    # Predict test
    print("\n=== Predict test ===")
    test_loader = _make_loader(DocDataset(dense_test), args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_preds, _ = _model_predictions(full_model, test_loader, device)

    # Save
    write_submission(args.submission_path, test_preds)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    val_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    bundle = {"version": 4, "model": "BiLSTM_v2", "seed": args.seed,
              "input_dim": input_dim, "hidden_dim": args.hidden_dim,
              "num_labels": NUM_LABELS, "dropout": args.dropout, "lstm_layers": args.lstm_layers,
              "state_dict": {k: v.cpu().clone() for k, v in full_model.state_dict().items()},
              "final_epochs": final_epochs, "oof_metrics": oof_metrics, "fold_reports": fold_reports}
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)

    report = {"version": 4, "model": "BiLSTM_v2", "train_rows": len(dense_train),
              "test_rows": len(dense_test), "folds": args.folds, "epochs": args.epochs,
              "input_dim": input_dim, "hidden_dim": args.hidden_dim,
              "final_epochs": final_epochs, "best_epochs": best_epochs,
              "oof_metrics": oof_metrics, "fold_reports": fold_reports,
              "submission_validation": val_info}
    write_json(args.report_path, report)

    # Clean up checkpoint on success
    if ckpt_path.exists(): ckpt_path.unlink()

    append_action_log(args.action_log, f"Training completed: {args.model_path}")
    print(f"\nDone! Model: {args.model_path}\nSubmission: {args.submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
