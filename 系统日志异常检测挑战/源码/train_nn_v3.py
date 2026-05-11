"""Train BiLSTM v3 with boundary-focused (start/end/type) heads.

5-fold CV with OOF threshold tuning on raw probabilities → full retrain → submission.
"""

from __future__ import annotations

import argparse
import copy
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
    TYPE_SPAN_LENGTHS,
    Prediction,
    append_action_log,
    evaluate_predictions,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_json,
    write_submission,
)
from model_nn_v3 import (
    LABEL_TO_TYPE,
    NUM_LABELS,
    O_LABEL,
    LogBiLSTMv3,
    _labels_to_targets,
    collate_batch,
    decode_boundary_spans,
    predictions_from_boundary,
)


class DocDataset(Dataset):
    def __init__(self, data): self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, i): return self.data[i]


def _compute_weights(dense_train: list[dict]):
    all_labels = np.concatenate([item["labels"] for item in dense_train])
    counts = np.bincount(all_labels, minlength=NUM_LABELS).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    type_weights = 1.0 / counts
    type_weights = type_weights / type_weights.mean()
    type_weights = torch.from_numpy(type_weights.astype(np.float32))

    n_start, n_end, n_total = 0, 0, 0
    for item in dense_train:
        labels = item["labels"]; n = len(labels); n_total += n
        i = 0
        while i < n:
            if labels[i] == O_LABEL: i += 1; continue
            n_start += 1; lid = labels[i]
            while i < n and labels[i] == lid: i += 1
            n_end += 1
    start_pw = float((n_total - n_start) / max(1, n_start))
    end_pw = float((n_total - n_end) / max(1, n_end))

    has_anom = np.array([item["has_anomaly"] for item in dense_train], dtype=np.float32)
    pos = has_anom.sum(); neg = len(has_anom) - pos
    doc_pw = float(neg / max(1, pos))
    return type_weights, start_pw, end_pw, doc_pw


def train_epoch(model, loader, optimizer, scheduler, type_weights, start_pw, end_pw, doc_pw, device, grad_clip):
    model.train()
    total_loss, n = 0.0, 0
    for batch in loader:
        features = batch["features"].to(device); mask = batch["mask"].to(device)
        labels = batch["labels"].to(device); has_anom = batch["has_anomaly"].to(device)
        boundary_logits, type_logits, doc_logits = model(features, mask)
        st, et, tt = _labels_to_targets(labels, mask)

        sl = F.binary_cross_entropy_with_logits(boundary_logits[:,:,0], st, pos_weight=torch.tensor(start_pw, device=device))
        el = F.binary_cross_entropy_with_logits(boundary_logits[:,:,1], et, pos_weight=torch.tensor(end_pw, device=device))
        tl = F.cross_entropy(type_logits.permute(0,2,1), tt, weight=type_weights.to(device), ignore_index=-100)
        dl = F.binary_cross_entropy_with_logits(doc_logits.squeeze(-1), has_anom, pos_weight=torch.tensor(doc_pw, device=device))
        loss = sl + el + tl + 0.5 * dl

        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item(); n += 1
    return total_loss / max(1, n)


@torch.no_grad()
def predict_with_raw_probs(model, loader, device):
    """Returns (decoded_predictions, raw_probs_list, val_loss).

    raw_probs_list: list of dicts with doc_id, start_probs, end_probs, type_probs, doc_prob, n_lines
    """
    model.eval()
    all_preds, all_raw = [], []
    total_loss, n = 0.0, 0

    for batch in loader:
        features = batch["features"].to(device); mask = batch["mask"].to(device)
        doc_ids = batch["doc_ids"]
        boundary_logits, type_logits, doc_logits = model(features, mask)

        start_probs = torch.sigmoid(boundary_logits[:,:,0]).cpu().numpy()
        end_probs = torch.sigmoid(boundary_logits[:,:,1]).cpu().numpy()
        type_probs = F.softmax(type_logits, dim=-1).cpu().numpy()
        doc_probs = torch.sigmoid(doc_logits).squeeze(-1).cpu().numpy()

        if "labels" in batch:
            labels = batch["labels"].to(device); has_anom = batch["has_anomaly"].to(device)
            st, et, tt = _labels_to_targets(labels, mask)
            sl = F.binary_cross_entropy_with_logits(boundary_logits[:,:,0], st, reduction="sum") / mask.sum().float()
            el = F.binary_cross_entropy_with_logits(boundary_logits[:,:,1], et, reduction="sum") / mask.sum().float()
            tl = F.cross_entropy(type_logits.permute(0,2,1), tt, ignore_index=-100)
            dl = F.binary_cross_entropy_with_logits(doc_logits.squeeze(-1), has_anom, reduction="mean")
            total_loss += (sl + el + tl + 0.5 * dl).item(); n += 1

        # Decode with default thresholds
        preds = predictions_from_boundary(doc_ids, start_probs, end_probs, type_probs, doc_probs, mask)
        all_preds.extend(preds)

        # Save raw probs for tuning
        for i, did in enumerate(doc_ids):
            nv = int(mask[i].sum())
            all_raw.append({
                "doc_id": str(did),
                "start_probs": start_probs[i, :nv].astype(np.float32),
                "end_probs": end_probs[i, :nv].astype(np.float32),
                "type_probs": type_probs[i, :nv].astype(np.float32),
                "doc_prob": float(doc_probs[i]),
                "n_lines": nv,
            })

    avg_loss = total_loss / max(1, n)
    return all_preds, all_raw, avg_loss


def re_decode_from_raw(raw_probs_list, thresholds):
    """Re-decode raw probabilities with given thresholds."""
    st = thresholds.get("start_threshold", 0.5)
    et = thresholds.get("end_threshold", 0.5)
    min_len = thresholds.get("min_span_len", 2)
    max_len = thresholds.get("max_span_len", 15)
    max_s = thresholds.get("max_spans", 3)
    tsl = thresholds.get("type_span_lengths", None)
    doc_thr = thresholds.get("doc_threshold", 0.5)

    results = []
    for rp in all_raw_probs_list:
        nv = rp["n_lines"]
        spans = decode_boundary_spans(
            rp["start_probs"], rp["end_probs"], rp["type_probs"],
            st, et, min_len, max_len, max_s, tsl,
        )
        doc_p = rp["doc_prob"]
        if not spans and doc_p < doc_thr:
            results.append(Prediction(str(rp["doc_id"]), 0, -1, -1, "none", ""))
        elif not spans:
            # Fallback: argmax on type probs
            type_preds = np.argmax(rp["type_probs"], axis=-1).astype(np.int64)
            spans = []
            j = 0
            while j < len(type_preds):
                if type_preds[j] == O_LABEL: j += 1; continue
                lid = type_preds[j]; s = j
                while j < len(type_preds) and type_preds[j] == lid: j += 1
                if lid in LABEL_TO_TYPE:
                    spans.append((int(s), int(j-1), LABEL_TO_TYPE[lid], 0.5))
            if not spans:
                results.append(Prediction(str(rp["doc_id"]), 0, -1, -1, "none", ""))
                continue
            p = spans[0]
            results.append(Prediction(str(rp["doc_id"]), 1, p[0], p[1], p[2],
                          ";".join(f"{s}|{e}|{t}" for s,e,t,_ in spans)))
        else:
            p = spans[0]
            results.append(Prediction(str(rp["doc_id"]), 1, p[0], p[1], p[2],
                          ";".join(f"{s}|{e}|{t}" for s,e,t,_ in spans)))
    return results


def tune_boundary_thresholds(train_docs, oof_raw, n_trials=200):
    """Search boundary thresholds by re-decoding OOF raw probabilities."""
    rng = np.random.default_rng(DEFAULT_SEED + 777)

    y_doc = np.array([d.has_anomaly for d in train_docs])
    pos_idx = np.flatnonzero(y_doc == 1); neg_idx = np.flatnonzero(y_doc == 0)
    if len(train_docs) > 5000:
        sampled = np.concatenate([
            rng.choice(pos_idx, size=min(2500, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(2500, len(neg_idx)), replace=False),
        ])
        search_idx = np.sort(sampled.astype(np.int64))
    else:
        search_idx = np.arange(len(train_docs))

    search_docs = [train_docs[int(i)] for i in search_idx]
    search_raw = [oof_raw[int(i)] for i in search_idx]

    best_score = -1.0
    best_thr = {"start_threshold": 0.5, "end_threshold": 0.5, "min_span_len": 2,
                "max_span_len": 15, "max_spans": 3, "doc_threshold": 0.5,
                "type_span_lengths": TYPE_SPAN_LENGTHS}

    for _ in tqdm(range(n_trials), desc="tune thresholds", unit="trial", dynamic_ncols=True):
        thr = {
            "start_threshold": float(rng.uniform(0.2, 0.8)),
            "end_threshold": float(rng.uniform(0.2, 0.8)),
            "min_span_len": int(rng.integers(1, 5)),
            "max_span_len": int(rng.integers(8, 20)),
            "max_spans": int(rng.integers(1, 4)),
            "doc_threshold": float(rng.uniform(0.3, 0.7)),
            "type_span_lengths": TYPE_SPAN_LENGTHS,
        }
        preds = re_decode_from_raw(search_raw, thr)
        metrics = evaluate_predictions(search_docs, preds)
        if metrics["score"] > best_score + 1e-12:
            best_score = metrics["score"]
            best_thr = thr

    return best_thr, best_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Train BiLSTM v3 boundary model.")
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
        args.model_path = data_dir / "模型" / "model_bundle_nn_v3.joblib"
    if args.submission_path is None:
        args.submission_path = data_dir / "提交结果" / "submission_nn_v3.csv"
    if args.report_path is None:
        args.report_path = data_dir / "模型" / "validation_report_nn_v3.json"
    if args.action_log is None:
        args.action_log = data_dir / "ACTION_LOG.md"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cache_dir = data_dir / "缓存"
    dense_train_path = cache_dir / "dense_train.joblib"
    dense_test_path = cache_dir / "dense_test.joblib"
    if not dense_train_path.exists():
        print("Dense features not found. Run 'python 源码/build_dense_features.py' first.")
        return 1

    print("Loading data...")
    dense_train = joblib.load(dense_train_path)
    dense_test = joblib.load(dense_test_path)
    train_docs = read_documents(data_dir / args.train_file, expect_labels=True)
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    for i, (f, d) in enumerate(zip(dense_train, train_docs)):
        if str(f["doc_id"]) != str(d.doc_id): raise ValueError(f"Order mismatch at {i}")
    print(f"  Train: {len(dense_train)} docs, Test: {len(dense_test)} docs")

    input_dim = dense_train[0]["features"].shape[1]
    type_weights, start_pw, end_pw, doc_pw = _compute_weights(dense_train)
    print(f"  Input dim: {input_dim}, start_pw: {start_pw:.1f}, end_pw: {end_pw:.1f}, doc_pw: {doc_pw:.2f}")

    append_action_log(args.action_log, "Training started (BiLSTM v3 boundary).")

    y = np.array([item["has_anomaly"] for item in dense_train], dtype=np.int32)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    oof_raw: list[dict | None] = [None] * len(train_docs)
    fold_reports, best_epochs = [], []

    for fold_idx, (tr_idx, va_idx) in enumerate(
        tqdm(cv.split(np.zeros(len(dense_train)), y), total=args.folds, desc="CV folds", unit="fold", dynamic_ncols=True),
        start=1,
    ):
        train_data = [dense_train[int(i)] for i in tr_idx]
        val_data = [dense_train[int(i)] for i in va_idx]
        train_loader = DataLoader(DocDataset(train_data), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
        val_loader = DataLoader(DocDataset(val_data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        model = LogBiLSTMv3(input_dim=input_dim, hidden_dim=args.hidden_dim,
                            num_labels=NUM_LABELS, dropout=args.dropout,
                            num_lstm_layers=args.lstm_layers).to(device)
        torch.manual_seed(args.seed + fold_idx * 1000)
        if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + fold_idx * 1000)

        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
        best_val_loss, best_epoch = float("inf"), 0
        best_state = copy.deepcopy(model.state_dict())
        patience_left = args.patience

        for epoch in range(1, args.epochs + 1):
            train_epoch(model, train_loader, optimizer, scheduler,
                       type_weights, start_pw, end_pw, doc_pw, device, args.grad_clip)
            _, _, val_loss = predict_with_raw_probs(model, val_loader, device)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss; best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict()); patience_left = args.patience
            else:
                patience_left -= 1
                if patience_left <= 0: break

        model.load_state_dict(best_state)
        val_preds, val_raw, _ = predict_with_raw_probs(model, val_loader, device)
        for local_pos, global_idx in enumerate(va_idx):
            oof_raw[int(global_idx)] = val_raw[local_pos]

        val_docs = [train_docs[int(i)] for i in va_idx]
        val_pred_objs = [Prediction(**d) for d in val_preds]
        val_metrics = evaluate_predictions(val_docs, val_pred_objs)
        best_epochs.append(best_epoch)
        fold_reports.append({"fold": fold_idx, "train_rows": len(train_data), "valid_rows": len(val_data),
                            "best_epoch": best_epoch, "val_score": round(val_metrics["score"], 6)})
        append_action_log(args.action_log, f"NNv3 Fold {fold_idx}: best_epoch={best_epoch}, val_score={val_metrics['score']:.6f}.")

    if any(p is None for p in oof_raw): raise RuntimeError("OOF collection failed")
    oof_raw_final = [p for p in oof_raw if p is not None]
    final_epochs = max(1, min(args.epochs, int(round(float(np.median(best_epochs))))))

    # Default thresholds OOF
    oof_preds_default = re_decode_from_raw(oof_raw_final, {"start_threshold": 0.5, "end_threshold": 0.5, "min_span_len": 2, "max_span_len": 15, "max_spans": 3, "doc_threshold": 0.5, "type_span_lengths": TYPE_SPAN_LENGTHS})
    oof_metrics_default = evaluate_predictions(train_docs, oof_preds_default)
    print(f"\nOOF (default thresholds): score={oof_metrics_default['score']:.6f}, IoU={oof_metrics_default['iou_loc']:.4f}")

    # Tune thresholds on raw probs
    best_thr, best_thr_score = tune_boundary_thresholds(train_docs, oof_raw_final)
    oof_preds_tuned = re_decode_from_raw(oof_raw_final, best_thr)
    oof_metrics_tuned = evaluate_predictions(train_docs, oof_preds_tuned)
    print(f"Tuned: start={best_thr['start_threshold']:.3f}, end={best_thr['end_threshold']:.3f}, min_len={best_thr['min_span_len']}, max_len={best_thr['max_span_len']}, max_spans={best_thr['max_spans']}")
    print(f"OOF tuned: score={oof_metrics_tuned['score']:.6f}, IoU={oof_metrics_tuned['iou_loc']:.4f}")

    append_action_log(args.action_log, f"NNv3 OOF default={oof_metrics_default['score']:.6f}, tuned={best_thr_score:.6f}, final_epochs={final_epochs}.")

    # Full retrain
    print("\n=== Full retrain ===")
    full_loader = DataLoader(DocDataset(dense_train), batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    full_model = LogBiLSTMv3(input_dim=input_dim, hidden_dim=args.hidden_dim,
                             num_labels=NUM_LABELS, dropout=args.dropout,
                             num_lstm_layers=args.lstm_layers).to(device)
    torch.manual_seed(args.seed + 90000)
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed + 90000)
    optimizer = AdamW(full_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=final_epochs)
    for epoch in range(1, final_epochs + 1):
        loss = train_epoch(full_model, full_loader, optimizer, scheduler,
                          type_weights, start_pw, end_pw, doc_pw, device, args.grad_clip)
        print(f"  Full epoch {epoch}/{final_epochs}: loss={loss:.6f}")

    # Predict test
    print("\n=== Predict test ===")
    test_loader = DataLoader(DocDataset(dense_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_preds_dict, _, _ = predict_with_raw_probs(full_model, test_loader, device)
    test_preds = [Prediction(**d) for d in test_preds_dict]
    write_submission(args.submission_path, test_preds)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    val_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    bundle = {
        "version": 5, "model": "BiLSTM_v3_boundary", "seed": args.seed,
        "input_dim": input_dim, "hidden_dim": args.hidden_dim,
        "num_labels": NUM_LABELS, "dropout": args.dropout, "lstm_layers": args.lstm_layers,
        "state_dict": {k: v.cpu().clone() for k, v in full_model.state_dict().items()},
        "thresholds": best_thr, "final_epochs": final_epochs,
        "oof_metrics": oof_metrics_tuned, "fold_reports": fold_reports,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_path, compress=3)

    report = {
        "version": 5, "model": "BiLSTM_v3_boundary",
        "train_rows": len(dense_train), "test_rows": len(dense_test),
        "folds": args.folds, "epochs": args.epochs,
        "input_dim": input_dim, "hidden_dim": args.hidden_dim,
        "final_epochs": final_epochs, "best_epochs": best_epochs,
        "oof_metrics_default": oof_metrics_default,
        "oof_metrics_tuned": oof_metrics_tuned,
        "thresholds": best_thr, "fold_reports": fold_reports,
        "submission_validation": val_info,
    }
    write_json(args.report_path, report)
    append_action_log(args.action_log, f"NNv3 Training completed: model={args.model_path}.")
    print(f"\nDone! Model: {args.model_path}, Submission: {args.submission_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
