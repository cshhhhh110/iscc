"""Diagnostic analysis: per-type, per-length, per-span-count breakdown.

Usage:
    python 源码/diagnose.py
    python 源码/diagnose.py --use-oof  (requires OOF predictions from training)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    ANOMALY_TYPES,
    TYPE_TO_INDEX,
    TYPE_SPAN_LENGTHS,
    Prediction,
    evaluate_predictions,
    read_documents,
    span_iou,
)
from model_nn_v2 import LABEL_TO_TYPE, NUM_LABELS, O_LABEL, LogBiLSTM, collate_batch


def _bucket_length(n_lines: int) -> str:
    if n_lines <= 20:
        return "short (<=20)"
    elif n_lines <= 50:
        return "medium (21-50)"
    elif n_lines <= 100:
        return "long (51-100)"
    else:
        return "verylong (>100)"


def _bucket_span_len(span_len: int) -> str:
    if span_len <= 2:
        return "tiny (1-2)"
    elif span_len <= 4:
        return "short (3-4)"
    elif span_len <= 7:
        return "medium (5-7)"
    else:
        return "long (8+)"


def _bucket_span_count(n: int) -> str:
    if n == 1:
        return "1 span"
    elif n == 2:
        return "2 spans"
    else:
        return "3+ spans"


def analyze(docs, predictions, label: str) -> dict:
    """Break down metrics by type, doc length, span count, span length."""
    results: dict[str, dict] = defaultdict(lambda: {
        "n_docs": 0, "detect_correct": 0, "n_anomaly_true": 0, "n_anomaly_pred": 0,
        "ious": [], "true_types": [], "pred_types": [],
        "span_len_errors": [],  # true_span_len - pred_span_len
    })

    for doc, pred in zip(docs, predictions):
        # Buckets
        len_bucket = _bucket_length(len(doc.lines))
        span_count = _bucket_span_count(len(doc.spans))

        for bucket in ["ALL", len_bucket, span_count]:
            r = results[bucket]
            r["n_docs"] += 1
            if doc.has_anomaly == 1:
                r["n_anomaly_true"] += 1
            if pred.has_anomaly == 1:
                r["n_anomaly_pred"] += 1
            if doc.has_anomaly == pred.has_anomaly:
                r["detect_correct"] += 1

            if doc.has_anomaly == 1 and pred.has_anomaly == 1:
                iou = span_iou(
                    pred.primary_start_idx, pred.primary_end_idx,
                    doc.primary_start_idx, doc.primary_end_idx,
                )
                r["ious"].append(iou)
                r["true_types"].append(doc.primary_anomaly_type)
                r["pred_types"].append(pred.primary_anomaly_type)
                r["span_len_errors"].append(
                    (doc.primary_end_idx - doc.primary_start_idx + 1) -
                    (pred.primary_end_idx - pred.primary_start_idx + 1)
                )

        # Per-type buckets
        for span in doc.spans:
            type_bucket = f"type:{span.label}"
            span_len_bucket = f"spanlen:{_bucket_span_len(span.end - span.start + 1)}"
            for b in [type_bucket, span_len_bucket]:
                r2 = results[b]
                r2["n_docs"] += 1
                if doc.has_anomaly == 1:
                    r2["n_anomaly_true"] += 1
                if pred.has_anomaly == 1:
                    r2["n_anomaly_pred"] += 1
                if doc.has_anomaly == pred.has_anomaly:
                    r2["detect_correct"] += 1
                if doc.has_anomaly == 1 and pred.has_anomaly == 1:
                    iou2 = span_iou(pred.primary_start_idx, pred.primary_end_idx, span.start, span.end)
                    r2["ious"].append(iou2)
                    r2["true_types"].append(span.label)
                    r2["pred_types"].append(pred.primary_anomaly_type)

    # Compute metrics per bucket
    out = {}
    for bucket_name, r in sorted(results.items()):
        n = r["n_docs"]
        tp = r["detect_correct"]
        detect_acc = tp / max(1, n)
        ious = r["ious"]
        mean_iou = float(np.mean(ious)) if ious else 0.0
        type_match = sum(1 for t, p in zip(r["true_types"], r["pred_types"]) if t == p)
        type_acc = type_match / max(1, len(r["true_types"]))

        len_errors = r["span_len_errors"]
        mean_len_err = float(np.mean(len_errors)) if len_errors else 0.0
        mae_len_err = float(np.mean([abs(e) for e in len_errors])) if len_errors else 0.0

        out[bucket_name] = {
            "n": n,
            "anomaly_true": r["n_anomaly_true"],
            "anomaly_pred": r["n_anomaly_pred"],
            "detect_acc": round(detect_acc, 4),
            "mean_iou": round(mean_iou, 4),
            "type_acc": round(type_acc, 4),
            "mean_len_err": round(mean_len_err, 2),
            "mae_len_err": round(mae_len_err, 2),
            "iou_count": len(ious),
        }

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose model performance by slices.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if args.model_path is None:
        args.model_path = data_dir / "模型" / "model_bundle_nn_v2.joblib"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    train_docs = read_documents(data_dir / "train.csv", expect_labels=True)
    dense_train = joblib.load(data_dir / "缓存" / "dense_train.joblib")

    # Load model
    bundle = joblib.load(args.model_path)
    model = LogBiLSTM(
        input_dim=bundle["input_dim"], hidden_dim=bundle["hidden_dim"],
        num_labels=bundle["num_labels"], dropout=bundle["dropout"],
        num_lstm_layers=bundle["lstm_layers"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    # Predict
    print("Running inference on training data...")
    class SimpleDS:
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    loader = DataLoader(SimpleDS(dense_train), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    all_preds: list[Prediction] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict", unit="batch", dynamic_ncols=True):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]
            line_logits, doc_logits = model(features, mask)
            line_preds = torch.argmax(line_logits, dim=-1) * mask.long()
            doc_probs = torch.sigmoid(doc_logits).squeeze(-1)

            for i, did in enumerate(doc_ids):
                nv = int(mask[i].sum())
                lbls = line_preds[i, :nv].cpu().numpy().astype(np.int64)
                spans = []
                j = 0
                while j < len(lbls):
                    if lbls[j] == O_LABEL:
                        j += 1
                        continue
                    lid = lbls[j]
                    s = j
                    while j < len(lbls) and lbls[j] == lid:
                        j += 1
                    e = j - 1
                    if lid in LABEL_TO_TYPE:
                        spans.append((s, e, LABEL_TO_TYPE[lid]))
                if not spans:
                    all_preds.append(Prediction(str(did), 0, -1, -1, "none", ""))
                else:
                    p = spans[0]
                    all_spans = ";".join(f"{s}|{e}|{t}" for s, e, t in spans)
                    all_preds.append(Prediction(str(did), 1, p[0], p[1], p[2], all_spans))

    # Overall
    metrics = evaluate_predictions(train_docs, all_preds)
    print(f"\n{'='*70}")
    print(f"Overall (in-sample): score={metrics['score']:.4f}, F1_d={metrics['f1_detect']:.4f}, IoU={metrics['iou_loc']:.4f}, F1_t={metrics['f1_type']:.4f}")

    # Breakdown
    breakdown = analyze(train_docs, all_preds, "train")
    sections = [
        ("\n--- By Document Length ---", [k for k in breakdown if "short" in k or "medium" in k or "long" in k or "verylong" in k]),
        ("\n--- By Span Count ---", [k for k in breakdown if "span" in k and "spanlen" not in k]),
        ("\n--- By Anomaly Type ---", [k for k in breakdown if k.startswith("type:")]),
        ("\n--- By Span Length ---", [k for k in breakdown if k.startswith("spanlen:")]),
    ]

    for title, keys in sections:
        print(title)
        print(f"{'Bucket':<30} {'N':>6} {'DetAcc':>8} {'IoU':>8} {'TypeAcc':>8} {'LenErr':>8} {'MAE':>8}")
        print("-" * 80)
        for k in keys:
            if k not in breakdown:
                continue
            v = breakdown[k]
            print(f"{k:<30} {v['n']:>6} {v['detect_acc']:>8.4f} {v['mean_iou']:>8.4f} {v['type_acc']:>8.4f} {v['mean_len_err']:>8.2f} {v['mae_len_err']:>8.2f}")

    # Summary: worst buckets
    print(f"\n{'='*70}")
    print("Top issues:")
    issues = []
    for k, v in breakdown.items():
        if k == "ALL" or v["iou_count"] < 10:
            continue
        issues.append((k, v["mean_iou"], v["type_acc"], v["mae_len_err"], v["n"]))
    issues.sort(key=lambda x: x[1])  # sort by IoU ascending
    for k, iou, ta, mae, n in issues[:15]:
        print(f"  {k:<35} IoU={iou:.4f}  TypeAcc={ta:.4f}  MAE={mae:.2f}  (n={n})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
