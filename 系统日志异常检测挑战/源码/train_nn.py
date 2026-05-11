"""Train BiLSTM+CRF model for log anomaly span detection.

Usage:
    python 源码/train_nn.py  [--epochs 30] [--batch-size 16]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from common import (
    ANOMALY_TYPES,
    DEFAULT_SEED,
    FEATURE_CONFIG,
    append_action_log,
    parse_document,
    parse_documents_batch,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_json,
    write_submission,
)
from model_nn import (
    LABEL_TO_TYPE,
    NUM_LABELS,
    TYPE_TO_LABEL,
    LogBiLSTMCRF,
    build_line_features,
    labels_from_doc,
    tags_to_prediction,
)


def collate_batch(docs, cache_data=None):
    """Build padded batch for NN training."""
    cached_parsed, cached_contexts, id_to_idx = cache_data if cache_data else (None, None, None)

    features_batch: list[np.ndarray] = []
    labels_batch: list[np.ndarray] = []
    lengths: list[int] = []

    for doc in docs:
        if cached_parsed is not None and id_to_idx is not None:
            idx = id_to_idx[doc.doc_id]
            item = cached_parsed[idx]
            norm_texts = item["norm_lines"]
            line_num = item["line_numeric"]
        else:
            parsed = parse_document(doc)
            norm_texts = parsed["norm_lines"]
            line_num = parsed["line_numeric"]

        n = len(norm_texts)
        feats = build_line_features(norm_texts, line_num)
        labs = labels_from_doc(doc, n)

        features_batch.append(feats)
        labels_batch.append(labs)
        lengths.append(n)

    max_len = max(lengths)
    B = len(docs)
    D = features_batch[0].shape[1]

    padded_feats = np.zeros((B, max_len, D), dtype=np.float32)
    padded_labels = np.zeros((B, max_len), dtype=np.int64)
    mask = np.zeros((B, max_len), dtype=np.float32)

    for i in range(B):
        L = lengths[i]
        padded_feats[i, :L, :] = features_batch[i]
        padded_labels[i, :L] = labels_batch[i]
        mask[i, :L] = 1.0

    return (
        torch.from_numpy(padded_feats),
        torch.from_numpy(padded_labels),
        torch.from_numpy(mask),
    ), lengths


def train_epoch(model, optimizer, train_docs, batch_size, device, cache_data):
    model.train()
    total_loss = 0.0
    n_batches = 0

    order = np.random.permutation(len(train_docs))
    for start in tqdm(range(0, len(order), batch_size), desc="train", unit="batch", dynamic_ncols=True, leave=False):
        indices = order[start: start + batch_size]
        batch_docs = [train_docs[int(i)] for i in indices]
        (feats, labels, mask), _ = collate_batch(batch_docs, cache_data)
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)

        optimizer.zero_grad()
        loss, _ = model(feats, mask, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(1, n_batches)


@torch.no_grad()
def predict_batch(model, docs, batch_size, device, cache_data):
    model.eval()
    all_preds: list[np.ndarray] = []

    for start in range(0, len(docs), batch_size):
        batch_docs = docs[start: start + batch_size]
        (feats, _, mask), _ = collate_batch(batch_docs, cache_data)
        feats, mask = feats.to(device), mask.to(device)

        preds, _ = model(feats, mask)
        all_preds.extend(p.cpu().numpy() for p in preds)

    return all_preds


def compute_oof_score(train_docs, oof_preds):
    """Score OOF predictions using competition metric."""
    y_true = np.array([d.has_anomaly for d in train_docs], dtype=np.int32)
    y_pred = np.zeros(len(train_docs), dtype=np.int32)
    ious: list[float] = []
    true_types: list[str] = []
    pred_types: list[str] = []

    for i, (doc, tags) in enumerate(zip(train_docs, oof_preds)):
        has_a, s, e, ptype, _ = tags_to_prediction(doc, tags)
        y_pred[i] = has_a

        if has_a and doc.has_anomaly:
            inter = max(0, min(e, doc.primary_end_idx) - max(s, doc.primary_start_idx) + 1)
            union_val = max(e, doc.primary_end_idx) - min(s, doc.primary_start_idx) + 1
            ious.append(inter / union_val if union_val > 0 else 0.0)
            true_types.append(doc.primary_anomaly_type)
            pred_types.append(ptype)

    from sklearn.metrics import f1_score
    f1_d = f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0)
    iou = float(np.mean(ious)) if ious else 0.0
    f1_t = f1_score(true_types, pred_types, labels=ANOMALY_TYPES, average="macro", zero_division=0) if true_types else 0.0
    score = 0.15 * f1_d + 0.50 * iou + 0.35 * f1_t
    return {"score": score, "f1_detect": f1_d, "iou_loc": iou, "f1_type": f1_t}


def parse_args():
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train BiLSTM+CRF for log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle_nn.pth")
    parser.add_argument("--submission-path", type=Path, default=default_root / "提交结果" / "submission.csv")
    parser.add_argument("--report-path", type=Path, default=default_root / "模型" / "validation_report_nn.json")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dev-limit", type=int, default=0)
    return parser.parse_args()


def load_nn_cache(data_dir: Path, train_file: str = "train.csv"):
    """Load feature cache. Same format as train.py cache but we only need parsed docs."""
    cache_dir = data_dir / "缓存"
    prefix = Path(train_file).stem
    parsed_path = cache_dir / f"{prefix}_parsed.joblib"
    doc_ids_path = cache_dir / f"{prefix}_ids.joblib"

    if not (parsed_path.exists() and doc_ids_path.exists()):
        return None

    parsed = joblib.load(parsed_path)
    doc_ids = joblib.load(doc_ids_path)
    id_to_idx = {did: i for i, did in enumerate(doc_ids)}
    return parsed, None, id_to_idx  # (parsed, contexts=None, id_to_idx)


def main():
    args = parse_args()
    data_dir = args.data_dir.resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    append_action_log(args.action_log, "Training started for system log anomaly detection (v1.4 BiLSTM+CRF).")
    train_docs = read_documents(data_dir / args.train_file, expect_labels=True)
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    if args.dev_limit > 0:
        train_docs = train_docs[:args.dev_limit]

    # Load cache
    cache_data = load_nn_cache(data_dir, args.train_file)
    if cache_data:
        print(f"Using feature cache for {len(cache_data[0])} documents")

    y = np.array([d.has_anomaly for d in train_docs], dtype=np.int32)
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof_preds: list[np.ndarray | None] = [None] * len(train_docs)

    for fold_idx, (tr_idx, va_idx) in enumerate(
        tqdm(cv.split(np.zeros(len(train_docs)), y), total=args.folds, desc="CV folds", unit="fold", dynamic_ncols=True),
        start=1,
    ):
        train_fold = [train_docs[int(i)] for i in tr_idx]
        val_fold = [train_docs[int(i)] for i in va_idx]

        # Compute input dim from first document
        first_parsed = parse_document(train_fold[0])
        sample_feat = build_line_features(first_parsed["norm_lines"], first_parsed["line_numeric"])
        input_dim = sample_feat.shape[1]

        model = LogBiLSTMCRF(
            input_dim=input_dim, hidden_dim=args.hidden_dim,
            num_labels=NUM_LABELS, dropout=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

        best_loss = float("inf")
        best_state = None
        patience = 8

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, optimizer, train_fold, args.batch_size, device, cache_data)
            val_preds = predict_batch(model, val_fold, args.batch_size, device, cache_data)
            val_score = compute_oof_score(val_fold, val_preds)
            scheduler.step(train_loss)

            if train_loss < best_loss * 0.999:
                best_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 8
            else:
                patience -= 1
                if patience <= 0:
                    break

            if epoch % 5 == 0 or epoch == 1:
                print(f"  Fold {fold_idx} epoch {epoch}: loss={train_loss:.4f}, val_score={val_score['score']:.4f} "
                      f"(F1d={val_score['f1_detect']:.3f}, IoU={val_score['iou_loc']:.3f}, F1t={val_score['f1_type']:.3f})")

        # Load best and predict OOF
        model.load_state_dict(best_state)
        val_preds = predict_batch(model, val_fold, args.batch_size, device, cache_data)
        for local_pos, global_idx in enumerate(va_idx):
            oof_preds[int(global_idx)] = val_preds[local_pos]

        append_action_log(args.action_log, f"NN Fold {fold_idx} completed.")

    # OOF evaluation
    if any(p is None for p in oof_preds):
        raise RuntimeError("OOF collection failed")
    oof_final = [p for p in oof_preds if p is not None]
    oof_metrics = compute_oof_score(train_docs, oof_final)
    print(f"OOF: score={oof_metrics['score']:.4f}, F1d={oof_metrics['f1_detect']:.3f}, "
          f"IoU={oof_metrics['iou_loc']:.3f}, F1t={oof_metrics['f1_type']:.3f}")
    append_action_log(args.action_log, f"NN OOF score={oof_metrics['score']:.6f}")

    # Train full model
    print("Training full model...")
    full_model = LogBiLSTMCRF(
        input_dim=input_dim, hidden_dim=args.hidden_dim,
        num_labels=NUM_LABELS, dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(full_model.parameters(), lr=args.lr, weight_decay=1e-4)
    for epoch in range(1, 21):
        train_loss = train_epoch(full_model, optimizer, train_docs, args.batch_size, device, cache_data)
        if epoch % 5 == 0:
            print(f"  full epoch {epoch}: loss={train_loss:.4f}")

    # Predict test
    print("Predicting test...")
    test_preds = predict_batch(full_model, test_docs, args.batch_size, device, None)  # no cache for test
    predictions = []
    for doc, tags in zip(test_docs, test_preds):
        has_a, s, e, ptype, spans = tags_to_prediction(doc, tags)
        from common import Prediction
        predictions.append(Prediction(
            doc_id=str(doc.doc_id), has_anomaly=has_a,
            primary_start_idx=s, primary_end_idx=e,
            primary_anomaly_type=ptype, all_spans=spans,
        ))

    write_submission(args.submission_path, predictions)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    val_info = validate_submission_file(args.submission_path, test_docs, sample_columns)

    # Save model
    torch.save(full_model.state_dict(), args.model_path)
    report = {
        "model": "BiLSTM+CRF",
        "input_dim": input_dim,
        "hidden_dim": args.hidden_dim,
        "oof_metrics": oof_metrics,
        "submission_validation": val_info,
    }
    write_json(args.report_path, report)
    append_action_log(args.action_log, f"NN Training completed: model={args.model_path}, submission={args.submission_path}.")
    print("Training completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
