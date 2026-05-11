"""Prediction script for BiLSTM v2 model.

Loads model bundle and pre-built dense test features, generates submission.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    Prediction,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_submission,
)
from model_nn_v2 import LABEL_TO_TYPE, NUM_LABELS, O_LABEL, LogBiLSTM, collate_batch


def _labels_to_spans(labels: np.ndarray) -> list[tuple[int, int, str]]:
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


def predict(model: LogBiLSTM, loader: DataLoader, device: torch.device) -> list[Prediction]:
    """Generate predictions for all test documents."""
    model.eval()
    all_preds: list[Prediction] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict test", unit="batch", dynamic_ncols=True):
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]

            line_logits, doc_logits = model(features, mask)
            line_preds = torch.argmax(line_logits, dim=-1) * mask.long()
            doc_probs = torch.sigmoid(doc_logits).squeeze(-1)

            for i, doc_id in enumerate(doc_ids):
                n_valid = int(mask[i].sum())
                pred_labels = line_preds[i, :n_valid].cpu().numpy().astype(np.int64)
                doc_p = float(doc_probs[i])

                spans = _labels_to_spans(pred_labels)

                if not spans:
                    all_preds.append(Prediction(
                        doc_id=str(doc_id), has_anomaly=0,
                        primary_start_idx=-1, primary_end_idx=-1,
                        primary_anomaly_type="none", all_spans="",
                    ))
                else:
                    primary = spans[0]
                    all_spans_str = ";".join(f"{s}|{e}|{t}" for s, e, t in spans)
                    all_preds.append(Prediction(
                        doc_id=str(doc_id), has_anomaly=1,
                        primary_start_idx=primary[0], primary_end_idx=primary[1],
                        primary_anomaly_type=primary[2], all_spans=all_spans_str,
                    ))

    return all_preds


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with BiLSTM v2 model.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if args.model_path is None:
        args.model_path = data_dir / "模型" / "model_bundle_nn_v2.joblib"
    if args.output_path is None:
        args.output_path = data_dir / "提交结果" / "submission_nn_v2.csv"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model bundle
    print(f"Loading model from {args.model_path}...")
    bundle = joblib.load(args.model_path)
    if bundle.get("model") != "BiLSTM_v2":
        raise ValueError(f"Expected BiLSTM_v2 model, got {bundle.get('model')}")

    # Build model
    model = LogBiLSTM(
        input_dim=bundle["input_dim"],
        hidden_dim=bundle["hidden_dim"],
        num_labels=bundle["num_labels"],
        dropout=bundle["dropout"],
        num_lstm_layers=bundle["lstm_layers"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    print(f"  Model: {bundle['model']}, input_dim={bundle['input_dim']}, hidden_dim={bundle['hidden_dim']}")

    # Load dense test features
    cache_dir = data_dir / "缓存"
    dense_test_path = cache_dir / "dense_test.joblib"
    if not dense_test_path.exists():
        print(f"Dense test features not found at {dense_test_path}")
        print("Run 'python 源码/build_dense_features.py' first.")
        return 1

    print(f"Loading dense test features from {dense_test_path}...")
    dense_test = joblib.load(dense_test_path)

    # Verify document order
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    for i, (feat, doc) in enumerate(zip(dense_test, test_docs)):
        if str(feat["doc_id"]) != str(doc.doc_id):
            raise ValueError(f"Test order mismatch at {i}: {feat['doc_id']} vs {doc.doc_id}")
    print(f"  {len(dense_test)} test documents, order verified")

    # Create DataLoader
    class SimpleDataset:
        def __init__(self, data):
            self.data = data
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            return self.data[idx]

    loader = DataLoader(
        SimpleDataset(dense_test),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
        num_workers=0,
    )

    # Predict
    print("Predicting...")
    predictions = predict(model, loader, device)

    # Write submission
    write_submission(args.output_path, predictions)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    validate_submission_file(args.output_path, test_docs, sample_columns)
    print(f"Submission written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
