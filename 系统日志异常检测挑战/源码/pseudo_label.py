"""Generate pseudo-labeled training data from test set predictions.

Uses the current model to predict on test.csv, filters high-confidence samples,
and creates an augmented training file.

Usage:
    python 源码/pseudo_label.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np

from common import (
    ANOMALY_TYPES,
    ID_COL,
    TEXT_COL,
    SUBMISSION_COLUMNS,
    decode_predictions,
    predict_raw_scores,
    read_documents,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate pseudo-labeled training data.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle.joblib")
    parser.add_argument("--output", type=Path, default=default_root / "pseudo_train.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    # Confidence thresholds
    parser.add_argument("--doc-low", type=float, default=0.10, help="Doc prob below this → confident normal")
    parser.add_argument("--doc-high", type=float, default=0.90, help="Doc prob above this → confident anomaly")
    parser.add_argument("--span-conf", type=float, default=0.70, help="Min span score for confident positive")
    # Safety caps
    parser.add_argument("--max-pseudo", type=int, default=3000, help="Max pseudo-labeled samples to add")
    return parser.parse_args()


def check_anomaly_confidence(
    doc_prob: float,
    line_scores: np.ndarray,
    thresholds: dict,
    doc: object,
) -> tuple[bool, float]:
    """Check if a predicted anomaly is confident enough for pseudo-labeling.
    Returns (is_confident, confidence_score).
    """
    # Doc-level confidence
    if doc_prob < 0.5:
        return True, 1.0 - doc_prob

    # Anomaly detected: check span and type confidence
    n_lines = len(doc.lines)
    if n_lines == 0:
        return False, 0.0

    # Find the strongest type
    max_type_score = 0.0
    for type_idx, label in enumerate(ANOMALY_TYPES):
        scores = line_scores[:, type_idx]
        max_type_score = max(max_type_score, float(scores.max()))

    # Combined confidence
    confidence = doc_prob * 0.4 + max_type_score * 0.6
    return confidence > (doc_prob * 0.4 + 0.6 * 0.3), confidence


def predict_and_label(args) -> tuple[list[dict], dict]:
    """Predict on test set and return confident pseudo-labels."""
    print("Loading model and test data...")
    bundle = joblib.load(args.model_path)
    test_docs = read_documents(args.data_dir / args.test_file, expect_labels=False)

    print(f"Predicting on {len(test_docs)} test documents...")
    doc_probs, line_scores = predict_raw_scores(
        test_docs, bundle["doc_model"], bundle["type_models"],
        batch_size=args.batch_size, desc="predict test",
    )

    print("Decoding predictions...")
    predictions = decode_predictions(test_docs, doc_probs, line_scores, bundle["thresholds"])

    stats = {"total": len(test_docs), "confident_normal": 0, "confident_anomaly": 0, "skipped": 0}
    pseudo_rows = []

    for i, (doc, pred, doc_prob, scores) in enumerate(zip(test_docs, predictions, doc_probs, line_scores)):
        if len(pseudo_rows) >= args.max_pseudo:
            stats["skipped"] += 1
            continue

        is_conf, conf = check_anomaly_confidence(doc_prob, scores, bundle["thresholds"], doc)

        if pred.has_anomaly == 0 and doc_prob <= args.doc_low:
            # Confident normal
            pseudo_rows.append({
                ID_COL: str(doc.doc_id),
                TEXT_COL: "\n".join(doc.lines),
                "has_anomaly": "0",
                "primary_start_idx": "-1",
                "primary_end_idx": "-1",
                "primary_anomaly_type": "none",
                "all_spans": "",
            })
            stats["confident_normal"] += 1

        elif pred.has_anomaly == 1 and doc_prob >= args.doc_high and conf >= 0.5:
            # Confident anomaly — use model's predicted spans
            pseudo_rows.append({
                ID_COL: str(doc.doc_id),
                TEXT_COL: "\n".join(doc.lines),
                "has_anomaly": str(pred.has_anomaly),
                "primary_start_idx": str(pred.primary_start_idx),
                "primary_end_idx": str(pred.primary_end_idx),
                "primary_anomaly_type": pred.primary_anomaly_type,
                "all_spans": pred.all_spans,
            })
            stats["confident_anomaly"] += 1
        else:
            stats["skipped"] += 1

    return pseudo_rows, stats


def main() -> int:
    args = parse_args()

    pseudo_rows, stats = predict_and_label(args)

    # Load original training data
    print("Loading original training data...")
    train_path = args.data_dir / args.train_file
    with train_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_rows = list(reader)
    print(f"  Original: {len(original_rows)} rows")

    # Merge and save
    all_rows = original_rows + pseudo_rows
    print(f"  Pseudo:   {len(pseudo_rows)} (normal={stats['confident_normal']}, anomaly={stats['confident_anomaly']})")
    print(f"  Skipped:  {stats['skipped']}")
    print(f"  Total:    {len(all_rows)} rows")

    fieldnames = list(original_rows[0].keys())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nSaved pseudo-labeled training data to: {args.output}")

    # Print confidence distribution
    if stats["confident_anomaly"] > 0:
        print(f"\nPseudo-label coverage: {len(pseudo_rows)}/{stats['total']} "
              f"({100*len(pseudo_rows)/stats['total']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
