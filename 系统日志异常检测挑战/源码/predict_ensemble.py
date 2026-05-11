"""Ensemble prediction: average probabilities from multiple model bundles.

Usage:
    python 源码/predict_ensemble.py --models 模型/model_seed42.joblib 模型/model_seed123.joblib 模型/model_seed999.joblib
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from common import (
    ANOMALY_TYPES,
    decode_predictions,
    parse_documents_batch,
    predict_raw_scores,
    probability_of_positive,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_submission,
)
from common import make_doc_feature_matrix as _mk_doc
from common import make_line_feature_matrix as _mk_line


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Ensemble prediction for ISCC log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--models", type=str, nargs="+", required=True, help="Paths to model bundles")
    parser.add_argument("--output-path", type=Path, default=default_root / "提交结果" / "submission_ensemble.csv")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)

    # Load all models
    bundles = [joblib.load(Path(p)) for p in args.models]
    print(f"Loaded {len(bundles)} models")

    # Use thresholds from the first model
    thresholds = bundles[0]["thresholds"]

    # Accumulate probabilities across models
    doc_probs_sum = np.zeros(len(test_docs), dtype=np.float64)
    line_scores_sum: list[np.ndarray] = [np.zeros((len(doc.lines), len(ANOMALY_TYPES)), dtype=np.float64) for doc in test_docs]

    for bi, bundle in enumerate(bundles):
        print(f"Predicting model {bi+1}/{len(bundles)}...")
        doc_model = bundle["doc_model"]
        type_models = bundle["type_models"]

        doc_probs, line_scores = predict_raw_scores(
            test_docs, doc_model, type_models,
            batch_size=args.batch_size,
            desc=f"model {bi+1}/{len(bundles)}",
        )

        doc_probs_sum += doc_probs.astype(np.float64)
        for i, scores in enumerate(line_scores):
            line_scores_sum[i] += scores.astype(np.float64)

    # Average
    n = len(bundles)
    doc_probs_avg = (doc_probs_sum / n).astype(np.float32)
    line_scores_avg = [(s / n).astype(np.float32) for s in line_scores_sum]

    # Decode with averaged probabilities
    print("Decoding ensemble predictions...")
    predictions = decode_predictions(test_docs, doc_probs_avg, line_scores_avg, thresholds)
    write_submission(args.output_path, predictions)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    validate_submission_file(args.output_path, test_docs, sample_columns)
    print(f"Ensemble submission written to: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
