from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from common import (
    append_action_log,
    decode_predictions,
    predict_raw_scores,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_submission,
)


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate submission from saved ISCC log anomaly model.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle.joblib")
    parser.add_argument("--output-path", type=Path, default=default_root / "提交结果" / "submission_reproduced.csv")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    bundle = joblib.load(args.model_path)
    doc_probs, line_scores = predict_raw_scores(
        test_docs,
        bundle["doc_model"],
        bundle["type_models"],
        batch_size=args.batch_size,
        desc="predict test",
    )
    predictions = decode_predictions(test_docs, doc_probs, line_scores, bundle["thresholds"])
    write_submission(args.output_path, predictions)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    info = validate_submission_file(args.output_path, test_docs, sample_columns)
    append_action_log(args.action_log, f"Prediction completed: output={args.output_path}, rows={info['rows']}.")
    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
