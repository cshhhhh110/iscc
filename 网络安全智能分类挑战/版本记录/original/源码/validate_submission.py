from __future__ import annotations

import argparse
from pathlib import Path

from common import LABEL_COL, read_table, validate_prediction_frame


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate a competition submission CSV.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--submission-path", type=Path, default=default_root / "提交结果" / "submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    submission = read_table(args.submission_path)

    allowed_labels = sorted(train_df[LABEL_COL].astype(str).unique().tolist())
    validate_prediction_frame(submission, test_df, allowed_labels, list(sample_df.columns))
    print(
        "Submission validation passed: "
        f"rows={len(submission)}, columns={list(submission.columns)}, labels={allowed_labels}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
