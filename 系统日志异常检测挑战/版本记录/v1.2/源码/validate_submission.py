from __future__ import annotations

import argparse
from pathlib import Path

from common import read_documents, read_sample_columns, validate_submission_file


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Validate ISCC log anomaly submission CSV.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--submission-path", type=Path, default=default_root / "提交结果" / "submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    expected_columns = read_sample_columns(data_dir / args.sample_file)
    info = validate_submission_file(args.submission_path, test_docs, expected_columns)
    print(f"Submission validation passed: rows={info['rows']}, columns={info['columns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
