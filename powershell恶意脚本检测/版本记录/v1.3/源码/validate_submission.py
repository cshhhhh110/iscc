from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import ARTIFACT_VERSION, LABELS, RESULT_DIR, TEST_PATH, append_log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate official submission format.")
    parser.add_argument("--submission", type=Path, default=RESULT_DIR / f"submission_{ARTIFACT_VERSION}.csv")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--no-log", action="store_true", help="Do not append project log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    submission = pd.read_csv(args.submission)
    test_df = pd.read_csv(args.test)

    expected_columns = ["name", "label"]
    if list(submission.columns) != expected_columns:
        raise ValueError(f"submission columns must be {expected_columns}, got {list(submission.columns)}")
    if len(submission) != len(test_df):
        raise ValueError(f"submission row count {len(submission)} != test row count {len(test_df)}")
    if submission["name"].tolist() != test_df["name"].tolist():
        raise ValueError("submission names do not exactly match test data names/order")
    if submission["label"].isna().any():
        raise ValueError("submission contains empty labels")
    labels = set(submission["label"].astype(int).unique().tolist())
    if not labels.issubset(set(LABELS)):
        raise ValueError(f"submission labels outside {LABELS}: {sorted(labels - set(LABELS))}")

    counts = submission["label"].value_counts().sort_index().to_dict()
    if not args.no_log:
        append_log(f"validated PowerShell submission: rows={len(submission)}, label_counts={counts}")
    print(f"OK rows={len(submission)} label_counts={counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
