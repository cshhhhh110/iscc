from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import MODEL_DIR, RESULT_DIR, TEST_PATH, append_log, load_test, predict_from_bundle, validate_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PowerShell test labels.")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "model_bundle.joblib")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / "submission.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    features = bundle["feature_columns"]

    test_df = load_test(args.test)
    validate_features(test_df, features)
    X = test_df[features].astype("int16")
    pred = predict_from_bundle(bundle, X).astype(int)

    submission = pd.DataFrame({"name": test_df["name"], "label": pred})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False, encoding="utf-8")
    append_log(
        f"generated PowerShell submission with {len(submission)} rows: {args.output}; "
        f"class_bias={bundle.get('class_bias', [1.0, 1.0, 1.0])}"
    )
    print(f"Saved submission: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
