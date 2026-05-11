from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import ID_COL, LABEL_COL, append_action_log, make_feature_frame, read_table, validate_prediction_frame


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Generate submission from a saved model bundle.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "模型" / "model_bundle.joblib")
    parser.add_argument("--output-path", type=Path, default=default_root / "提交结果" / "submission_reproduced.csv")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    return parser.parse_args()


def stack_probabilities(probabilities_by_model: dict[str, np.ndarray], model_names: list[str]) -> np.ndarray:
    return np.concatenate([probabilities_by_model[name] for name in model_names], axis=1)


def predict_from_bundle(bundle: dict, x_test: pd.DataFrame) -> np.ndarray:
    if "base_models" in bundle and "meta_model" in bundle:
        model_names = bundle["base_model_names"]
        base_models = bundle["base_models"]
        stack_test = stack_probabilities(
            {name: base_models[name].predict_proba(x_test).astype(np.float32) for name in model_names},
            model_names,
        )
        return bundle["meta_model"].predict_proba(stack_test)

    if "fold_models" in bundle:
        model_weights = bundle.get("model_weights")
        if model_weights is None:
            model_names = list(bundle["fold_models"][0]["models"].keys())
            model_weights = {name: 1.0 for name in model_names}
        else:
            model_names = list(model_weights.keys())
        total_weight = float(sum(model_weights[name] for name in model_names))
        probs = None
        for fold in bundle["fold_models"]:
            fold_probs = None
            for name in model_names:
                model_probs = fold["models"][name].predict_proba(x_test)
                fold_probs = (
                    model_weights[name] * model_probs
                    if fold_probs is None
                    else fold_probs + model_weights[name] * model_probs
                )
            fold_probs = fold_probs / total_weight
            probs = fold_probs if probs is None else probs + fold_probs
        return probs / len(bundle["fold_models"])

    raise KeyError("Unsupported model bundle format.")


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    bundle = joblib.load(args.model_path)

    feature_cols = bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"test data is missing model feature columns: {missing[:5]}")

    x_test = make_feature_frame(test_df, feature_cols)
    probs = predict_from_bundle(bundle, x_test)
    labels = np.array(bundle["label_names"])[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, bundle["label_names"], list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False, encoding="utf-8")
    append_action_log(args.action_log, f"Prediction completed: output={args.output_path}.")
    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
