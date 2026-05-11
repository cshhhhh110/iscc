from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import ID_COL, LABEL_COL, append_action_log, make_feature_frame, read_table, validate_prediction_frame
from tree_v1_7_core import VERSION, load_bundle, log_key_metrics


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=f"Predict with {VERSION} XGBoost+LightGBM ensemble.")
    p.add_argument("--data-dir", type=Path, default=root)
    p.add_argument("--test-file", type=str, default="test_data.csv")
    p.add_argument("--sample-file", type=str, default="sample_submission.csv")
    p.add_argument("--bundle-path", type=Path, default=root / "模型" / f"tree_bundle_{VERSION}.pkl")
    p.add_argument("--output-path", type=Path, default=root / "提交结果" / f"submission_tree_{VERSION}.csv")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    args = p.parse_args()

    if args.smoke:
        args.output_path = args.output_path.with_name("smoke_" + args.output_path.name)
        args.bundle_path = args.bundle_path.with_name("smoke_" + args.bundle_path.name)
    return args


def main() -> int:
    args = parse_args()
    bundle = load_bundle(args.bundle_path)
    data_dir = args.data_dir.resolve()
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    label_names = list(bundle["label_names"])
    feature_cols = list(bundle["feature_columns"])

    x_test_raw = make_feature_frame(test_df, feature_cols)
    x_np = x_test_raw.to_numpy(dtype=np.float32)
    x_df = pd.DataFrame(x_np, columns=feature_cols)

    n_models = len(bundle["xgb_models"])
    probs = np.zeros((len(test_df), len(label_names)), dtype=np.float32)

    for model in bundle["xgb_models"]:
        probs += model.predict_proba(x_np).astype(np.float32) / (n_models * 2)
    for model in bundle["lgbm_models"]:
        probs += model.predict_proba(x_df).astype(np.float32) / (n_models * 2)

    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8", newline="") as f:
        submission.to_csv(f, index=False)

    append_action_log(args.action_log,
        f"{VERSION} predict done: output={args.output_path}, xgb_models={n_models}, lgbm_models={n_models}.")

    log_key_metrics(root=data_dir, metrics={
        "version": VERSION, "stage": "predict", "model": "xgb+lgbm_avg",
        "n_features": len(feature_cols), "seeds": "-", "folds": n_models,
        "local_acc": "-", "local_macro_f1": "-", "weak_f1": "-",
        "platform_score": "-", "notes": f"submission={args.output_path.name}",
    })

    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
