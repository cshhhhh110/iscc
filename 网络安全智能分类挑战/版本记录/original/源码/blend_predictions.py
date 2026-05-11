from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from common import ID_COL, LABEL_COL, make_feature_frame, read_table, validate_prediction_frame
from predict import predict_from_bundle


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Blend GPU neural network probabilities with sklearn probabilities.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--gpu-probs-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_test_probs.npy")
    parser.add_argument("--gpu-model-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_model_bundle.pt")
    parser.add_argument("--sklearn-probs-path", type=Path, default=default_root / "\u6a21\u578b" / "sklearn_test_probs.npy")
    parser.add_argument("--sklearn-model-path", type=Path, default=default_root / "\u6a21\u578b" / "model_bundle.joblib")
    parser.add_argument("--output-path", type=Path, default=default_root / "\u63d0\u4ea4\u7ed3\u679c" / "submission_blend.csv")
    parser.add_argument("--metadata-path", type=Path, default=default_root / "\u6a21\u578b" / "blend_metadata.json")
    parser.add_argument("--gpu-weight", type=float, default=0.60)
    parser.add_argument("--sklearn-weight", type=float, default=0.40)
    return parser.parse_args()


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_label_names(train_df: pd.DataFrame, gpu_model_path: Path) -> list[str]:
    if gpu_model_path.exists():
        bundle = torch_load(gpu_model_path)
        if "label_names" in bundle:
            return list(bundle["label_names"])
    return sorted(train_df[LABEL_COL].astype(str).unique().tolist())


def load_or_compute_sklearn_probs(args: argparse.Namespace, test_df: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    if args.sklearn_probs_path.exists():
        return np.load(args.sklearn_probs_path), f"loaded {args.sklearn_probs_path}"

    if not args.sklearn_model_path.exists():
        return None, "missing sklearn model bundle"

    bundle = joblib.load(args.sklearn_model_path)
    feature_cols = bundle["feature_columns"]
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"test data is missing sklearn feature columns: {missing[:5]}")
    x_test = make_feature_frame(test_df, feature_cols)
    probs = predict_from_bundle(bundle, x_test).astype(np.float32)
    args.sklearn_probs_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.sklearn_probs_path, probs)
    return probs, f"computed from {args.sklearn_model_path}"


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)

    if not args.gpu_probs_path.exists():
        raise FileNotFoundError(f"GPU probabilities not found: {args.gpu_probs_path}")

    label_names = load_label_names(train_df, args.gpu_model_path)
    gpu_probs = np.load(args.gpu_probs_path).astype(np.float32)
    sklearn_probs, sklearn_source = load_or_compute_sklearn_probs(args, test_df)

    if gpu_probs.shape != (len(test_df), len(label_names)):
        raise ValueError(
            f"GPU probabilities shape mismatch: got {gpu_probs.shape}, "
            f"expected {(len(test_df), len(label_names))}"
        )

    if sklearn_probs is None:
        final_probs = gpu_probs
        source = "gpu_only"
        effective_weights = {"gpu": 1.0, "sklearn": 0.0}
    else:
        if sklearn_probs.shape != gpu_probs.shape:
            raise ValueError(f"sklearn probabilities shape {sklearn_probs.shape} does not match GPU {gpu_probs.shape}")
        total_weight = args.gpu_weight + args.sklearn_weight
        if total_weight <= 0:
            raise ValueError("blend weights must sum to a positive value")
        final_probs = (args.gpu_weight * gpu_probs + args.sklearn_weight * sklearn_probs) / total_weight
        source = "gpu_sklearn_blend"
        effective_weights = {
            "gpu": args.gpu_weight / total_weight,
            "sklearn": args.sklearn_weight / total_weight,
        }

    pred = np.argmax(final_probs, axis=1)
    labels = np.array(label_names)[pred]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_path, index=False, encoding="utf-8")

    metadata = {
        "source": source,
        "gpu_probs_path": str(args.gpu_probs_path),
        "sklearn_source": sklearn_source,
        "output_path": str(args.output_path),
        "rows": int(len(submission)),
        "label_names": label_names,
        "weights": effective_weights,
    }
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Wrote blended submission: {args.output_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
