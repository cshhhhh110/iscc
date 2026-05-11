from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import ID_COL, LABEL_COL, append_action_log, make_feature_frame, read_table, validate_prediction_frame
from gpu_fttransformer_v1_4_core import (
    VERSION,
    apply_robust_stats,
    build_model,
    predict_probs,
    torch_load_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Predict with {VERSION} FT-Transformer bundle.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_fttransformer_model_bundle_{VERSION}.pt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root / "\u63d0\u4ea4\u7ed3\u679c" / f"submission_gpu_fttransformer_{VERSION}.csv",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    args = parser.parse_args()

    if args.smoke:
        prefix = "smoke_"
        args.output_path = args.output_path.with_name(prefix + args.output_path.name)
        args.batch_size = min(args.batch_size, 1024)
    return args


def build_from_bundle(bundle: dict, n_features: int, n_classes: int) -> nn.Module:
    config = bundle["config"]
    return build_model(n_features=n_features, n_classes=n_classes, config=config)


def write_csv_with_retry(frame: pd.DataFrame, path: Path, retries: int = 5, delay: float = 1.0) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, retries + 1):
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available. Use iscc-gpu or pass --allow-cpu.")

    bundle = torch_load_bundle(args.model_path)
    data_dir = args.data_dir.resolve()
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    label_names = list(bundle["label_names"])
    feature_cols = list(bundle["feature_columns"])
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"test data is missing feature columns: {missing[:5]}")

    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    probs = np.zeros((len(test_df), len(label_names)), dtype=np.float32)

    fold_iter = bundle["fold_models"]
    if args.smoke:
        fold_iter = fold_iter[: min(2, len(fold_iter))]
    for record in tqdm(fold_iter, desc=f"{VERSION} predict", unit="fold", dynamic_ncols=True):
        model = build_from_bundle(bundle, len(feature_cols), len(label_names)).to(device)
        model.load_state_dict(record["state_dict"])
        x_test = apply_robust_stats(
            x_test_raw,
            np.asarray(record["center"], dtype=np.float32).reshape(1, -1),
            np.asarray(record["scale"], dtype=np.float32).reshape(1, -1),
        )
        probs += predict_probs(model, x_test, device, args.batch_size, args.num_workers) / len(fold_iter)

    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(
        submission,
        test_df,
        label_names,
        list(sample_df.columns),
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)
    append_action_log(
        args.action_log,
        f"{VERSION} predict completed: output={args.output_path}, bundle={args.model_path}, folds={len(fold_iter)}.",
    )
    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

