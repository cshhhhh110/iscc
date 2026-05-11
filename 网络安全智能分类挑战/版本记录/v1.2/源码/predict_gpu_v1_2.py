from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import ID_COL, LABEL_COL, append_action_log, ensure_feature_columns, make_feature_frame, read_table, validate_prediction_frame
from gpu_v1_2_core import VERSION, predict_probs_from_bundle, torch_load_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Predict with the {VERSION} pure-GPU bundle.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument(
        "--bundle-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_model_bundle_{VERSION}.pt",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root / "\u63d0\u4ea4\u7ed3\u679c" / f"submission_gpu_{VERSION}.csv",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


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
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available. Use the iscc-gpu environment or pass --allow-cpu.")

    bundle = torch_load_bundle(args.bundle_path)
    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)

    feature_cols = ensure_feature_columns(train_df, test_df)
    x_test = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    final_probs, _, ensemble_meta = predict_probs_from_bundle(
        bundle,
        x_test,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    label_names = list(bundle["label_names"])
    labels = np.asarray(label_names)[np.argmax(final_probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)
    append_action_log(
        args.action_log,
        f"{VERSION} predict completed: source={ensemble_meta.get('selected_source', 'bundle')}, output={args.output_path}.",
    )
    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
