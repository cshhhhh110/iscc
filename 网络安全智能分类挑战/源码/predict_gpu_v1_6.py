from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import ID_COL, LABEL_COL, append_action_log, make_feature_frame, read_table, validate_prediction_frame
from gpu_v1_6_core import (
    VERSION, apply_quantile, build_model, log_key_metrics,
    predict_probs, torch_load_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=f"Predict with {VERSION} FT-Transformer.")
    p.add_argument("--data-dir", type=Path, default=root)
    p.add_argument("--test-file", type=str, default="test_data.csv")
    p.add_argument("--sample-file", type=str, default="sample_submission.csv")
    p.add_argument("--bundle-path", type=Path,
                   default=root / "模型" / f"gpu_ft_bundle_{VERSION}.pt")
    p.add_argument("--output-path", type=Path,
                   default=root / "提交结果" / f"submission_gpu_{VERSION}.csv")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    args = p.parse_args()

    if args.smoke:
        args.output_path = args.output_path.with_name("smoke_" + args.output_path.name)
        args.bundle_path = args.bundle_path.with_name("smoke_" + args.bundle_path.name)
        args.batch_size = min(args.batch_size, 1024)
    return args


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
        raise RuntimeError("CUDA not available.")

    bundle = torch_load_bundle(args.bundle_path)
    data_dir = args.data_dir.resolve()
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    label_names = list(bundle["label_names"])
    feature_cols = list(bundle["feature_columns"])
    dropped = bundle.get("dropped_features", [])

    for c in dropped:
        if c in test_df.columns:
            test_df.drop(columns=c, inplace=True)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)

    probs = np.zeros((len(test_df), len(label_names)), dtype=np.float32)
    fold_iter = bundle["fold_models"]
    if args.smoke:
        fold_iter = fold_iter[:min(2, len(fold_iter))]

    for record in tqdm(fold_iter, desc=f"{VERSION} predict", unit="fold", dynamic_ncols=True):
        model = build_model(len(feature_cols), len(label_names), bundle["config"]).to(device)
        model.load_state_dict(record["state_dict"])
        x_norm = apply_quantile(x_test_raw, record["qt"])
        probs += predict_probs(model, x_norm, device, args.batch_size, args.num_workers) / len(fold_iter)

    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)
    append_action_log(args.action_log,
        f"{VERSION} predict done: output={args.output_path}, folds={len(fold_iter)}.")

    log_key_metrics(root=data_dir, metrics={
        "version": VERSION, "stage": "predict", "model": "ft_simplified",
        "n_features": len(feature_cols), "seeds": "-", "folds": len(fold_iter),
        "local_acc": "-", "local_macro_f1": "-", "weak_f1": "-",
        "platform_score": "-", "notes": f"submission={args.output_path.name}",
    })

    print(f"Wrote submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
