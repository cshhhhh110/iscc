from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from common import ID_COL, LABEL_COL, append_action_log, make_feature_frame, read_table, validate_prediction_frame
from gpu_v1_5_core import (
    VERSION, apply_robust_stats, build_ft_transformer, deduplicate_features,
    log_key_metrics, predict_ft_probs, torch_load_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Predict with {VERSION} FT+CatBoost ensemble.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--ft-bundle-path", type=Path,
                        default=root / "模型" / f"gpu_ft_bundle_{VERSION}.pt")
    parser.add_argument("--cb-bundle-path", type=Path,
                        default=root / "模型" / f"gpu_cb_bundle_{VERSION}.cbm")
    parser.add_argument("--output-path", type=Path,
                        default=root / "提交结果" / f"submission_gpu_{VERSION}.csv")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    args = parser.parse_args()

    if args.smoke:
        args.output_path = args.output_path.with_name("smoke_" + args.output_path.name)
        args.ft_bundle_path = args.ft_bundle_path.with_name("smoke_" + args.ft_bundle_path.name)
        args.cb_bundle_path = args.cb_bundle_path.with_name("smoke_" + args.cb_bundle_path.name)
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
        raise RuntimeError("CUDA not available. Use iscc-gpu or --allow-cpu.")

    ft_bundle = torch_load_bundle(args.ft_bundle_path)
    cb_bundle = None
    if args.cb_bundle_path.exists():
        with open(args.cb_bundle_path, "rb") as f:
            cb_bundle = pickle.load(f)

    data_dir = args.data_dir.resolve()
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    label_names = list(ft_bundle["label_names"])
    feature_cols_raw = list(ft_bundle["feature_columns"])
    dropped = ft_bundle.get("dropped_features", [])
    n_features = len(feature_cols_raw)

    # Apply same feature selection as training
    test_df_dedup = test_df.copy()
    for c in dropped:
        if c in test_df_dedup.columns:
            test_df_dedup.drop(columns=c, inplace=True)
    x_test_raw = make_feature_frame(test_df_dedup, feature_cols_raw).to_numpy(dtype=np.float32)
    # Also prepare raw (unnormalized) for CatBoost
    x_test_raw_cb = x_test_raw.copy()

    probs = np.zeros((len(test_df), len(label_names)), dtype=np.float32)
    ft_probs = np.zeros_like(probs)

    # FT-Transformer prediction
    fold_iter = ft_bundle["fold_models"]
    if args.smoke:
        fold_iter = fold_iter[:min(2, len(fold_iter))]
    for record in tqdm(fold_iter, desc=f"{VERSION} FT predict", unit="fold", dynamic_ncols=True):
        model = build_ft_transformer(n_features, len(label_names), ft_bundle["config"]).to(device)
        model.load_state_dict(record["state_dict"])
        x_norm = apply_robust_stats(
            x_test_raw,
            np.asarray(record["center"], dtype=np.float32).reshape(1, -1),
            np.asarray(record["scale"], dtype=np.float32).reshape(1, -1),
        )
        ft_probs += predict_ft_probs(model, x_norm, device, args.batch_size, args.num_workers) / len(fold_iter)
    probs += ft_probs

    # CatBoost prediction
    if cb_bundle is not None:
        cb_models = cb_bundle["fold_models"]
        if args.smoke:
            cb_models = cb_models[:min(2, len(cb_models))]
        cb_probs = np.zeros_like(probs)
        for model in tqdm(cb_models, desc=f"{VERSION} CB predict", unit="fold", dynamic_ncols=True):
            cb_probs += model.predict_proba(x_test_raw_cb).astype(np.float32) / len(cb_models)
        probs = (probs + cb_probs) / 2.0
        ensemble_label = "ft+cb_avg"
    else:
        ensemble_label = "ft_only"

    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)
    append_action_log(args.action_log,
        f"{VERSION} predict done: ensemble={ensemble_label}, output={args.output_path}, "
        f"ft_folds={len(fold_iter)}.")

    log_key_metrics(root=data_dir, metrics={
        "version": VERSION, "stage": "predict", "model": ensemble_label,
        "n_features": n_features, "seeds": "-", "folds": len(fold_iter),
        "local_acc": "-", "local_macro_f1": "-", "weak_f1": "-",
        "platform_score": "-", "notes": f"submission={args.output_path.name}",
    })

    print(f"Wrote submission: {args.output_path} ({ensemble_label})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
