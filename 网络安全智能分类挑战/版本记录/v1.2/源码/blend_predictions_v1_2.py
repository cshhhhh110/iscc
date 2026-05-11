from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder

from common import ID_COL, LABEL_COL, append_action_log, ensure_feature_columns, make_feature_frame, read_table, validate_prediction_frame
from gpu_v1_2_core import VERSION, blend_probs, predict_probs_from_bundle, search_blend_weight, torch_load_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Blend pure-GPU probabilities for {VERSION}.")
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
        "--ft-oof-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_ft_oof_probs_{VERSION}.npy",
    )
    parser.add_argument(
        "--ft-test-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_ft_test_probs_{VERSION}.npy",
    )
    parser.add_argument(
        "--mlp-oof-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_mlp_oof_probs_{VERSION}.npy",
    )
    parser.add_argument(
        "--mlp-test-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_mlp_test_probs_{VERSION}.npy",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root / "\u63d0\u4ea4\u7ed3\u679c" / f"submission_blend_{VERSION}.csv",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"blend_metadata_{VERSION}.json",
    )
    parser.add_argument("--weight-ft", type=float, default=None)
    parser.add_argument("--weight-mlp", type=float, default=None)
    parser.add_argument("--search-weights", action="store_true", help="Search OOF blend weight from saved probs.")
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


def maybe_load_probs(path: Path) -> np.ndarray | None:
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


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
    label_names = list(bundle["label_names"])

    ensemble_meta = bundle.get("ensemble", {})
    if args.weight_ft is not None or args.weight_mlp is not None:
        weight_ft = 0.5 if args.weight_ft is None else float(args.weight_ft)
        weight_mlp = 1.0 - weight_ft if args.weight_mlp is None else float(args.weight_mlp)
        weight_source = "manual"
    elif args.search_weights:
        ft_oof = maybe_load_probs(args.ft_oof_path)
        mlp_oof = maybe_load_probs(args.mlp_oof_path)
        if ft_oof is None or mlp_oof is None:
            raise FileNotFoundError("search_weights requires ft/mlp OOF probability files from training.")
        y_true = LabelEncoder().fit_transform(train_df[LABEL_COL].astype(str))
        search_best = search_blend_weight(y_true, ft_oof, mlp_oof)
        weight_ft = search_best["weight_a"]
        weight_mlp = search_best["weight_b"]
        weight_source = "search"
    else:
        weight_ft = float(ensemble_meta.get("selected_weight_ft", 1.0))
        weight_mlp = float(ensemble_meta.get("selected_weight_mlp", 0.0))
        weight_source = ensemble_meta.get("selected_source", "bundle")

    ft_test = maybe_load_probs(args.ft_test_path)
    mlp_test = maybe_load_probs(args.mlp_test_path)
    if ft_test is not None and mlp_test is not None:
        weight_sum = weight_ft + weight_mlp
        if weight_sum <= 0:
            raise ValueError("blend weights must sum to a positive value")
        final_probs = blend_probs(ft_test, mlp_test, weight_ft / weight_sum)
    else:
        final_probs, family_probs, _ = predict_probs_from_bundle(
            bundle,
            x_test,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            weight_ft=weight_ft,
            weight_mlp=weight_mlp,
        )
        if ft_test is None and "ft_transformer" in family_probs:
            ft_test = family_probs["ft_transformer"]
        if mlp_test is None and "residual_mlp" in family_probs:
            mlp_test = family_probs["residual_mlp"]

    labels = np.asarray(label_names)[np.argmax(final_probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)

    metadata = {
        "version": VERSION,
        "source": weight_source,
        "weights": {
            "ft_transformer": weight_ft,
            "residual_mlp": weight_mlp,
        },
        "bundle_path": str(args.bundle_path),
        "ft_oof_path": str(args.ft_oof_path),
        "mlp_oof_path": str(args.mlp_oof_path),
        "ft_test_path": str(args.ft_test_path),
        "mlp_test_path": str(args.mlp_test_path),
        "output_path": str(args.output_path),
        "rows": int(len(submission)),
    }
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"{VERSION} blend completed: source={weight_source}, output={args.output_path}.",
    )
    print(f"Wrote blended submission: {args.output_path}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
