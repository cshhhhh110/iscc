from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from common import ID_COL, LABEL_COL, make_feature_frame, read_table, validate_prediction_frame
from train_gpu_fttransformer_v1_1 import FTTransformerClassifier, VERSION, apply_robust_stats


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Predict with {VERSION} FT-Transformer bundle.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
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
        default=root / "\u63d0\u4ea4\u7ed3\u679c" / f"submission_gpu_fttransformer_reproduced_{VERSION}.csv",
    )
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def torch_load(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def build_from_bundle(bundle: dict, n_features: int, n_classes: int) -> FTTransformerClassifier:
    config = bundle["config"]
    return FTTransformerClassifier(
        n_features=n_features,
        n_classes=n_classes,
        d_token=config["d_token"],
        n_blocks=config["n_blocks"],
        n_heads=config["n_heads"],
        d_ffn=config["d_ffn"],
        attention_dropout=config["attention_dropout"],
        residual_dropout=config["residual_dropout"],
        ffn_dropout=config["ffn_dropout"],
        head_dropout=config["head_dropout"],
    )


def predict_proba(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    amp_enabled = device.type == "cuda"
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


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
        raise RuntimeError("CUDA is not available. Use iscc-gpu or pass --allow-cpu.")

    bundle = torch_load(args.model_path)
    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    label_names = list(bundle["label_names"])
    feature_cols = list(bundle["feature_columns"])
    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        raise ValueError(f"test data is missing feature columns: {missing[:5]}")

    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    probs = np.zeros((len(test_df), len(label_names)), dtype=np.float32)
    for record in bundle["fold_models"]:
        model = build_from_bundle(bundle, len(feature_cols), len(label_names)).to(device)
        model.load_state_dict(record["state_dict"])
        x_test = apply_robust_stats(
            x_test_raw,
            np.asarray(record["center"], dtype=np.float32).reshape(1, -1),
            np.asarray(record["scale"], dtype=np.float32).reshape(1, -1),
        )
        probs += predict_proba(model, x_test, device, args.batch_size, args.num_workers) / len(bundle["fold_models"])

    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})
    validate_prediction_frame(
        submission,
        test_df,
        sorted(train_df[LABEL_COL].astype(str).unique().tolist()),
        list(sample_df.columns),
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_with_retry(submission, args.output_path)
    print(f"Wrote reproduced {VERSION} submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
