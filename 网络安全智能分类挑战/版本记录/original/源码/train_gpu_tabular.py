from __future__ import annotations

import argparse
import json
import random
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from common import (
    DEFAULT_SEED,
    ID_COL,
    LABEL_COL,
    append_action_log,
    classification_summary,
    encode_labels,
    ensure_feature_columns,
    make_feature_frame,
    read_table,
    validate_prediction_frame,
)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class NumericResNetMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int,
        block_hidden_dim: int,
        num_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, block_hidden_dim, dropout) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_fold(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def predict_proba(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch_x)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            chunks.append(probs.astype(np.float32))
    return np.concatenate(chunks, axis=0)


def build_model(args: argparse.Namespace, input_dim: int, num_classes: int) -> NumericResNetMLP:
    return NumericResNetMLP(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        block_hidden_dim=args.block_hidden_dim,
        num_blocks=args.num_blocks,
        dropout=args.dropout,
    )


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train a CUDA PyTorch tabular neural network.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_model_bundle.pt")
    parser.add_argument("--oof-probs-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_oof_probs.npy")
    parser.add_argument("--test-probs-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_test_probs.npy")
    parser.add_argument("--report-path", type=Path, default=default_root / "\u6a21\u578b" / "gpu_validation_report.json")
    parser.add_argument("--submission-path", type=Path, default=default_root / "\u63d0\u4ea4\u7ed3\u679c" / "submission_gpu.csv")
    parser.add_argument("--action-log", type=Path, default=default_root / "ACTION_LOG.md")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(f"{DEFAULT_SEED},{DEFAULT_SEED + 1}"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.04)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--block-hidden-dim", type=int, default=512)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def train_one_fold(
    args: argparse.Namespace,
    seed: int,
    fold_idx: int,
    x_train_raw: np.ndarray,
    y: np.ndarray,
    x_test_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed + fold_idx * 1000)

    mean = x_train_raw[tr_idx].mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train_raw[tr_idx].std(axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    x_tr = normalize_fold(x_train_raw[tr_idx], mean, std)
    x_va = normalize_fold(x_train_raw[va_idx], mean, std)
    x_test = normalize_fold(x_test_raw, mean, std)
    y_tr = y[tr_idx].astype(np.int64)
    y_va = y[va_idx].astype(np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args, x_train_raw.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    best_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"seed {seed} fold {fold_idx}",
        unit="epoch",
        dynamic_ncols=True,
        leave=False,
    )
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_probs = predict_proba(model, x_va, device, args.batch_size, amp_enabled)
        val_pred = np.argmax(val_probs, axis=1)
        val_f1 = float(f1_score(y_va, val_pred, average="macro"))

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_loss = train_loss
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_macro_f1=f"{val_f1:.5f}",
            best_f1=f"{best_f1:.5f}",
            patience=f"{bad_epochs}/{args.patience}",
        )
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    va_probs = predict_proba(model, x_va, device, args.batch_size, amp_enabled)
    test_probs = predict_proba(model, x_test, device, args.batch_size, amp_enabled)
    record = {
        "seed": seed,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_train_loss": best_loss,
        "mean": mean.squeeze(0),
        "std": std.squeeze(0),
        "state_dict": best_state,
    }
    return va_probs, test_probs, record


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is not available in this Python environment. Install a CUDA-enabled PyTorch build first, "
            "then verify with: python -c \"import torch; print(torch.cuda.is_available())\""
        )

    for path in [args.model_path, args.oof_probs_path, args.test_probs_path, args.report_path, args.submission_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    feature_cols = ensure_feature_columns(train_df, test_df)

    x_train_raw = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    label_encoder, y = encode_labels(train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()
    num_classes = len(label_names)

    append_action_log(
        args.action_log,
        f"GPU training started: device={device}, train_rows={len(train_df)}, test_rows={len(test_df)}, "
        f"features={len(feature_cols)}, labels={num_classes}, seeds={args.seeds}, folds={args.folds}.",
    )

    oof_probs = np.zeros((len(train_df), num_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), num_classes), dtype=np.float32)
    fold_records: list[dict] = []
    total_runs = len(args.seeds) * args.folds
    run_bar = tqdm(total=total_runs, desc="GPU CV runs", unit="run", dynamic_ncols=True)

    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_raw, y), start=1):
            va_probs, fold_test_probs, record = train_one_fold(
                args=args,
                seed=seed,
                fold_idx=fold_idx,
                x_train_raw=x_train_raw,
                y=y,
                x_test_raw=x_test_raw,
                tr_idx=tr_idx,
                va_idx=va_idx,
                device=device,
                num_classes=num_classes,
            )
            oof_probs[va_idx] += va_probs / len(args.seeds)
            test_probs += fold_test_probs / total_runs
            fold_records.append(record)
            run_bar.update(1)
            run_bar.set_postfix(best_f1=f"{record['best_macro_f1']:.5f}", seed=seed, fold=fold_idx)
            append_action_log(
                args.action_log,
                f"GPU seed={seed} fold={fold_idx}/{args.folds} finished: "
                f"best_macro_f1={record['best_macro_f1']:.6f}, best_epoch={record['best_epoch']}.",
            )
    run_bar.close()

    oof_pred = np.argmax(oof_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)
    test_labels = label_encoder.inverse_transform(test_pred)
    report = classification_summary(y, oof_pred, label_names)
    report.update(
        {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "seeds": args.seeds,
            "folds": args.folds,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "hidden_dim": args.hidden_dim,
            "block_hidden_dim": args.block_hidden_dim,
            "num_blocks": args.num_blocks,
            "dropout": args.dropout,
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_cols)),
            "label_count": int(num_classes),
            "label_names": label_names,
            "fold_records": [
                {
                    "seed": r["seed"],
                    "fold": r["fold"],
                    "best_epoch": r["best_epoch"],
                    "best_macro_f1": r["best_macro_f1"],
                    "best_train_loss": r["best_train_loss"],
                }
                for r in fold_records
            ],
        }
    )

    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: test_labels})
    validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))
    submission.to_csv(args.submission_path, index=False, encoding="utf-8")
    np.save(args.oof_probs_path, oof_probs)
    np.save(args.test_probs_path, test_probs)

    torch.save(
        {
            "schema_version": 1,
            "model_name": "NumericResNetMLP",
            "feature_columns": feature_cols,
            "label_names": label_names,
            "config": {
                "hidden_dim": args.hidden_dim,
                "block_hidden_dim": args.block_hidden_dim,
                "num_blocks": args.num_blocks,
                "dropout": args.dropout,
            },
            "fold_models": fold_records,
            "validation": report,
        },
        args.model_path,
    )
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"GPU training completed: macro_f1={report['macro_f1']:.6f}, accuracy={report['accuracy']:.6f}, "
        f"submission={args.submission_path}.",
    )
    print(json.dumps({"macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}, indent=2))
    print(f"Wrote GPU submission: {args.submission_path}")
    print(f"Wrote GPU model bundle: {args.model_path}")
    print(f"Wrote GPU test probabilities: {args.test_probs_path}")
    print(f"Wrote GPU validation report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
