from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

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


VERSION = "v1.1"


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * torch.nn.functional.gelu(gate)


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_token,
            num_heads=n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(residual_dropout)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn * 2),
            GEGLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
        )
        self.ffn_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.attn_norm(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(h)
        h = self.ffn(self.ffn_norm(x))
        return x + self.ffn_dropout(h)


class FTTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_token: int,
        n_blocks: int,
        n_heads: int,
        d_ffn: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    d_token=d_token,
                    n_heads=n_heads,
                    d_ffn=d_ffn,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    ffn_dropout=ffn_dropout,
                )
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_token * 2),
            nn.Linear(d_token * 2, d_token * 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_token * 2, n_classes),
        )
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.tokenizer(x)
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.blocks(tokens)
        cls_out = tokens[:, 0]
        mean_out = tokens[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls_out, mean_out], dim=1))


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_output_paths(root: Path, smoke: bool) -> dict[str, Path]:
    model_dir = root / "\u6a21\u578b"
    result_dir = root / "\u63d0\u4ea4\u7ed3\u679c"
    prefix = "smoke_" if smoke else ""
    return {
        "model_path": model_dir / f"{prefix}gpu_fttransformer_model_bundle_{VERSION}.pt",
        "oof_probs_path": model_dir / f"{prefix}gpu_fttransformer_oof_probs_{VERSION}.npy",
        "test_probs_path": model_dir / f"{prefix}gpu_fttransformer_test_probs_{VERSION}.npy",
        "report_path": model_dir / f"{prefix}gpu_fttransformer_validation_report_{VERSION}.json",
        "submission_path": result_dir / f"{prefix}submission_gpu_fttransformer_{VERSION}.csv",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=f"Train {VERSION} CUDA FT-Transformer for numeric tabular classification."
    )
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--oof-probs-path", type=Path, default=None)
    parser.add_argument("--test-probs-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--submission-path", type=Path, default=None)
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny test and write smoke_* outputs.")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(str(DEFAULT_SEED)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.03)
    parser.add_argument("--d-token", type=int, default=64)
    parser.add_argument("--n-blocks", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--d-ffn", type=int, default=192)
    parser.add_argument("--attention-dropout", type=float, default=0.10)
    parser.add_argument("--residual-dropout", type=float, default=0.05)
    parser.add_argument("--ffn-dropout", type=float, default=0.10)
    parser.add_argument("--head-dropout", type=float, default=0.15)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.dev_limit = args.dev_limit or 720
        args.folds = min(args.folds, 2)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 256)

    paths = default_output_paths(root, args.smoke)
    for key, value in paths.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def build_model(args: argparse.Namespace, n_features: int, n_classes: int) -> FTTransformerClassifier:
    return FTTransformerClassifier(
        n_features=n_features,
        n_classes=n_classes,
        d_token=args.d_token,
        n_blocks=args.n_blocks,
        n_heads=args.n_heads,
        d_ffn=args.d_ffn,
        attention_dropout=args.attention_dropout,
        residual_dropout=args.residual_dropout,
        ffn_dropout=args.ffn_dropout,
        head_dropout=args.head_dropout,
    )


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit < len(np.unique(labels)) * 2:
        raise ValueError("dev-limit is too small for stratified smoke validation")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def fit_robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0, keepdims=True).astype(np.float32)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True).astype(np.float32)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    fallback = np.nanstd(x, axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, fallback, scale).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return center, scale


def apply_robust_stats(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (x - center) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(x, -8.0, 8.0).astype(np.float32)


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
    amp_enabled = device.type == "cuda"
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def train_one_fold(
    args: argparse.Namespace,
    seed: int,
    fold_idx: int,
    total_folds: int,
    x_train_raw: np.ndarray,
    y: np.ndarray,
    x_test_raw: np.ndarray,
    tr_idx: np.ndarray,
    va_idx: np.ndarray,
    device: torch.device,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    set_seed(seed + fold_idx * 1000)
    center, scale = fit_robust_stats(x_train_raw[tr_idx])
    x_tr = apply_robust_stats(x_train_raw[tr_idx], center, scale)
    x_va = apply_robust_stats(x_train_raw[va_idx], center, scale)
    x_test = apply_robust_stats(x_test_raw, center, scale)
    y_tr = y[tr_idx].astype(np.int64)
    y_va = y[va_idx].astype(np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(args, x_train_raw.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    best_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0

    print(f"[{VERSION}] seed={seed} fold={fold_idx}/{total_folds} train_rows={len(tr_idx)} val_rows={len(va_idx)}")
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"{VERSION} fold {fold_idx}",
        unit="epoch",
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in epoch_bar:
        model.train()
        running_loss = 0.0
        seen = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_probs = predict_proba(model, x_va, device, args.batch_size, args.num_workers)
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
    va_probs = predict_proba(model, x_va, device, args.batch_size, args.num_workers)
    test_probs = predict_proba(model, x_test, device, args.batch_size, args.num_workers)
    record = {
        "version": VERSION,
        "seed": seed,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_train_loss": best_loss,
        "center": center.squeeze(0).astype(np.float32),
        "scale": scale.squeeze(0).astype(np.float32),
        "state_dict": best_state,
    }
    print(f"[{VERSION}] fold={fold_idx} done best_macro_f1={best_f1:.6f} best_epoch={best_epoch}")
    return va_probs, test_probs, record


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError(
            "CUDA is not available. Use the iscc-gpu environment or pass --allow-cpu for a smoke check."
        )

    for path in [args.model_path, args.oof_probs_path, args.test_probs_path, args.report_path, args.submission_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    full_train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    feature_cols = ensure_feature_columns(full_train_df, test_df)
    label_encoder, y_full = encode_labels(full_train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()

    selected_idx = select_dev_indices(y_full, args.dev_limit, DEFAULT_SEED)
    train_df = full_train_df.iloc[selected_idx].reset_index(drop=True)
    y = y_full[selected_idx]

    x_train_raw = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    n_classes = len(label_names)

    append_action_log(
        args.action_log,
        f"{VERSION} FT-Transformer training started: smoke={args.smoke}, device={device}, "
        f"train_rows={len(train_df)}/{len(full_train_df)}, test_rows={len(test_df)}, "
        f"features={len(feature_cols)}, labels={n_classes}, seeds={args.seeds}, folds={args.folds}.",
    )

    oof_probs = np.zeros((len(train_df), n_classes), dtype=np.float32)
    test_probs = np.zeros((len(test_df), n_classes), dtype=np.float32)
    fold_records: list[dict] = []
    total_runs = len(args.seeds) * args.folds

    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_raw, y), start=1):
            va_probs, fold_test_probs, record = train_one_fold(
                args=args,
                seed=seed,
                fold_idx=fold_idx,
                total_folds=args.folds,
                x_train_raw=x_train_raw,
                y=y,
                x_test_raw=x_test_raw,
                tr_idx=tr_idx,
                va_idx=va_idx,
                device=device,
                n_classes=n_classes,
            )
            oof_probs[va_idx] += va_probs / len(args.seeds)
            test_probs += fold_test_probs / total_runs
            fold_records.append(record)
            append_action_log(
                args.action_log,
                f"{VERSION} FT-Transformer seed={seed} fold={fold_idx}/{args.folds} finished: "
                f"best_macro_f1={record['best_macro_f1']:.6f}, best_epoch={record['best_epoch']}.",
            )

    oof_pred = np.argmax(oof_probs, axis=1)
    test_pred = np.argmax(test_probs, axis=1)
    test_labels = label_encoder.inverse_transform(test_pred)
    report = classification_summary(y, oof_pred, label_names)
    report.update(
        {
            "version": VERSION,
            "smoke": bool(args.smoke),
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
            "architecture": "FTTransformerClassifier",
            "d_token": args.d_token,
            "n_blocks": args.n_blocks,
            "n_heads": args.n_heads,
            "d_ffn": args.d_ffn,
            "attention_dropout": args.attention_dropout,
            "residual_dropout": args.residual_dropout,
            "ffn_dropout": args.ffn_dropout,
            "head_dropout": args.head_dropout,
            "train_rows": int(len(train_df)),
            "full_train_rows": int(len(full_train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": int(len(feature_cols)),
            "label_count": int(n_classes),
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
    write_csv_with_retry(submission, args.submission_path)
    np.save(args.oof_probs_path, oof_probs)
    np.save(args.test_probs_path, test_probs)

    torch.save(
        {
            "schema_version": 1,
            "version": VERSION,
            "model_name": "FTTransformerClassifier",
            "feature_columns": feature_cols,
            "label_names": label_names,
            "config": {
                "d_token": args.d_token,
                "n_blocks": args.n_blocks,
                "n_heads": args.n_heads,
                "d_ffn": args.d_ffn,
                "attention_dropout": args.attention_dropout,
                "residual_dropout": args.residual_dropout,
                "ffn_dropout": args.ffn_dropout,
                "head_dropout": args.head_dropout,
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
        f"{VERSION} FT-Transformer training completed: smoke={args.smoke}, "
        f"macro_f1={report['macro_f1']:.6f}, accuracy={report['accuracy']:.6f}, "
        f"submission={args.submission_path}.",
    )
    print(json.dumps({"version": VERSION, "macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}, indent=2))
    print(f"Wrote submission: {args.submission_path}")
    print(f"Wrote model bundle: {args.model_path}")
    print(f"Wrote test probabilities: {args.test_probs_path}")
    print(f"Wrote validation report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
