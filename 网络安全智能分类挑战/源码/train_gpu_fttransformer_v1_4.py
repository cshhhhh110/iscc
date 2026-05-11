from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from common import LABEL_COL, ID_COL, append_action_log, classification_summary, encode_labels, ensure_feature_columns, make_feature_frame, read_table
from gpu_fttransformer_v1_4_core import (
    DEFAULT_SEEDS,
    VERSION,
    apply_robust_stats,
    build_model,
    fit_robust_stats,
    parse_seeds,
    predict_probs,
    set_seed,
)


def default_output_paths(root: Path, smoke: bool) -> dict[str, Path]:
    model_dir = root / "\u6a21\u578b"
    prefix = "smoke_" if smoke else ""
    return {
        "model_path": model_dir / f"{prefix}gpu_fttransformer_model_bundle_{VERSION}.pt",
        "oof_probs_path": model_dir / f"{prefix}gpu_fttransformer_oof_probs_{VERSION}.npy",
        "test_probs_path": model_dir / f"{prefix}gpu_fttransformer_test_probs_{VERSION}.npy",
        "report_path": model_dir / f"{prefix}gpu_fttransformer_validation_report_{VERSION}.json",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=f"Train {VERSION} FT-Transformer for numeric tabular classification."
    )
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--oof-probs-path", type=Path, default=None)
    parser.add_argument("--test-probs-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny test and write smoke_* outputs.")
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(",".join(str(seed) for seed in DEFAULT_SEEDS)))
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
        args.seeds = args.seeds[:1]
        args.folds = min(args.folds, 2)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 256)

    paths = default_output_paths(root, args.smoke)
    for key, value in paths.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit < len(np.unique(labels)) * 2:
        raise ValueError("dev-limit is too small for stratified smoke validation")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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

    model = build_model(
        n_features=x_train_raw.shape[1],
        n_classes=n_classes,
        config={
            "d_token": args.d_token,
            "n_blocks": args.n_blocks,
            "n_heads": args.n_heads,
            "d_ffn": args.d_ffn,
            "attention_dropout": args.attention_dropout,
            "residual_dropout": args.residual_dropout,
            "ffn_dropout": args.ffn_dropout,
            "head_dropout": args.head_dropout,
        },
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0

    print(f"[{VERSION}] seed={seed} fold={fold_idx}/{total_folds} train_rows={len(tr_idx)} val_rows={len(va_idx)}")
    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"{VERSION} seed {seed} fold {fold_idx}/{total_folds}",
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
        val_probs = predict_probs(model, x_va, device, args.batch_size, args.num_workers)
        val_pred = np.argmax(val_probs, axis=1)
        val_accuracy = float(accuracy_score(y_va, val_pred))
        val_macro_f1 = float(f1_score(y_va, val_pred, average="macro"))

        improved = val_accuracy > best_accuracy + 1e-12 or (
            abs(val_accuracy - best_accuracy) <= 1e-12 and val_macro_f1 > best_macro_f1 + 1e-12
        )
        if improved:
            best_accuracy = val_accuracy
            best_macro_f1 = val_macro_f1
            best_loss = train_loss
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1

        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.5f}",
            val_acc=f"{val_accuracy:.5f}",
            val_macro_f1=f"{val_macro_f1:.5f}",
            best_acc=f"{best_accuracy:.5f}",
            patience=f"{bad_epochs}/{args.patience}",
        )
        if bad_epochs >= args.patience:
            break

    model.load_state_dict(best_state)
    va_probs = predict_probs(model, x_va, device, args.batch_size, args.num_workers)
    test_probs = predict_probs(model, x_test, device, args.batch_size, args.num_workers)
    record = {
        "version": VERSION,
        "seed": seed,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "best_train_loss": best_loss,
        "center": center.squeeze(0).astype(np.float32),
        "scale": scale.squeeze(0).astype(np.float32),
        "state_dict": best_state,
    }
    print(
        f"[{VERSION}] fold={fold_idx} done best_accuracy={best_accuracy:.6f} "
        f"best_macro_f1={best_macro_f1:.6f} best_epoch={best_epoch}"
    )
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

    for path in [args.model_path, args.oof_probs_path, args.test_probs_path, args.report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    full_train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    feature_cols = ensure_feature_columns(full_train_df, test_df)
    label_encoder, y_full = encode_labels(full_train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()

    selected_idx = select_dev_indices(y_full, args.dev_limit, DEFAULT_SEEDS[0])
    train_df = full_train_df.iloc[selected_idx].reset_index(drop=True)
    y = y_full[selected_idx]

    x_train_raw = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    n_classes = len(label_names)

    append_action_log(
        args.action_log,
        f"{VERSION} FT-Transformer training started: smoke={args.smoke}, device={device}, "
        f"train_rows={len(train_df)}/{len(full_train_df)}, test_rows={len(test_df)}, "
        f"features={len(feature_cols)}, labels={n_classes}, seeds={args.seeds}, folds={args.folds}, "
        f"selection_metric=accuracy.",
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
                f"best_accuracy={record['best_accuracy']:.6f}, best_macro_f1={record['best_macro_f1']:.6f}, "
                f"best_epoch={record['best_epoch']}.",
            )

    oof_pred = np.argmax(oof_probs, axis=1)
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
            "selection_metric": "accuracy",
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
                    "best_accuracy": r["best_accuracy"],
                    "best_macro_f1": r["best_macro_f1"],
                    "best_train_loss": r["best_train_loss"],
                }
                for r in fold_records
            ],
        }
    )

    np.save(args.oof_probs_path, oof_probs)
    np.save(args.test_probs_path, test_probs)

    torch.save(
        {
            "schema_version": 1,
            "version": VERSION,
            "model_name": "FTTransformerClassifier",
            "feature_columns": feature_cols,
            "label_names": label_names,
            "selection_metric": "accuracy",
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
    write_json(args.report_path, report)

    append_action_log(
        args.action_log,
        f"{VERSION} FT-Transformer training completed: smoke={args.smoke}, "
        f"macro_f1={report['macro_f1']:.6f}, accuracy={report['accuracy']:.6f}, "
        f"bundle={args.model_path}.",
    )
    print(json.dumps({"version": VERSION, "macro_f1": report["macro_f1"], "accuracy": report["accuracy"]}, indent=2))
    print(f"Wrote model bundle: {args.model_path}")
    print(f"Wrote OOF probabilities: {args.oof_probs_path}")
    print(f"Wrote test probabilities: {args.test_probs_path}")
    print(f"Wrote validation report: {args.report_path}")
    print("Train does not write a submission. Run predict_gpu_fttransformer_v1_4.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

