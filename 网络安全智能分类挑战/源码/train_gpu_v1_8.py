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

from common import LABEL_COL, append_action_log, classification_summary, encode_labels, ensure_feature_columns, make_feature_frame, read_table
from gpu_v1_8_core import (
    VERSION, DEFAULT_SEEDS, SWA,
    apply_robust_stats, build_model, fit_robust_stats,
    log_key_metrics, parse_seeds, predict_probs, set_seed,
)


def default_output_paths(root: Path, smoke: bool) -> dict[str, Path]:
    model_dir = root / "模型"
    prefix = "smoke_" if smoke else ""
    return {
        "bundle_path": model_dir / f"{prefix}gpu_ft_bundle_{VERSION}.pt",
        "oof_probs_path": model_dir / f"{prefix}gpu_oof_probs_{VERSION}.npy",
        "test_probs_path": model_dir / f"{prefix}gpu_test_probs_{VERSION}.npy",
        "report_path": model_dir / f"{prefix}gpu_validation_report_{VERSION}.json",
    }


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=f"Train {VERSION} FT-Transformer with SWA.")
    p.add_argument("--data-dir", type=Path, default=root)
    p.add_argument("--train-file", type=str, default="train_data.csv")
    p.add_argument("--test-file", type=str, default="test_data.csv")
    p.add_argument("--bundle-path", type=Path, default=None)
    p.add_argument("--oof-probs-path", type=Path, default=None)
    p.add_argument("--test-probs-path", type=Path, default=None)
    p.add_argument("--report-path", type=Path, default=None)
    p.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dev-limit", type=int, default=None)
    p.add_argument("--seeds", type=parse_seeds, default=parse_seeds(",".join(str(s) for s in DEFAULT_SEEDS)))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=90)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=14)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=2e-4)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--d-token", type=int, default=64)
    p.add_argument("--n-blocks", type=int, default=4)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--d-ffn", type=int, default=192)
    p.add_argument("--attention-dropout", type=float, default=0.10)
    p.add_argument("--residual-dropout", type=float, default=0.05)
    p.add_argument("--ffn-dropout", type=float, default=0.10)
    p.add_argument("--head-dropout", type=float, default=0.15)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--swa-start-frac", type=float, default=0.6,
                   help="Fraction of epochs after which SWA begins (default: 0.6)")
    args = p.parse_args()

    if args.smoke:
        args.dev_limit = args.dev_limit or 720
        args.seeds = args.seeds[:1]
        args.folds = min(args.folds, 2)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 256)

    paths = default_output_paths(root, args.smoke)
    for key, value in paths.items():
        if getattr(args, key.replace("-", "_")) is None:
            setattr(args, key.replace("-", "_"), value)
    return args


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def train_fold(
    args: argparse.Namespace, seed: int, fold_idx: int, total_folds: int,
    x_train_raw: np.ndarray, y: np.ndarray, x_test_raw: np.ndarray,
    tr_idx: np.ndarray, va_idx: np.ndarray,
    device: torch.device, n_classes: int,
    swa_start_epoch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    set_seed(seed + fold_idx * 1000)
    center, scale = fit_robust_stats(x_train_raw[tr_idx])
    x_tr = apply_robust_stats(x_train_raw[tr_idx], center, scale)
    x_va = apply_robust_stats(x_train_raw[va_idx], center, scale)
    x_test = apply_robust_stats(x_test_raw, center, scale)
    y_tr = y[tr_idx].astype(np.int64)
    y_va = y[va_idx].astype(np.int64)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", drop_last=False,
    )

    config = {
        "d_token": args.d_token, "n_blocks": args.n_blocks, "n_heads": args.n_heads,
        "d_ffn": args.d_ffn, "attention_dropout": args.attention_dropout,
        "residual_dropout": args.residual_dropout, "ffn_dropout": args.ffn_dropout,
        "head_dropout": args.head_dropout,
    }
    model = build_model(x_train_raw.shape[1], n_classes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    swa = SWA(model, swa_start_epoch)

    # Best individual model (macro F1 based)
    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    best_macro_f1 = -1.0
    best_accuracy = -1.0
    bad_epochs = 0

    epoch_bar = tqdm(range(1, args.epochs + 1),
                     desc=f"FT seed={seed} fold={fold_idx}/{total_folds}",
                     unit="epoch", dynamic_ncols=True, leave=True)
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

        # Evaluate
        val_probs = predict_probs(model, x_va, device, args.batch_size, args.num_workers)
        val_pred = np.argmax(val_probs, axis=1)
        val_macro_f1 = float(f1_score(y_va, val_pred, average="macro"))
        val_accuracy = float(accuracy_score(y_va, val_pred))
        train_loss = running_loss / max(seen, 1)

        # SWA step (after evaluation)
        swa.step(epoch)

        # Best selection via macro F1
        improved = val_macro_f1 > best_macro_f1 + 1e-12 or (
            abs(val_macro_f1 - best_macro_f1) <= 1e-12 and val_accuracy > best_accuracy + 1e-12
        )
        if improved:
            best_macro_f1 = val_macro_f1
            best_accuracy = val_accuracy
            best_epoch = epoch
            best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            bad_epochs = 0
        else:
            bad_epochs += 1

        swa_tag = f" swa={swa.n_updates}" if epoch >= swa_start_epoch else ""
        epoch_bar.set_postfix(
            train_loss=f"{train_loss:.5f}", val_mf1=f"{val_macro_f1:.5f}",
            best_mf1=f"{best_macro_f1:.5f}", patience=f"{bad_epochs}/{args.patience}{swa_tag}",
        )
        if bad_epochs >= args.patience:
            break

    # ── Best individual model predictions ──
    model.load_state_dict(best_state)
    best_va = predict_probs(model, x_va, device, args.batch_size, args.num_workers)
    best_te = predict_probs(model, x_test, device, args.batch_size, args.num_workers)

    # ── SWA model predictions ──
    swa_va = best_va
    swa_te = best_te
    if swa.n_updates > 0:
        swa_state = swa.apply()
        model.load_state_dict(swa_state)
        swa_va = predict_probs(model, x_va, device, args.batch_size, args.num_workers)
        swa_te = predict_probs(model, x_test, device, args.batch_size, args.num_workers)
        swa_pred = np.argmax(swa_va, axis=1)
        swa_mf1 = float(f1_score(y_va, swa_pred, average="macro"))
        print(f"[{VERSION}] fold={fold_idx} best_mf1={best_macro_f1:.6f} ep={best_epoch} swa_mf1={swa_mf1:.6f}(+{swa.n_updates})")
    else:
        print(f"[{VERSION}] fold={fold_idx} best_mf1={best_macro_f1:.6f} ep={best_epoch} swa=inactive")

    record = {
        "version": VERSION, "seed": seed, "fold": fold_idx,
        "best_epoch": best_epoch, "best_macro_f1": best_macro_f1,
        "best_accuracy": best_accuracy, "best_train_loss": float(train_loss),
        "swa_updates": swa.n_updates,
        "center": center.squeeze(0).astype(np.float32),
        "scale": scale.squeeze(0).astype(np.float32),
        "best_state": best_state,
        "swa_state": swa.apply() if swa.n_updates > 0 else best_state,
    }
    return best_va, swa_va, best_te, swa_te, record


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

    for p in [args.bundle_path, args.oof_probs_path, args.test_probs_path, args.report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    swa_start = max(1, int(args.epochs * args.swa_start_frac))

    data_dir = args.data_dir.resolve()
    full_train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    feature_cols = ensure_feature_columns(full_train_df, test_df)
    label_encoder, y_full = encode_labels(full_train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()
    n_classes = len(label_names)

    selected_idx = select_dev_indices(y_full, args.dev_limit, DEFAULT_SEEDS[0])
    train_df = full_train_df.iloc[selected_idx].reset_index(drop=True)
    y = y_full[selected_idx]

    x_train_raw = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)

    append_action_log(args.action_log,
        f"{VERSION} training: device={device}, rows={len(train_df)}/{len(full_train_df)}, "
        f"test={len(test_df)}, features={len(feature_cols)}, labels={n_classes}, "
        f"seeds={args.seeds}, folds={args.folds}, epochs={args.epochs}, "
        f"arch=(d={args.d_token},blk={args.n_blocks},h={args.n_heads},ffn={args.d_ffn}), "
        f"swa_start={swa_start}({args.swa_start_frac}), "
        f"drop=({args.attention_dropout},{args.residual_dropout},{args.ffn_dropout},{args.head_dropout}), "
        f"ls={args.label_smoothing}, mf1_early_stop.")

    total_runs = len(args.seeds) * args.folds
    best_oof = np.zeros((len(train_df), n_classes), dtype=np.float32)
    swa_oof = np.zeros_like(best_oof)
    best_test = np.zeros((len(test_df), n_classes), dtype=np.float32)
    swa_test = np.zeros_like(best_test)
    fold_records: list[dict] = []

    t0 = time.perf_counter()
    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_raw, y), start=1):
            b_va, s_va, b_te, s_te, record = train_fold(
                args=args, seed=seed, fold_idx=fold_idx, total_folds=args.folds,
                x_train_raw=x_train_raw, y=y, x_test_raw=x_test_raw,
                tr_idx=tr_idx, va_idx=va_idx, device=device, n_classes=n_classes,
                swa_start_epoch=swa_start,
            )
            best_oof[va_idx] += b_va / len(args.seeds)
            swa_oof[va_idx] += s_va / len(args.seeds)
            best_test += b_te / total_runs
            swa_test += s_te / total_runs
            fold_records.append(record)
            append_action_log(args.action_log,
                f"{VERSION} FT seed={seed} fold={fold_idx}/{args.folds}: "
                f"best_mf1={record['best_macro_f1']:.6f} ep={record['best_epoch']} swa_updates={record['swa_updates']}")
    elapsed = time.perf_counter() - t0
    print(f"Training done in {elapsed/60:.1f} min")

    # ── Compare best vs SWA ──
    best_pred = np.argmax(best_oof, axis=1)
    swa_pred = np.argmax(swa_oof, axis=1)
    best_mf1 = float(f1_score(y, best_pred, average="macro"))
    swa_mf1 = float(f1_score(y, swa_pred, average="macro"))
    best_acc = float(accuracy_score(y, best_pred))
    swa_acc = float(accuracy_score(y, swa_pred))

    use_swa = swa_mf1 >= best_mf1
    final_oof = swa_oof if use_swa else best_oof
    final_test = swa_test if use_swa else best_test
    final_label = "swa" if use_swa else "best_individual"

    report = classification_summary(y, np.argmax(final_oof, axis=1), label_names)
    weak_f1 = min(report["per_class_f1"].values())
    weak_class = min(report["per_class_f1"], key=report["per_class_f1"].get)
    report.update({
        "version": VERSION, "smoke": bool(args.smoke),
        "selected": final_label, "best_macro_f1": best_mf1, "swa_macro_f1": swa_mf1,
        "best_accuracy": best_acc, "swa_accuracy": swa_acc,
        "python_version": sys.version, "torch_version": torch.__version__,
        "device": str(device), "seeds": args.seeds, "folds": args.folds,
        "epochs": args.epochs, "swa_start_epoch": swa_start,
        "batch_size": args.batch_size, "patience": args.patience,
        "lr": args.lr, "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "d_token": args.d_token, "n_blocks": args.n_blocks, "n_heads": args.n_heads, "d_ffn": args.d_ffn,
        "attention_dropout": args.attention_dropout, "residual_dropout": args.residual_dropout,
        "ffn_dropout": args.ffn_dropout, "head_dropout": args.head_dropout,
        "total_runs": total_runs, "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "label_names": label_names, "label_count": n_classes,
    })

    np.save(args.oof_probs_path, final_oof)
    np.save(args.test_probs_path, final_test)

    torch.save({
        "schema_version": 4, "version": VERSION, "selected": final_label,
        "model_name": "FTTransformerClassifier_SWA",
        "feature_columns": feature_cols, "label_names": label_names,
        "best_vs_swa": {"best_mf1": best_mf1, "swa_mf1": swa_mf1},
        "config": {
            "d_token": args.d_token, "n_blocks": args.n_blocks,
            "n_heads": args.n_heads, "d_ffn": args.d_ffn,
            "attention_dropout": args.attention_dropout, "residual_dropout": args.residual_dropout,
            "ffn_dropout": args.ffn_dropout, "head_dropout": args.head_dropout,
        },
        "fold_models": fold_records, "validation": report,
    }, args.bundle_path)
    write_json(args.report_path, report)

    summary = (
        f"{VERSION} done ({elapsed/60:.1f}min): "
        f"best mf1={best_mf1:.4f} acc={best_acc:.4f} | "
        f"swa mf1={swa_mf1:.4f} acc={swa_acc:.4f} | "
        f"selected={final_label} mf1={report['macro_f1']:.4f} "
        f"weak=({weak_class}:{weak_f1:.4f})"
    )
    print(summary)
    append_action_log(args.action_log, summary)

    log_key_metrics(root=data_dir, metrics={
        "version": VERSION, "stage": "smoke" if args.smoke else "full",
        "model": f"ft_swa_{final_label}", "n_features": len(feature_cols),
        "seeds": len(args.seeds), "folds": args.folds,
        "local_acc": f"{report['accuracy']:.4f}",
        "local_macro_f1": f"{report['macro_f1']:.4f}",
        "weak_f1": f"{weak_class}:{weak_f1:.4f}",
        "platform_score": "-",
        "notes": f"best:{best_mf1:.4f} swa:{swa_mf1:.4f}->{final_label}",
    })

    print(f"Bundle: {args.bundle_path}")
    print(f"OOF: {args.oof_probs_path}")
    print(f"Report: {args.report_path}")
    print("Run predict_gpu_v1_8.py next.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
