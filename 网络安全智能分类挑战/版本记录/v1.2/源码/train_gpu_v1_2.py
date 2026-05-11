from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from common import LABEL_COL, append_action_log, ensure_feature_columns, make_feature_frame, read_table
from gpu_v1_2_core import (
    DEFAULT_SEEDS,
    FT_FAMILY,
    MLP_FAMILY,
    VERSION,
    apply_standardizer,
    build_loss,
    build_model,
    class_balanced_weights,
    blend_probs,
    parse_seeds,
    predict_probs,
    search_blend_weight,
    set_seed,
    summarize_predictions,
    torch_save_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Train the {VERSION} pure-GPU dual-model baseline.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--bundle-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_model_bundle_{VERSION}.pt")
    parser.add_argument("--ft-oof-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_ft_oof_probs_{VERSION}.npy")
    parser.add_argument("--ft-test-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_ft_test_probs_{VERSION}.npy")
    parser.add_argument("--mlp-oof-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_mlp_oof_probs_{VERSION}.npy")
    parser.add_argument("--mlp-test-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_mlp_test_probs_{VERSION}.npy")
    parser.add_argument("--oof-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_oof_probs_{VERSION}.npy")
    parser.add_argument("--test-path", type=Path, default=root / "\u6a21\u578b" / f"gpu_test_probs_{VERSION}.npy")
    parser.add_argument(
        "--report-path",
        type=Path,
        default=root / "\u6a21\u578b" / f"gpu_validation_report_{VERSION}.json",
    )
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(",".join(str(x) for x in DEFAULT_SEEDS)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--loss-type", choices=["focal", "weighted_ce"], default="focal")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny test and write smoke_* outputs.")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.dev_limit = args.dev_limit or 720
        args.folds = min(args.folds, 2)
        args.epochs = min(args.epochs, 2)
        args.patience = min(args.patience, 1)
        args.batch_size = min(args.batch_size, 256)
        args.seeds = args.seeds[:1]

    prefix = "smoke_" if args.smoke else ""
    model_dir = root / "\u6a21\u578b"
    for attr, suffix in [
        ("bundle_path", f"{prefix}gpu_model_bundle_{VERSION}.pt"),
        ("ft_oof_path", f"{prefix}gpu_ft_oof_probs_{VERSION}.npy"),
        ("ft_test_path", f"{prefix}gpu_ft_test_probs_{VERSION}.npy"),
        ("mlp_oof_path", f"{prefix}gpu_mlp_oof_probs_{VERSION}.npy"),
        ("mlp_test_path", f"{prefix}gpu_mlp_test_probs_{VERSION}.npy"),
        ("oof_path", f"{prefix}gpu_oof_probs_{VERSION}.npy"),
        ("test_path", f"{prefix}gpu_test_probs_{VERSION}.npy"),
        ("report_path", f"{prefix}gpu_validation_report_{VERSION}.json"),
    ]:
        if getattr(args, attr) is None:
            setattr(args, attr, model_dir / suffix)
    return args


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit < len(np.unique(labels)) * 2:
        raise ValueError("dev-limit is too small for stratified smoke validation")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def train_one_fold(
    *,
    family_name: str,
    family_config: dict,
    loss_type: str,
    gamma: float,
    batch_size: int,
    epochs: int,
    patience: int,
    grad_clip: float,
    num_workers: int,
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
    family_offset = 0 if family_name == FT_FAMILY.model_name else 100_000
    set_seed(seed + fold_idx * 1000 + family_offset)

    mean = np.mean(x_train_raw[tr_idx], axis=0, keepdims=True).astype(np.float32)
    std = np.std(x_train_raw[tr_idx], axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    x_tr = apply_standardizer(x_train_raw[tr_idx], mean, std)
    x_va = apply_standardizer(x_train_raw[va_idx], mean, std)
    x_test = apply_standardizer(x_test_raw, mean, std)
    y_tr = y[tr_idx].astype(np.int64)
    y_va = y[va_idx].astype(np.int64)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(family_name, x_train_raw.shape[1], n_classes, family_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=family_config["lr"],
        weight_decay=family_config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    class_weights = torch.tensor(class_balanced_weights(y_tr), dtype=torch.float32, device=device)
    criterion = build_loss(loss_type=loss_type, class_weights=class_weights, gamma=gamma)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    best_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0

    print(f"[{VERSION}] {family_name} seed={seed} fold={fold_idx}/{total_folds} train_rows={len(tr_idx)} val_rows={len(va_idx)}")
    epoch_bar = tqdm(
        range(1, epochs + 1),
        desc=f"{VERSION} {family_name} fold {fold_idx}",
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
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu()) * len(batch_y)
            seen += len(batch_y)

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_probs = predict_probs(model, x_va, device, batch_size, num_workers)
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
            patience=f"{bad_epochs}/{patience}",
        )
        if bad_epochs >= patience:
            break

    model.load_state_dict(best_state)
    va_probs = predict_probs(model, x_va, device, batch_size, num_workers)
    test_probs = predict_probs(model, x_test, device, batch_size, num_workers)
    record = {
        "family": family_name,
        "seed": seed,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_macro_f1": best_f1,
        "best_train_loss": best_loss,
        "mean": mean.squeeze(0).astype(np.float32),
        "std": std.squeeze(0).astype(np.float32),
        "state_dict": best_state,
    }
    print(f"[{VERSION}] {family_name} fold={fold_idx} done best_macro_f1={best_f1:.6f} best_epoch={best_epoch}")
    return va_probs, test_probs, record


def train_family(
    *,
    family_name: str,
    family_config: dict,
    args: argparse.Namespace,
    x_train_raw: np.ndarray,
    y: np.ndarray,
    x_test_raw: np.ndarray,
    device: torch.device,
    n_classes: int,
    label_names: list[str],
) -> tuple[np.ndarray, np.ndarray, dict, list[dict]]:
    family_oof = np.zeros((len(x_train_raw), n_classes), dtype=np.float32)
    family_test = np.zeros((len(x_test_raw), n_classes), dtype=np.float32)
    records: list[dict] = []
    cv_total_runs = len(args.seeds) * args.folds

    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_raw, y), start=1):
            va_probs, fold_test_probs, record = train_one_fold(
                family_name=family_name,
                family_config=family_config,
                loss_type=args.loss_type,
                gamma=args.gamma,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                grad_clip=args.grad_clip,
                num_workers=args.num_workers,
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
            family_oof[va_idx] += va_probs / len(args.seeds)
            family_test += fold_test_probs / cv_total_runs
            records.append(record)
            append_action_log(
                args.action_log,
                f"{VERSION} {family_name} seed={seed} fold={fold_idx}/{args.folds} finished: "
                f"best_macro_f1={record['best_macro_f1']:.6f}, best_epoch={record['best_epoch']}.",
            )

    summary = summarize_predictions(y, family_oof, label_names)
    summary.update(
        {
            "model_name": family_name,
            "config": family_config,
            "fold_records": [
                {
                    "seed": r["seed"],
                    "fold": r["fold"],
                    "best_epoch": r["best_epoch"],
                    "best_macro_f1": r["best_macro_f1"],
                    "best_train_loss": r["best_train_loss"],
                }
                for r in records
            ],
        }
    )
    return family_oof, family_test, summary, records


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available. Use the iscc-gpu environment or pass --allow-cpu for a smoke check.")

    for path in [
        args.bundle_path,
        args.ft_oof_path,
        args.ft_test_path,
        args.mlp_oof_path,
        args.mlp_test_path,
        args.oof_path,
        args.test_path,
        args.report_path,
    ]:
        path.parent.mkdir(parents=True, exist_ok=True)

    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    feature_cols = ensure_feature_columns(train_df, test_df)
    label_encoder = LabelEncoder()
    y_full = label_encoder.fit_transform(train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()

    selected_idx = select_dev_indices(y_full, args.dev_limit, args.seeds[0])
    train_df = train_df.iloc[selected_idx].reset_index(drop=True)
    y = y_full[selected_idx]
    x_train_raw = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test_raw = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    n_classes = len(label_names)

    append_action_log(
        args.action_log,
        f"{VERSION} training started: device={device}, train_rows={len(train_df)}/{len(y_full)}, "
        f"test_rows={len(test_df)}, features={len(feature_cols)}, labels={n_classes}, "
        f"seeds={args.seeds}, folds={args.folds}, loss_type={args.loss_type}.",
    )

    print(f"[{VERSION}] training FT family")
    ft_oof, ft_test, ft_report, ft_records = train_family(
        family_name=FT_FAMILY.model_name,
        family_config=FT_FAMILY.config,
        args=args,
        x_train_raw=x_train_raw,
        y=y,
        x_test_raw=x_test_raw,
        device=device,
        n_classes=n_classes,
        label_names=label_names,
    )
    np.save(args.ft_oof_path, ft_oof)
    np.save(args.ft_test_path, ft_test)

    print(f"[{VERSION}] training MLP family")
    mlp_oof, mlp_test, mlp_report, mlp_records = train_family(
        family_name=MLP_FAMILY.model_name,
        family_config=MLP_FAMILY.config,
        args=args,
        x_train_raw=x_train_raw,
        y=y,
        x_test_raw=x_test_raw,
        device=device,
        n_classes=n_classes,
        label_names=label_names,
    )
    np.save(args.mlp_oof_path, mlp_oof)
    np.save(args.mlp_test_path, mlp_test)

    search_best = search_blend_weight(y, ft_oof, mlp_oof, grid=np.linspace(0.0, 1.0, 51))
    ft_macro_f1 = ft_report["macro_f1"]
    mlp_macro_f1 = mlp_report["macro_f1"]
    best_single_family = FT_FAMILY.model_name if ft_macro_f1 >= mlp_macro_f1 else MLP_FAMILY.model_name
    best_single_macro_f1 = max(ft_macro_f1, mlp_macro_f1)

    selected_source = "ensemble" if search_best["macro_f1"] > best_single_macro_f1 else f"single_{best_single_family}"
    if selected_source == "ensemble":
        selected_weight_ft = float(search_best["weight_a"])
    else:
        selected_weight_ft = 1.0 if best_single_family == FT_FAMILY.model_name else 0.0

    selected_weight_mlp = 1.0 - selected_weight_ft
    selected_oof = blend_probs(ft_oof, mlp_oof, selected_weight_ft)
    selected_test = blend_probs(ft_test, mlp_test, selected_weight_ft)
    selected_report = summarize_predictions(y, selected_oof, label_names)
    np.save(args.oof_path, selected_oof)
    np.save(args.test_path, selected_test)

    bundle = {
        "schema_version": 2,
        "version": VERSION,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "feature_columns": feature_cols,
        "label_names": label_names,
        "seeds": args.seeds,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss_type": args.loss_type,
        "gamma": args.gamma,
        "grad_clip": args.grad_clip,
        "families": {
            FT_FAMILY.model_name: {
                "config": FT_FAMILY.config,
                "fold_models": ft_records,
            },
            MLP_FAMILY.model_name: {
                "config": MLP_FAMILY.config,
                "fold_models": mlp_records,
            },
        },
        "ensemble": {
            "search_best_weight_ft": search_best["weight_a"],
            "search_best_weight_mlp": search_best["weight_b"],
            "search_best_macro_f1": search_best["macro_f1"],
            "search_best_accuracy": search_best["accuracy"],
            "best_single_family": best_single_family,
            "best_single_macro_f1": best_single_macro_f1,
            "selected_source": selected_source,
            "selected_weight_ft": selected_weight_ft,
            "selected_weight_mlp": selected_weight_mlp,
            "selected_macro_f1": selected_report["macro_f1"],
            "selected_accuracy": selected_report["accuracy"],
        },
        "family_reports": {
            FT_FAMILY.model_name: ft_report,
            MLP_FAMILY.model_name: mlp_report,
        },
        "selected_report": selected_report,
    }
    torch_save_bundle(bundle, args.bundle_path)

    report = {
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
        "loss_type": args.loss_type,
        "gamma": args.gamma,
        "grad_clip": args.grad_clip,
        "train_rows": int(len(train_df)),
        "full_train_rows": int(len(y_full)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "label_count": int(n_classes),
        "label_names": label_names,
        "ft": ft_report,
        "mlp": mlp_report,
        "ensemble_search": search_best,
        "selected": {
            "source": selected_source,
            "weight_ft": selected_weight_ft,
            "weight_mlp": selected_weight_mlp,
            "macro_f1": selected_report["macro_f1"],
            "accuracy": selected_report["accuracy"],
        },
    }
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"{VERSION} training completed: selected={selected_source}; macro_f1={selected_report['macro_f1']:.6f}; "
        f"accuracy={selected_report['accuracy']:.6f}; bundle={args.bundle_path}.",
    )

    print(json.dumps({"version": VERSION, "selected": selected_source, "macro_f1": selected_report["macro_f1"], "accuracy": selected_report["accuracy"]}, indent=2))
    print(f"Wrote bundle: {args.bundle_path}")
    print(f"Wrote ft oof: {args.ft_oof_path}")
    print(f"Wrote ft test: {args.ft_test_path}")
    print(f"Wrote mlp oof: {args.mlp_oof_path}")
    print(f"Wrote mlp test: {args.mlp_test_path}")
    print(f"Wrote selected oof: {args.oof_path}")
    print(f"Wrote selected test: {args.test_path}")
    print(f"Wrote report: {args.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
