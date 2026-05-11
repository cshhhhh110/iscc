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
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from common import LABEL_COL, append_action_log, ensure_feature_columns, make_feature_frame, read_table
from gpu_v1_3_core import (
    DEFAULT_SEEDS,
    MODEL_NAME,
    MODEL_SPEC,
    VERSION,
    apply_class_bias,
    apply_standardizer,
    build_loss,
    build_model,
    choose_best_candidate,
    class_balanced_weights,
    fit_standardizer,
    parse_seeds,
    predict_probs,
    search_class_bias,
    set_seed,
    summarize_predictions,
    torch_save_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Train the {VERSION} pure-GPU compact ResMLP.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--oof-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds(",".join(str(x) for x in DEFAULT_SEEDS)))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--loss-type", choices=["ce", "weighted_ce"], default="ce")
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--class-power", type=float, default=0.35)
    parser.add_argument("--selection-metric", choices=["accuracy", "macro_f1"], default="accuracy")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--feature-noise", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--dev-limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny test and write smoke_* outputs.")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.dev_limit = args.dev_limit or 720
        args.folds = 2
        args.epochs = min(args.epochs, 2)
        args.patience = 1
        args.batch_size = min(args.batch_size, 256)
        args.seeds = args.seeds[:1]

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    prefix = "smoke_" if args.smoke else ""
    model_dir = root / "模型"
    defaults = {
        "bundle_path": f"{prefix}gpu_model_bundle_{VERSION}.pt",
        "oof_path": f"{prefix}gpu_resmlp_oof_probs_{VERSION}.npy",
        "test_path": f"{prefix}gpu_resmlp_test_probs_{VERSION}.npy",
        "report_path": f"{prefix}gpu_validation_report_{VERSION}.json",
    }
    for attr, filename in defaults.items():
        if getattr(args, attr) is None:
            setattr(args, attr, model_dir / filename)
    return args


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    if limit < len(np.unique(labels)) * 2:
        raise ValueError("dev-limit is too small for stratified smoke validation")
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def metric_is_better(
    current_primary: float,
    current_tie: float,
    best_primary: float,
    best_tie: float,
) -> bool:
    return current_primary > best_primary + 1e-12 or (
        abs(current_primary - best_primary) <= 1e-12 and current_tie > best_tie + 1e-12
    )


def save_npy_with_retry(array: np.ndarray, path: Path, retries: int = 5, delay: float = 1.0) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, retries + 1):
        try:
            np.save(path, array)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def train_one_fold(
    *,
    config: dict,
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

    mean, std = fit_standardizer(x_train_raw[tr_idx])
    x_tr = apply_standardizer(x_train_raw[tr_idx], mean, std)
    x_va = apply_standardizer(x_train_raw[va_idx], mean, std)
    x_test = apply_standardizer(x_test_raw, mean, std)
    y_tr = y[tr_idx].astype(np.int64)
    y_va = y[va_idx].astype(np.int64)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )

    model = build_model(x_train_raw.shape[1], n_classes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    class_weights = torch.tensor(
        class_balanced_weights(y_tr, power=args.class_power),
        dtype=torch.float32,
        device=device,
    )
    criterion = build_loss(
        loss_type=args.loss_type,
        class_weights=class_weights if args.loss_type == "weighted_ce" else None,
        label_smoothing=args.label_smoothing,
    )
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_state = deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
    best_epoch = 0
    best_score = -1.0
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_loss = float("inf")
    bad_epochs = 0

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc=f"{VERSION} seed={seed} fold={fold_idx}/{total_folds}",
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
            if args.feature_noise > 0:
                batch_x = torch.clamp(batch_x + torch.randn_like(batch_x) * args.feature_noise, -8.0, 8.0)

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
        current_primary = val_accuracy if args.selection_metric == "accuracy" else val_macro_f1
        current_tie = val_macro_f1 if args.selection_metric == "accuracy" else val_accuracy
        best_tie = best_macro_f1 if args.selection_metric == "accuracy" else best_accuracy

        if metric_is_better(current_primary, current_tie, best_score, best_tie):
            best_score = current_primary
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
        "model_name": MODEL_NAME,
        "seed": seed,
        "fold": fold_idx,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_accuracy": best_accuracy,
        "best_macro_f1": best_macro_f1,
        "best_train_loss": best_loss,
        "mean": mean.squeeze(0).astype(np.float32),
        "std": std.squeeze(0).astype(np.float32),
        "state_dict": best_state,
    }
    return va_probs, test_probs, record


def train_cv(
    *,
    args: argparse.Namespace,
    x_train_raw: np.ndarray,
    y: np.ndarray,
    x_test_raw: np.ndarray,
    device: torch.device,
    n_classes: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], list[dict]]:
    all_oof = np.zeros((len(x_train_raw), n_classes), dtype=np.float32)
    all_test = np.zeros((len(x_test_raw), n_classes), dtype=np.float32)
    seed_oofs: dict[str, np.ndarray] = {}
    seed_tests: dict[str, np.ndarray] = {}
    records: list[dict] = []

    for seed in args.seeds:
        seed_key = str(seed)
        seed_oof = np.zeros_like(all_oof)
        seed_test = np.zeros_like(all_test)
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train_raw, y), start=1):
            print(
                f"[{VERSION}] {MODEL_NAME} seed={seed} fold={fold_idx}/{args.folds} "
                f"train_rows={len(tr_idx)} val_rows={len(va_idx)}"
            )
            va_probs, fold_test_probs, record = train_one_fold(
                config=MODEL_SPEC.config,
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
            seed_oof[va_idx] = va_probs
            seed_test += fold_test_probs / args.folds
            records.append(record)

            append_action_log(
                args.action_log,
                f"{VERSION} {MODEL_NAME} seed={seed} fold={fold_idx}/{args.folds} finished: "
                f"best_accuracy={record['best_accuracy']:.6f}, "
                f"best_macro_f1={record['best_macro_f1']:.6f}, best_epoch={record['best_epoch']}.",
            )
            print(
                f"[{VERSION}] seed={seed} fold={fold_idx}/{args.folds} done "
                f"best_accuracy={record['best_accuracy']:.6f} "
                f"best_macro_f1={record['best_macro_f1']:.6f} best_epoch={record['best_epoch']}"
            )

        seed_oofs[seed_key] = seed_oof.astype(np.float32)
        seed_tests[seed_key] = seed_test.astype(np.float32)
        all_oof += seed_oof / len(args.seeds)
        all_test += seed_test / len(args.seeds)

    return all_oof.astype(np.float32), all_test.astype(np.float32), seed_oofs, seed_tests, records


def build_candidates(
    *,
    y: np.ndarray,
    label_names: list[str],
    all_oof: np.ndarray,
    all_test: np.ndarray,
    seed_oofs: dict[str, np.ndarray],
    seed_tests: dict[str, np.ndarray],
    selection_metric: str,
) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    bias_acc = search_class_bias(y, all_oof, metric="accuracy")
    bias_macro = search_class_bias(y, all_oof, metric="macro_f1")

    candidate_oof: dict[str, np.ndarray] = {"all_average": all_oof}
    candidate_test: dict[str, np.ndarray] = {"all_average": all_test}
    for seed_key in sorted(seed_oofs):
        candidate_oof[f"seed_{seed_key}"] = seed_oofs[seed_key]
        candidate_test[f"seed_{seed_key}"] = seed_tests[seed_key]
    candidate_oof["bias_accuracy"] = bias_acc["probs"]
    candidate_test["bias_accuracy"] = apply_class_bias(all_test, bias_acc["bias"])
    candidate_oof["bias_macro_f1"] = bias_macro["probs"]
    candidate_test["bias_macro_f1"] = apply_class_bias(all_test, bias_macro["bias"])

    candidate_reports = {
        name: summarize_predictions(y, probs, label_names)
        for name, probs in candidate_oof.items()
    }
    selected_source = choose_best_candidate(candidate_reports, metric=selection_metric)
    bias_candidates = {
        "accuracy": {k: v for k, v in bias_acc.items() if k != "probs"},
        "macro_f1": {k: v for k, v in bias_macro.items() if k != "probs"},
    }
    selected = {
        "source": selected_source,
        "selection_metric": selection_metric,
        "accuracy": candidate_reports[selected_source]["accuracy"],
        "macro_f1": candidate_reports[selected_source]["macro_f1"],
    }
    candidate_metadata = {
        "selected": selected,
        "candidate_reports": candidate_reports,
        "bias_candidates": bias_candidates,
    }
    return candidate_metadata, candidate_oof[selected_source], candidate_test[selected_source], candidate_test


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

    for path in [args.bundle_path, args.oof_path, args.test_path, args.report_path]:
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
        f"{VERSION} training started: smoke={args.smoke}, device={device}, "
        f"train_rows={len(train_df)}/{len(y_full)}, test_rows={len(test_df)}, features={len(feature_cols)}, "
        f"labels={n_classes}, seeds={args.seeds}, folds={args.folds}, loss_type={args.loss_type}, "
        f"selection_metric={args.selection_metric}.",
    )

    all_oof, all_test, seed_oofs, seed_tests, records = train_cv(
        args=args,
        x_train_raw=x_train_raw,
        y=y,
        x_test_raw=x_test_raw,
        device=device,
        n_classes=n_classes,
    )
    candidate_metadata, selected_oof, selected_test, _ = build_candidates(
        y=y,
        label_names=label_names,
        all_oof=all_oof,
        all_test=all_test,
        seed_oofs=seed_oofs,
        seed_tests=seed_tests,
        selection_metric=args.selection_metric,
    )
    selected = candidate_metadata["selected"]

    save_npy_with_retry(selected_oof, args.oof_path)
    save_npy_with_retry(selected_test, args.test_path)

    bundle = {
        "schema_version": 1,
        "version": VERSION,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "model_name": MODEL_NAME,
        "model_config": MODEL_SPEC.config,
        "feature_columns": feature_cols,
        "label_names": label_names,
        "seeds": args.seeds,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "class_power": args.class_power,
        "selection_metric": args.selection_metric,
        "grad_clip": args.grad_clip,
        "feature_noise": args.feature_noise,
        "fold_records": records,
        "selected": selected,
        "candidate_reports": candidate_metadata["candidate_reports"],
        "bias_candidates": candidate_metadata["bias_candidates"],
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
        "model_name": MODEL_NAME,
        "model_config": MODEL_SPEC.config,
        "seeds": args.seeds,
        "folds": args.folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "loss_type": args.loss_type,
        "label_smoothing": args.label_smoothing,
        "class_power": args.class_power,
        "selection_metric": args.selection_metric,
        "grad_clip": args.grad_clip,
        "feature_noise": args.feature_noise,
        "train_rows": int(len(train_df)),
        "full_train_rows": int(len(y_full)),
        "test_rows": int(len(test_df)),
        "feature_count": int(len(feature_cols)),
        "label_count": int(n_classes),
        "label_names": label_names,
        "selected": selected,
        "candidate_reports": candidate_metadata["candidate_reports"],
        "bias_candidates": candidate_metadata["bias_candidates"],
        "fold_records": [
            {
                "seed": r["seed"],
                "fold": r["fold"],
                "best_epoch": r["best_epoch"],
                "best_score": r["best_score"],
                "best_accuracy": r["best_accuracy"],
                "best_macro_f1": r["best_macro_f1"],
                "best_train_loss": r["best_train_loss"],
            }
            for r in records
        ],
    }
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"{VERSION} training completed: selected={selected['source']}; "
        f"accuracy={selected['accuracy']:.6f}; macro_f1={selected['macro_f1']:.6f}; "
        f"bundle={args.bundle_path}.",
    )

    print(
        json.dumps(
            {
                "version": VERSION,
                "selected": selected,
                "bundle": str(args.bundle_path),
                "oof_probs": str(args.oof_path),
                "test_probs": str(args.test_path),
                "report": str(args.report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
