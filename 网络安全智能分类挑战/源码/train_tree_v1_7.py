from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from common import LABEL_COL, append_action_log, classification_summary, encode_labels, ensure_feature_columns, make_feature_frame, read_table
from tree_v1_7_core import VERSION, DEFAULT_SEEDS, log_key_metrics, save_bundle


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=f"Train {VERSION} XGBoost + LightGBM ensemble.")
    p.add_argument("--data-dir", type=Path, default=root)
    p.add_argument("--train-file", type=str, default="train_data.csv")
    p.add_argument("--test-file", type=str, default="test_data.csv")
    p.add_argument("--bundle-path", type=Path, default=root / "模型" / f"tree_bundle_{VERSION}.pkl")
    p.add_argument("--report-path", type=Path, default=root / "模型" / f"tree_validation_report_{VERSION}.json")
    p.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--dev-limit", type=int, default=None)
    p.add_argument("--seeds", type=lambda s: [int(x) for x in s.split(",")], default=DEFAULT_SEEDS)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--allow-cpu", action="store_true")
    # XGB param overrides
    p.add_argument("--xgb-lr", type=float, default=0.03)
    p.add_argument("--xgb-depth", type=int, default=6)
    p.add_argument("--xgb-n-estimators", type=int, default=3000)
    # LGBM param overrides
    p.add_argument("--lgbm-lr", type=float, default=0.03)
    p.add_argument("--lgbm-depth", type=int, default=6)
    p.add_argument("--lgbm-n-estimators", type=int, default=3000)
    args = p.parse_args()

    if args.smoke:
        args.dev_limit = args.dev_limit or 720
        args.seeds = args.seeds[:1]
        args.folds = min(args.folds, 2)
        args.xgb_n_estimators = min(args.xgb_n_estimators, 50)
        args.lgbm_n_estimators = min(args.lgbm_n_estimators, 50)
        args.bundle_path = args.bundle_path.with_name("smoke_" + args.bundle_path.name)
        args.report_path = args.report_path.with_name("smoke_" + args.report_path.name)
    return args


def select_dev_indices(labels: np.ndarray, limit: int | None, seed: int) -> np.ndarray:
    if limit is None or limit >= len(labels):
        return np.arange(len(labels))
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=limit, random_state=seed)
    selected, _ = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(selected)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as h:
        json.dump(payload, h, ensure_ascii=False, indent=2)


def train_xgb_fold(x_tr, y_tr, x_va, y_va, x_test, n_classes, args, seed, fold):
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=args.xgb_n_estimators, learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=seed + fold * 1000,
        early_stopping_rounds=100, verbosity=0, n_jobs=-1,
        reg_alpha=0.1, reg_lambda=1.0,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    va_probs = model.predict_proba(x_va).astype(np.float32)
    test_probs = model.predict_proba(x_test).astype(np.float32)
    return va_probs, test_probs, model


def train_lgbm_fold(x_tr, y_tr, x_va, y_va, x_test, n_classes, args, seed, fold):
    from lightgbm import LGBMClassifier
    model = LGBMClassifier(
        n_estimators=args.lgbm_n_estimators, learning_rate=args.lgbm_lr,
        max_depth=args.lgbm_depth, subsample=0.8, colsample_bytree=0.8,
        objective="multiclass", num_class=n_classes,
        random_state=seed + fold * 1000, early_stopping_round=100,
        verbose=-1, n_jobs=-1, reg_alpha=0.1, reg_lambda=1.0,
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)])
    va_probs = model.predict_proba(x_va).astype(np.float32)
    test_probs = model.predict_proba(x_test).astype(np.float32)
    return va_probs, test_probs, model


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()

    for p in [args.bundle_path, args.report_path]:
        p.parent.mkdir(parents=True, exist_ok=True)

    full_train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    feature_cols = ensure_feature_columns(full_train_df, test_df)
    label_encoder, y_full = encode_labels(full_train_df[LABEL_COL].astype(str))
    label_names = label_encoder.classes_.tolist()
    n_classes = len(label_names)

    selected_idx = select_dev_indices(y_full, args.dev_limit, DEFAULT_SEEDS[0])
    train_df = full_train_df.iloc[selected_idx].reset_index(drop=True)
    y = y_full[selected_idx]

    x_train = make_feature_frame(train_df, feature_cols).to_numpy(dtype=np.float32)
    x_test = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)

    append_action_log(args.action_log,
        f"{VERSION} training: rows={len(train_df)}/{len(full_train_df)}, test={len(test_df)}, "
        f"features={len(feature_cols)}, labels={n_classes}, seeds={args.seeds}, folds={args.folds}, "
        f"xgb=(lr={args.xgb_lr},d={args.xgb_depth},n={args.xgb_n_estimators}), "
        f"lgbm=(lr={args.lgbm_lr},d={args.lgbm_depth},n={args.lgbm_n_estimators})")

    n_runs = len(args.seeds) * args.folds
    xgb_oof = np.zeros((len(train_df), n_classes), dtype=np.float32)
    xgb_test = np.zeros((len(test_df), n_classes), dtype=np.float32)
    lgbm_oof = np.zeros_like(xgb_oof)
    lgbm_test = np.zeros_like(xgb_test)
    xgb_models, lgbm_models = [], []

    t0 = time.perf_counter()
    for seed in args.seeds:
        cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(x_train, y), start=1):
            x_tr, y_tr = x_train[tr_idx], y[tr_idx]
            x_va, y_va = x_train[va_idx], y[va_idx]

            # XGBoost
            va_p, te_p, model = train_xgb_fold(x_tr, y_tr, x_va, y_va, x_test, n_classes, args, seed, fold_idx)
            xgb_oof[va_idx] += va_p / len(args.seeds)
            xgb_test += te_p / n_runs
            xgb_models.append(model)
            va_mf1 = float(f1_score(y_va, np.argmax(va_p, axis=1), average="macro"))
            print(f"[XGB] seed={seed} fold={fold_idx} mf1={va_mf1:.4f}")

            # LightGBM
            va_p, te_p, model = train_lgbm_fold(x_tr, y_tr, x_va, y_va, x_test, n_classes, args, seed, fold_idx)
            lgbm_oof[va_idx] += va_p / len(args.seeds)
            lgbm_test += te_p / n_runs
            lgbm_models.append(model)
            va_mf1 = float(f1_score(y_va, np.argmax(va_p, axis=1), average="macro"))
            print(f"[LGBM] seed={seed} fold={fold_idx} mf1={va_mf1:.4f}")

            prefix = "smoke_" if args.smoke else ""
            append_action_log(args.action_log,
                f"{VERSION} XGB seed={seed} fold={fold_idx}/{args.folds}: "
                f"mf1={float(f1_score(y_va, np.argmax(xgb_oof[va_idx] * len(args.seeds), axis=1), average='macro')):.4f} "
                f"| LGBM fold={fold_idx}: "
                f"mf1={float(f1_score(y_va, np.argmax(lgbm_oof[va_idx] * len(args.seeds), axis=1), average='macro')):.4f}")

    elapsed = time.perf_counter() - t0

    # Ensemble: average XGB + LGBM
    ens_oof = (xgb_oof + lgbm_oof) / 2.0
    ens_test = (xgb_test + lgbm_test) / 2.0

    ens_pred = np.argmax(ens_oof, axis=1)
    xgb_pred = np.argmax(xgb_oof, axis=1)
    lgbm_pred = np.argmax(lgbm_oof, axis=1)

    report = classification_summary(y, ens_pred, label_names)
    weak_f1 = min(report["per_class_f1"].values())
    weak_class = min(report["per_class_f1"], key=report["per_class_f1"].get)
    report.update({
        "version": VERSION, "smoke": bool(args.smoke),
        "xgb_macro_f1": float(f1_score(y, xgb_pred, average="macro")),
        "xgb_accuracy": float(accuracy_score(y, xgb_pred)),
        "lgbm_macro_f1": float(f1_score(y, lgbm_pred, average="macro")),
        "lgbm_accuracy": float(accuracy_score(y, lgbm_pred)),
        "seeds": args.seeds, "folds": args.folds,
        "xgb_params": {"lr": args.xgb_lr, "depth": args.xgb_depth, "n_estimators": args.xgb_n_estimators},
        "lgbm_params": {"lr": args.lgbm_lr, "depth": args.lgbm_depth, "n_estimators": args.lgbm_n_estimators},
        "train_rows": int(len(train_df)), "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols), "label_count": n_classes, "label_names": label_names,
        "elapsed_min": round(elapsed / 60, 1),
    })

    # Save bundle with models, OOF/TEST probabilities
    bundle = {
        "version": VERSION, "feature_columns": feature_cols, "label_names": label_names,
        "xgb_models": xgb_models, "lgbm_models": lgbm_models,
        "xgb_oof": xgb_oof, "xgb_test": xgb_test,
        "lgbm_oof": lgbm_oof, "lgbm_test": lgbm_test,
        "validation": report,
    }
    save_bundle(args.bundle_path, bundle)
    write_json(args.report_path, report)

    xgb_mf1 = float(f1_score(y, xgb_pred, average="macro"))
    lgbm_mf1 = float(f1_score(y, lgbm_pred, average="macro"))
    summary = (
        f"{VERSION} done ({elapsed/60:.1f}min): "
        f"XGB mf1={xgb_mf1:.4f} | LGBM mf1={lgbm_mf1:.4f} | "
        f"ENS mf1={report['macro_f1']:.4f} acc={report['accuracy']:.4f} "
        f"weak=({weak_class}:{weak_f1:.4f})"
    )
    print(summary)
    append_action_log(args.action_log, summary)

    log_key_metrics(root=data_dir, metrics={
        "version": VERSION, "stage": "smoke" if args.smoke else "full",
        "model": "xgb+lgbm_avg", "n_features": len(feature_cols),
        "seeds": len(args.seeds), "folds": args.folds,
        "local_acc": f"{report['accuracy']:.4f}",
        "local_macro_f1": f"{report['macro_f1']:.4f}",
        "weak_f1": f"{weak_class}:{weak_f1:.4f}",
        "platform_score": "-",
        "notes": f"XGB:{xgb_mf1:.4f} LGBM:{lgbm_mf1:.4f}",
    })

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
