from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

VERSION = "v1.7"
DEFAULT_SEEDS = [20260504, 20260505, 20260506]


def log_key_metrics(root: Path, metrics: dict) -> None:
    from datetime import datetime
    record_path = root / "KEY_METRICS.md"
    existing = ""
    if record_path.exists():
        existing = record_path.read_text(encoding="utf-8")
    if not existing:
        existing = (
            "# KEY_METRICS — v1.5+ 迭代关键数据\n\n"
            "| 时间 | 版本 | 类型 | 模型 | features | seeds | folds | "
            "local_acc | local_macro_f1 | 弱类F1 | 平台分 | 备注 |\n"
            "|------|------|------|------|----------|-------|-------|"
            "-----------|---------------|--------|--------|------|\n"
        )
    ts = datetime.now().strftime("%m-%d %H:%M")
    row = (
        f"| {ts} | {metrics.get('version', '-')} | {metrics.get('stage', '-')} | "
        f"{metrics.get('model', '-')} | {metrics.get('n_features', '-')} | "
        f"{metrics.get('seeds', '-')} | {metrics.get('folds', '-')} | "
        f"{metrics.get('local_acc', '-')} | {metrics.get('local_macro_f1', '-')} | "
        f"{metrics.get('weak_f1', '-')} | {metrics.get('platform_score', '-')} | "
        f"{metrics.get('notes', '-')} |\n"
    )
    with record_path.open("w", encoding="utf-8") as f:
        f.write(existing + row)


def save_bundle(path: Path, obj: dict) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_bundle(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)
