from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import Booster
from sklearn.metrics import classification_report, confusion_matrix, f1_score

LABELS = [0, 1, 2]
N_STUDENTS = 3
N_FOLDS = 5
SUBMISSION_NAME = "submission_robust_model_balanced.csv"
FINAL_BIAS = [1.0, 1.0791, 1.0007]
DEFAULT_PRIOR = (0.49, 0.26, 0.24, 0)
MODEL_DIR_NAME = "\u6a21\u578b\uff08\u5fc5\u4ea4\uff09"
OUTPUT_DIR_NAME = "\u63d0\u4ea4\u7ed3\u679c\uff08\u5fc5\u4ea4\uff09"

StatsMaps = Tuple[
    Dict[Tuple[int, ...], int],
    Dict[Tuple[int, ...], int],
    Dict[Tuple[int, ...], Tuple[float, float, float, int]],
]


def package_root(source_file: str | Path) -> Path:
    return Path(source_file).resolve().parents[1]


def default_model_dir(root: Path) -> Path:
    return root / MODEL_DIR_NAME


def default_output_dir(root: Path) -> Path:
    return root / OUTPUT_DIR_NAME


def resolve_data_path(root: Path, kind: str, explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"{kind} not found: {path}")

    filename = f"data_{kind}.csv"
    candidates = [
        root / "data" / filename,
        root.parent / "data" / filename,
        root.parent / "powershell-main" / "data" / filename,
        Path.cwd() / "data" / filename,
    ]
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError(f"Cannot find {filename}; pass --{kind} explicitly.")


def normalize(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-12, None)
    return probs / probs.sum(axis=1, keepdims=True)


def key_tuple(row: Iterable[int]) -> Tuple[int, ...]:
    return tuple(int(v) for v in row)


def feature_columns(train: pd.DataFrame) -> List[str]:
    return [c for c in train.columns if c not in ("name", "label")]


def build_key_stats(train: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    grouped = train.groupby(list(features))["label"]
    stats = grouped.agg(train_count="size", label_nunique="nunique").reset_index()
    frac = (
        train.groupby(list(features))["label"]
        .value_counts(normalize=True)
        .unstack(fill_value=0.0)
        .reset_index()
    )
    for label in LABELS:
        if label not in frac.columns:
            frac[label] = 0.0
    frac = frac[list(features) + LABELS].rename(
        columns={0: "c0_frac", 1: "c1_frac", 2: "c2_frac"}
    )
    return stats.merge(frac, on=list(features), how="left")


def save_key_stats(stats: pd.DataFrame, model_dir: Path) -> None:
    stats.to_json(model_dir / "key_stats.json", orient="records", force_ascii=False)
    stale_csv = model_dir / "key_stats.csv"
    if stale_csv.exists():
        stale_csv.unlink()


def load_key_stats(model_dir: Path) -> pd.DataFrame:
    return pd.read_json(model_dir / "key_stats.json")


def stats_maps(stats: pd.DataFrame, features: Sequence[str]) -> StatsMaps:
    count_map: Dict[Tuple[int, ...], int] = {}
    nunique_map: Dict[Tuple[int, ...], int] = {}
    dist_map: Dict[Tuple[int, ...], Tuple[float, float, float, int]] = {}
    n_features = len(features)

    for row in stats.itertuples(index=False):
        key = key_tuple(row[:n_features])
        count = int(getattr(row, "train_count"))
        count_map[key] = count
        nunique_map[key] = int(getattr(row, "label_nunique"))
        dist_map[key] = (
            float(getattr(row, "c0_frac")),
            float(getattr(row, "c1_frac")),
            float(getattr(row, "c2_frac")),
            count,
        )
    return count_map, nunique_map, dist_map


def make_model_features(
    frame: pd.DataFrame,
    features: Sequence[str],
    pair_features: Sequence[Sequence[str]],
    maps: StatsMaps,
) -> np.ndarray:
    count_map, nunique_map, dist_map = maps
    part = frame[list(features)].copy()
    blocks = [part[c].to_numpy(dtype=np.float32).reshape(-1, 1) for c in features]

    for c1, c2 in pair_features:
        cross = part[c1].astype(np.int16) * 10 + part[c2].astype(np.int16)
        blocks.append(cross.to_numpy(dtype=np.float32).reshape(-1, 1))

    keys = [key_tuple(row) for row in part.itertuples(index=False, name=None)]
    freq = np.array([count_map.get(key, 0) for key in keys], dtype=np.float32)
    blocks.append(freq.reshape(-1, 1))
    blocks.append(np.log1p(freq).reshape(-1, 1))
    blocks.append(part.sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1))
    blocks.append((part > 0).sum(axis=1).to_numpy(dtype=np.float32).reshape(-1, 1))

    conflict = np.array([nunique_map.get(key, 1) for key in keys], dtype=np.float32)
    dist = np.array([dist_map.get(key, DEFAULT_PRIOR) for key in keys], dtype=np.float32)
    blocks.extend(
        [
            np.log1p(freq).reshape(-1, 1),
            conflict.reshape(-1, 1),
            (conflict == 1).astype(np.float32).reshape(-1, 1),
            (conflict >= 2).astype(np.float32).reshape(-1, 1),
            dist[:, 0:1],
            dist[:, 1:2],
            dist[:, 2:3],
            dist[:, 3:4],
        ]
    )
    return np.column_stack(blocks)


def load_metadata(model_dir: Path) -> dict:
    return json.loads((model_dir / "metadata.json").read_text(encoding="utf-8-sig"))


def save_metadata(model_dir: Path, metadata: dict) -> None:
    (model_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def predict_stage(stage_dir: Path, x: np.ndarray, n_students: int, n_folds: int, labels: List[int]) -> np.ndarray:
    student_predictions = []
    for student_id in range(n_students):
        probs = np.zeros((len(x), len(labels)), dtype=float)
        for fold_id in range(n_folds):
            for class_pos, class_id in enumerate(labels):
                path = stage_dir / f"s{student_id}_fold{fold_id}_class{class_id}.txt"
                booster = Booster(model_str=path.read_text(encoding="utf-8"))
                probs[:, class_pos] += booster.predict(x) / n_folds
        student_predictions.append(normalize(probs))
    return normalize(np.mean(student_predictions, axis=0))


def candidate_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict:
    pred = probs.argmax(axis=1)
    report = classification_report(y_true, pred, labels=LABELS, output_dict=True, zero_division=0)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "class2_precision": float(report["2"]["precision"]),
        "class2_recall": float(report["2"]["recall"]),
        "class2_f1": float(report["2"]["f1-score"]),
        "label_distribution": {str(k): int(v) for k, v in pd.Series(pred).value_counts().sort_index().items()},
        "confusion_matrix": confusion_matrix(y_true, pred, labels=LABELS).tolist(),
    }


def write_submission(test: pd.DataFrame, labels: np.ndarray, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / name
    pd.DataFrame({"name": test["name"].to_numpy(), "label": labels.astype(int)}).to_csv(
        out_path, index=False, encoding="utf-8"
    )
    return out_path
