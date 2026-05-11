from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


LABEL_COL = "label"
ID_COL = "id"
DEFAULT_SEED = 20260504
MODEL_WEIGHTS = {"hgb": 0.6, "et": 0.4}


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in {ID_COL, LABEL_COL}]


def ensure_feature_columns(train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
    train_features = feature_columns(train_df)
    test_features = [c for c in test_df.columns if c != ID_COL]
    if train_features != test_features:
        missing_in_test = [c for c in train_features if c not in test_features]
        extra_in_test = [c for c in test_features if c not in train_features]
        raise ValueError(
            "train/test feature columns do not match: "
            f"missing_in_test={missing_in_test[:5]}, extra_in_test={extra_in_test[:5]}"
        )
    return train_features


def make_feature_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    frame = df.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return frame.astype(np.float32)


def encode_labels(labels: Sequence[str]):
    from sklearn.preprocessing import LabelEncoder

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return encoder, y


def safe_f1_macro(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def append_action_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def validate_prediction_frame(
    submission: pd.DataFrame,
    test_df: pd.DataFrame,
    allowed_labels: Iterable[str],
    expected_columns: Sequence[str],
) -> None:
    allowed_set = set(allowed_labels)
    if list(submission.columns) != list(expected_columns):
        raise ValueError(
            f"submission columns mismatch: got {list(submission.columns)}, "
            f"expected {list(expected_columns)}"
        )
    if len(submission) != len(test_df):
        raise ValueError(
            f"submission row count {len(submission)} does not match test row count {len(test_df)}"
        )
    if submission["id"].tolist() != test_df["id"].tolist():
        raise ValueError("submission ids do not match test ids exactly")
    if submission["label"].isna().any():
        raise ValueError("submission contains missing labels")
    invalid = sorted(set(submission["label"]) - allowed_set)
    if invalid:
        raise ValueError(f"submission contains invalid labels: {invalid}")


def classification_summary(y_true, y_pred, label_names: Sequence[str]) -> dict:
    from sklearn.metrics import confusion_matrix

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": safe_f1_macro(y_true, y_pred),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                label_names,
                f1_score(y_true, y_pred, average=None, labels=list(range(len(label_names)))),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(len(label_names)))).tolist(),
    }
