from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VERSION = "v1.3"
DEFAULT_SEEDS = [20260504, 20260505]
MODEL_NAME = "compact_resmlp"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    config: dict


MODEL_SPEC = ModelSpec(
    model_name=MODEL_NAME,
    config={
        "hidden_dim": 320,
        "block_hidden_dim": 640,
        "n_blocks": 4,
        "feature_dropout": 0.02,
        "stem_dropout": 0.04,
        "block_dropout": 0.06,
        "head_dropout": 0.08,
        "bn_momentum": 0.05,
        "lr": 8e-4,
        "weight_decay": 8e-5,
    },
)


class FeatureExpansion(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.nan_to_num(x.float(), nan=0.0, posinf=8.0, neginf=-8.0)
        x2 = torch.clamp(x * x, max=64.0)
        return self.dropout(torch.cat([x, x2, torch.abs(x)], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float, bn_momentum: float) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(dim, momentum=bn_momentum)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc1(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        return x + self.out_dropout(h)


class CompactResMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int,
        block_hidden_dim: int,
        n_blocks: int,
        feature_dropout: float,
        stem_dropout: float,
        block_dropout: float,
        head_dropout: float,
        bn_momentum: float,
    ) -> None:
        super().__init__()
        expanded_dim = n_features * 3
        self.expand = FeatureExpansion(feature_dropout)
        self.stem = nn.Sequential(
            nn.BatchNorm1d(expanded_dim, momentum=bn_momentum),
            nn.Linear(expanded_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(stem_dropout),
        )
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_dim, block_hidden_dim, block_dropout, bn_momentum)
                for _ in range(n_blocks)
            ]
        )
        self.head = nn.Sequential(
            nn.BatchNorm1d(hidden_dim, momentum=bn_momentum),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(self.expand(x))
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0, keepdims=True).astype(np.float32)
    std = np.std(x, axis=0, keepdims=True).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def apply_standardizer(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (x - mean) / std
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(x, -8.0, 8.0).astype(np.float32)


def class_balanced_weights(labels: np.ndarray, power: float = 0.35) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64))
    counts = np.maximum(counts, 1).astype(np.float32)
    weights = np.power(np.mean(counts) / counts, power)
    weights = weights / np.mean(weights)
    return weights.astype(np.float32)


def build_model(n_features: int, n_classes: int, config: dict) -> nn.Module:
    return CompactResMLPClassifier(
        n_features=n_features,
        n_classes=n_classes,
        hidden_dim=config["hidden_dim"],
        block_hidden_dim=config["block_hidden_dim"],
        n_blocks=config["n_blocks"],
        feature_dropout=config["feature_dropout"],
        stem_dropout=config["stem_dropout"],
        block_dropout=config["block_dropout"],
        head_dropout=config["head_dropout"],
        bn_momentum=config["bn_momentum"],
    )


def build_loss(loss_type: str, class_weights: torch.Tensor | None, label_smoothing: float) -> nn.Module:
    if loss_type == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"unsupported loss_type: {loss_type}")


def predict_probs(
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
    with torch.inference_mode():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def summarize_predictions(y_true: np.ndarray, probs: np.ndarray, label_names: Sequence[str]) -> dict:
    pred = np.argmax(probs, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, average="macro")),
        "per_class_f1": {
            label: float(score)
            for label, score in zip(
                label_names,
                f1_score(y_true, pred, average=None, labels=list(range(len(label_names)))),
            )
        },
        "confusion_matrix": confusion_matrix(y_true, pred, labels=list(range(len(label_names)))).tolist(),
    }


def apply_class_bias(probs: np.ndarray, bias: Sequence[float]) -> np.ndarray:
    bias_arr = np.asarray(bias, dtype=np.float32).reshape(1, -1)
    logits = np.log(np.clip(probs, 1e-12, 1.0)) + bias_arr
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / np.sum(exp_logits, axis=1, keepdims=True)).astype(np.float32)


def _score_biased_logits(
    y_true: np.ndarray,
    log_probs: np.ndarray,
    bias: np.ndarray,
    metric: str,
) -> float:
    pred = np.argmax(log_probs + bias.reshape(1, -1), axis=1)
    if metric == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric == "macro_f1":
        return float(f1_score(y_true, pred, average="macro"))
    raise ValueError(f"unsupported metric: {metric}")


def search_class_bias(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str,
    rounds: Sequence[float] = (0.18, 0.08, 0.03),
) -> dict:
    log_probs = np.log(np.clip(probs, 1e-12, 1.0))
    bias = np.zeros(probs.shape[1], dtype=np.float32)
    best_score = _score_biased_logits(y_true, log_probs, bias, metric)
    for step in rounds:
        for class_idx in range(probs.shape[1]):
            current = float(bias[class_idx])
            best_local = current
            best_local_score = best_score
            for multiplier in (-2.0, -1.0, 0.0, 1.0, 2.0):
                trial = bias.copy()
                trial[class_idx] = current + step * multiplier
                trial = trial - np.mean(trial)
                score = _score_biased_logits(y_true, log_probs, trial, metric)
                if score > best_local_score + 1e-12:
                    best_local_score = score
                    best_local = float(trial[class_idx])
            if best_local_score > best_score + 1e-12:
                bias[class_idx] = best_local
                bias = bias - np.mean(bias)
                best_score = best_local_score

    biased_probs = apply_class_bias(probs, bias)
    return {
        "metric": metric,
        "score": best_score,
        "bias": bias.astype(np.float32).tolist(),
        "rounds": [float(x) for x in rounds],
        "probs": biased_probs,
    }


def choose_best_candidate(candidate_reports: dict[str, dict], metric: str = "accuracy") -> str:
    if metric not in {"accuracy", "macro_f1"}:
        raise ValueError(f"unsupported selection metric: {metric}")
    tie_metric = "macro_f1" if metric == "accuracy" else "accuracy"
    return max(
        candidate_reports,
        key=lambda name: (
            candidate_reports[name][metric],
            candidate_reports[name][tie_metric],
            1 if name == "all_average" else 0,
        ),
    )


def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


def predict_record_probs(
    bundle: dict,
    record: dict,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    n_features = len(bundle["feature_columns"])
    n_classes = len(bundle["label_names"])
    model = build_model(n_features, n_classes, bundle["model_config"]).to(device)
    model.load_state_dict(record["state_dict"])
    x_norm = apply_standardizer(
        x,
        np.asarray(record["mean"], dtype=np.float32).reshape(1, -1),
        np.asarray(record["std"], dtype=np.float32).reshape(1, -1),
    )
    return predict_probs(model, x_norm, device, batch_size, num_workers)


def predict_probs_from_bundle(
    bundle: dict,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    records = bundle["fold_records"]
    n_classes = len(bundle["label_names"])
    all_probs = np.zeros((len(x), n_classes), dtype=np.float32)
    seed_sums: dict[str, np.ndarray] = {}
    seed_counts: dict[str, int] = {}

    for record in records:
        probs = predict_record_probs(bundle, record, x, device, batch_size, num_workers)
        all_probs += probs / max(len(records), 1)
        seed_key = str(record["seed"])
        if seed_key not in seed_sums:
            seed_sums[seed_key] = np.zeros_like(all_probs)
            seed_counts[seed_key] = 0
        seed_sums[seed_key] += probs
        seed_counts[seed_key] += 1

    seed_probs = {
        seed_key: (seed_sums[seed_key] / max(seed_counts[seed_key], 1)).astype(np.float32)
        for seed_key in seed_sums
    }
    return all_probs.astype(np.float32), seed_probs


def build_candidate_probs_from_bundle(
    bundle: dict,
    all_probs: np.ndarray,
    seed_probs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidates: dict[str, np.ndarray] = {"all_average": all_probs}
    for seed_key, probs in sorted(seed_probs.items(), key=lambda item: item[0]):
        candidates[f"seed_{seed_key}"] = probs

    bias_candidates = bundle.get("bias_candidates", {})
    if "accuracy" in bias_candidates:
        candidates["bias_accuracy"] = apply_class_bias(all_probs, bias_candidates["accuracy"]["bias"])
    if "macro_f1" in bias_candidates:
        candidates["bias_macro_f1"] = apply_class_bias(all_probs, bias_candidates["macro_f1"]["bias"])
    return candidates
