from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VERSION = "v1.2"
DEFAULT_SEEDS = [20260504, 20260505]
FT_MODEL_NAME = "ft_transformer"
MLP_MODEL_NAME = "residual_mlp"


@dataclass(frozen=True)
class FamilySpec:
    model_name: str
    config: dict


FT_FAMILY = FamilySpec(
    model_name=FT_MODEL_NAME,
    config={
        "d_token": 64,
        "n_blocks": 5,
        "n_heads": 8,
        "d_ffn": 192,
        "token_dropout": 0.03,
        "attention_dropout": 0.10,
        "residual_dropout": 0.06,
        "ffn_dropout": 0.10,
        "head_dropout": 0.15,
        "lr": 8e-4,
        "weight_decay": 2e-4,
    },
)

MLP_FAMILY = FamilySpec(
    model_name=MLP_MODEL_NAME,
    config={
        "hidden_dim": 384,
        "block_hidden_dim": 768,
        "n_blocks": 5,
        "stem_dropout": 0.05,
        "block_dropout": 0.12,
        "head_dropout": 0.15,
        "lr": 1.2e-3,
        "weight_decay": 1.5e-4,
    },
)


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * torch.nn.functional.gelu(gate)


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int, dropout: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return self.dropout(tokens)


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
        token_dropout: float,
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token, token_dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.ModuleList(
            [
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
        cls = self.cls_token.expand(tokens.size(0), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = tokens[:, 0]
        mean_out = tokens[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls_out, mean_out], dim=1))


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim * 2)
        self.act = GEGLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(self.norm(x))
        h = self.act(h)
        h = self.dropout(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return x + h


class ResidualMLPClassifier(nn.Module):
    def __init__(
        self,
        n_features: int,
        n_classes: int,
        hidden_dim: int,
        block_hidden_dim: int,
        n_blocks: int,
        stem_dropout: float,
        block_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(stem_dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, block_hidden_dim, block_dropout) for _ in range(n_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
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


def class_balanced_weights(labels: np.ndarray, beta: float = 0.9995) -> np.ndarray:
    counts = np.bincount(labels.astype(np.int64))
    counts = np.maximum(counts, 1).astype(np.float32)
    if beta <= 0.0 or beta >= 1.0:
        weights = 1.0 / np.sqrt(counts)
    else:
        weights = (1.0 - beta) / (1.0 - np.power(beta, counts))
    weights = weights / weights.mean()
    return weights.astype(np.float32)


class WeightedFocalLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor | None = None, gamma: float = 1.5) -> None:
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        ce = F.nll_loss(log_probs, targets, reduction="none")
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt).clamp(min=1e-6).pow(self.gamma)
        if self.class_weights is not None:
            alpha = self.class_weights.gather(0, targets)
        else:
            alpha = 1.0
        loss = alpha * focal * ce
        return loss.mean()


def build_loss(
    loss_type: str,
    class_weights: torch.Tensor,
    gamma: float,
) -> nn.Module:
    if loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    if loss_type == "focal":
        return WeightedFocalLoss(class_weights=class_weights, gamma=gamma)
    raise ValueError(f"unsupported loss_type: {loss_type}")


def build_model(
    family_name: str,
    n_features: int,
    n_classes: int,
    config: dict,
) -> nn.Module:
    if family_name == FT_MODEL_NAME:
        return FTTransformerClassifier(
            n_features=n_features,
            n_classes=n_classes,
            d_token=config["d_token"],
            n_blocks=config["n_blocks"],
            n_heads=config["n_heads"],
            d_ffn=config["d_ffn"],
            token_dropout=config["token_dropout"],
            attention_dropout=config["attention_dropout"],
            residual_dropout=config["residual_dropout"],
            ffn_dropout=config["ffn_dropout"],
            head_dropout=config["head_dropout"],
        )
    if family_name == MLP_MODEL_NAME:
        return ResidualMLPClassifier(
            n_features=n_features,
            n_classes=n_classes,
            hidden_dim=config["hidden_dim"],
            block_hidden_dim=config["block_hidden_dim"],
            n_blocks=config["n_blocks"],
            stem_dropout=config["stem_dropout"],
            block_dropout=config["block_dropout"],
            head_dropout=config["head_dropout"],
        )
    raise ValueError(f"unsupported family_name: {family_name}")


def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def torch_save_bundle(bundle: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, path)


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
    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(batch_x)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def summarize_predictions(y_true: np.ndarray, probs: np.ndarray, label_names: Sequence[str]) -> dict:
    pred = np.argmax(probs, axis=1)
    from sklearn.metrics import accuracy_score, confusion_matrix

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


def blend_probs(probs_a: np.ndarray, probs_b: np.ndarray, weight_a: float) -> np.ndarray:
    weight_a = float(weight_a)
    weight_b = 1.0 - weight_a
    return weight_a * probs_a + weight_b * probs_b


def search_blend_weight(
    y_true: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    grid: Iterable[float] | None = None,
) -> dict:
    if grid is None:
        grid = np.linspace(0.0, 1.0, 51)
    best = {
        "weight_a": 0.0,
        "weight_b": 1.0,
        "macro_f1": -1.0,
        "accuracy": -1.0,
    }
    for weight_a in grid:
        probs = blend_probs(probs_a, probs_b, float(weight_a))
        pred = np.argmax(probs, axis=1)
        macro_f1 = float(f1_score(y_true, pred, average="macro"))
        if macro_f1 > best["macro_f1"]:
            from sklearn.metrics import accuracy_score

            best = {
                "weight_a": float(weight_a),
                "weight_b": float(1.0 - weight_a),
                "macro_f1": macro_f1,
                "accuracy": float(accuracy_score(y_true, pred)),
            }
    return best


def predict_family_probs_from_bundle(
    bundle: dict,
    family_name: str,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> np.ndarray:
    family = bundle["families"][family_name]
    n_features = len(bundle["feature_columns"])
    n_classes = len(bundle["label_names"])
    family_probs = np.zeros((len(x), n_classes), dtype=np.float32)
    fold_models = family["fold_models"]
    for record in fold_models:
        model = build_model(family_name, n_features, n_classes, family["config"]).to(device)
        model.load_state_dict(record["state_dict"])
        x_norm = apply_standardizer(
            x,
            np.asarray(record["mean"], dtype=np.float32).reshape(1, -1),
            np.asarray(record["std"], dtype=np.float32).reshape(1, -1),
        )
        family_probs += predict_probs(model, x_norm, device, batch_size, num_workers) / max(len(fold_models), 1)
    return family_probs


def predict_probs_from_bundle(
    bundle: dict,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    weight_ft: float | None = None,
    weight_mlp: float | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    ft_name = FT_FAMILY.model_name
    mlp_name = MLP_FAMILY.model_name
    family_probs: dict[str, np.ndarray] = {}
    if ft_name in bundle["families"]:
        family_probs[ft_name] = predict_family_probs_from_bundle(
            bundle, ft_name, x, device, batch_size, num_workers
        )
    if mlp_name in bundle["families"]:
        family_probs[mlp_name] = predict_family_probs_from_bundle(
            bundle, mlp_name, x, device, batch_size, num_workers
        )

    ensemble_meta = bundle.get("ensemble", {})
    if weight_ft is None:
        weight_ft = float(ensemble_meta.get("selected_weight_ft", 1.0))
    if weight_mlp is None:
        weight_mlp = float(ensemble_meta.get("selected_weight_mlp", 0.0))

    if ft_name not in family_probs:
        final_probs = family_probs[mlp_name]
    elif mlp_name not in family_probs:
        final_probs = family_probs[ft_name]
    else:
        weight_sum = weight_ft + weight_mlp
        if weight_sum <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        final_probs = (weight_ft * family_probs[ft_name] + weight_mlp * family_probs[mlp_name]) / weight_sum

    return final_probs.astype(np.float32), family_probs, ensemble_meta


def build_family_report(y_true: np.ndarray, probs: np.ndarray, label_names: Sequence[str]) -> dict:
    return summarize_predictions(y_true, probs, label_names)
