from __future__ import annotations

import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


VERSION = "v1.4"
DEFAULT_SEEDS = [20260504, 20260505]


class GEGLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = x.chunk(2, dim=-1)
        return value * torch.nn.functional.gelu(gate)


class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features: int, d_token: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.bias, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


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
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        head_dropout: float,
    ) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.Sequential(
            *[
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
        cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.blocks(tokens)
        cls_out = tokens[:, 0]
        mean_out = tokens[:, 1:].mean(dim=1)
        return self.head(torch.cat([cls_out, mean_out], dim=1))


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_robust_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0, keepdims=True).astype(np.float32)
    q25 = np.nanpercentile(x, 25, axis=0, keepdims=True).astype(np.float32)
    q75 = np.nanpercentile(x, 75, axis=0, keepdims=True).astype(np.float32)
    scale = (q75 - q25).astype(np.float32)
    fallback = np.nanstd(x, axis=0, keepdims=True).astype(np.float32)
    scale = np.where(scale < 1e-6, fallback, scale).astype(np.float32)
    scale = np.where(scale < 1e-6, 1.0, scale).astype(np.float32)
    return center, scale


def apply_robust_stats(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    x = (x - center) / scale
    x = np.nan_to_num(x, nan=0.0, posinf=8.0, neginf=-8.0)
    return np.clip(x, -8.0, 8.0).astype(np.float32)


def build_model(n_features: int, n_classes: int, config: dict) -> nn.Module:
    return FTTransformerClassifier(
        n_features=n_features,
        n_classes=n_classes,
        d_token=config["d_token"],
        n_blocks=config["n_blocks"],
        n_heads=config["n_heads"],
        d_ffn=config["d_ffn"],
        attention_dropout=config["attention_dropout"],
        residual_dropout=config["residual_dropout"],
        ffn_dropout=config["ffn_dropout"],
        head_dropout=config["head_dropout"],
    )


def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


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

