"""PyTorch multitask model and helper utilities for the v1.2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm


@dataclass(frozen=True)
class TabularNormalizer:
    mean: np.ndarray
    std: np.ndarray


def fit_tabular_normalizer(X: np.ndarray) -> TabularNormalizer:
    mean = np.asarray(X, dtype=np.float32).mean(axis=0)
    std = np.asarray(X, dtype=np.float32).std(axis=0)
    std[std < 1e-6] = 1.0
    return TabularNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def apply_tabular_normalizer(X: np.ndarray, normalizer: TabularNormalizer) -> np.ndarray:
    data = np.asarray(X, dtype=np.float32)
    normalized = (data - normalizer.mean) / normalizer.std
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(normalized, -10.0, 10.0).astype(np.float32, copy=False)


def build_cwe_class_weights(y_cwe: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_cwe[y_cwe >= 0], minlength=num_classes).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    weights = np.clip(weights, 0.35, 12.0)
    return torch.tensor(weights, dtype=torch.float32)


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ByteMetaMultiTaskNet(nn.Module):
    """Lightweight byte + metadata network with binary and CWE heads."""

    def __init__(
        self,
        tabular_dim: int,
        num_cwe_classes: int,
        dropout: float = 0.25,
        byte_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.byte_embedding = nn.Embedding(256, byte_embedding_dim)
        self.byte_encoder = nn.Sequential(
            nn.Conv1d(byte_embedding_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualConvBlock(64, dropout * 0.5),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualConvBlock(128, dropout * 0.5),
            nn.Conv1d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
            ResidualConvBlock(192, dropout * 0.5),
            nn.Conv1d(192, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
        )
        self.byte_projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.byte_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.byte_max_pool = nn.AdaptiveMaxPool1d(1)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 192),
            nn.LayerNorm(192),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 192, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(320, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_head = nn.Linear(256, 1)
        self.cwe_head = nn.Linear(256, num_cwe_classes)

    def forward(self, byte_x: torch.Tensor, tabular_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if byte_x.dim() == 3 and byte_x.size(1) == 1:
            byte_x = byte_x.squeeze(1)
        byte_x = byte_x.long().clamp(0, 255)
        byte_x = self.byte_embedding(byte_x).permute(0, 2, 1).contiguous()
        byte_feat = self.byte_encoder(byte_x)
        byte_feat = torch.cat(
            [
                self.byte_avg_pool(byte_feat).squeeze(-1),
                self.byte_max_pool(byte_feat).squeeze(-1),
            ],
            dim=1,
        )
        byte_feat = self.byte_projection(byte_feat)
        tab_feat = self.tabular_encoder(tabular_x.float())
        fused = self.fusion(torch.cat([byte_feat, tab_feat], dim=1))
        label_logits = self.label_head(fused).squeeze(-1)
        cwe_logits = self.cwe_head(fused)
        return label_logits, cwe_logits


def _chunk_indices(n_items: int, batch_size: int) -> Iterable[slice]:
    for start in range(0, n_items, batch_size):
        yield slice(start, min(start + batch_size, n_items))


@torch.no_grad()
def predict_multitask(
    model: ByteMetaMultiTaskNet,
    byte_matrix: np.ndarray,
    tabular_matrix: np.ndarray,
    batch_size: int,
    device: torch.device,
    desc: str,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    label_probs: List[np.ndarray] = []
    cwe_probs: List[np.ndarray] = []
    total_steps = ceil(len(byte_matrix) / batch_size) if len(byte_matrix) else 0
    with tqdm(total=total_steps, desc=desc, unit="batch") as progress:
        for slc in _chunk_indices(len(byte_matrix), batch_size):
            byte_batch = torch.from_numpy(byte_matrix[slc]).to(device)
            tab_batch = torch.from_numpy(tabular_matrix[slc]).to(device)
            label_logits, cwe_logits = model(byte_batch, tab_batch)
            label_probs.append(torch.sigmoid(label_logits).cpu().numpy())
            cwe_probs.append(torch.softmax(cwe_logits, dim=1).cpu().numpy())
            progress.update(1)
    if label_probs:
        return np.concatenate(label_probs, axis=0), np.concatenate(cwe_probs, axis=0)
    return np.empty((0,), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
