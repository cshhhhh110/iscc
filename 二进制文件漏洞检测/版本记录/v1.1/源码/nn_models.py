"""PyTorch multitask model and helper utilities for the v1.1 pipeline."""

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
    weights = counts.sum() / (len(counts) * counts)
    weights = np.clip(weights, 0.25, 20.0)
    return torch.tensor(weights, dtype=torch.float32)


class ByteMetaMultiTaskNet(nn.Module):
    """Lightweight byte + metadata network with binary and CWE heads."""

    def __init__(
        self,
        tabular_dim: int,
        num_cwe_classes: int,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.byte_encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
        )
        self.byte_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.byte_max_pool = nn.AdaptiveMaxPool1d(1)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fusion = nn.Sequential(
            nn.Linear(256 + 128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_head = nn.Linear(256, 1)
        self.cwe_head = nn.Linear(256, num_cwe_classes)

    def forward(self, byte_x: torch.Tensor, tabular_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if byte_x.dim() == 2:
            byte_x = byte_x.unsqueeze(1)
        byte_x = byte_x.float().div(255.0)
        byte_feat = self.byte_encoder(byte_x)
        byte_feat = torch.cat(
            [
                self.byte_avg_pool(byte_feat).squeeze(-1),
                self.byte_max_pool(byte_feat).squeeze(-1),
            ],
            dim=1,
        )
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
