"""PyTorch multitask model and helper utilities for the v1.4 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from utils import tqdm


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


def build_balanced_sampler_weights(
    y_cwe: np.ndarray, train_idx: np.ndarray, num_classes: int
) -> np.ndarray:
    """Sample weights for WeightedRandomSampler: 1/sqrt(class_count) per sample."""
    valid_mask = y_cwe[train_idx] >= 0
    cwe_vals = y_cwe[train_idx]
    counts = np.bincount(cwe_vals[valid_mask], minlength=num_classes).astype(np.float64)
    counts[counts <= 0] = 1.0
    class_weight = 1.0 / np.sqrt(counts)
    weights = np.ones(len(train_idx), dtype=np.float64)
    for i, c in enumerate(cwe_vals):
        if c >= 0:
            weights[i] = class_weight[c]
    weights /= weights.sum()
    return weights


class FocalLoss(nn.Module):
    """Focal Loss for long-tail multi-class classification."""

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else torch.empty(0))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (1 - pt) ** self.gamma * ce
        if self.alpha.numel() > 0:
            focal = self.alpha[targets] * focal
        return focal.mean()


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
    """Byte + metadata network with self-attention and dual heads (v2.0)."""

    def __init__(
        self,
        tabular_dim: int,
        num_cwe_classes: int,
        dropout: float = 0.25,
        byte_embedding_dim: int = 48,
    ) -> None:
        super().__init__()
        self.byte_embedding = nn.Embedding(256, byte_embedding_dim)
        self.byte_encoder = nn.Sequential(
            nn.Conv1d(byte_embedding_dim, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            ResidualConvBlock(64, dropout * 0.4),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.GELU(),
            ResidualConvBlock(128, dropout * 0.4),
            nn.Conv1d(128, 192, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(192),
            nn.GELU(),
            ResidualConvBlock(192, dropout * 0.4),
            nn.Conv1d(192, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU(),
            ResidualConvBlock(256, dropout * 0.4),
            nn.Conv1d(256, 320, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(320),
            nn.GELU(),
        )
        # Self-attention over byte sequence positions (v2.0)
        self.byte_attn = nn.MultiheadAttention(embed_dim=320, num_heads=4, dropout=dropout * 0.3, batch_first=True)
        self.byte_attn_norm = nn.LayerNorm(320)
        self.byte_projection = nn.Sequential(
            nn.Linear(640, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(384, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )
        self.byte_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.byte_max_pool = nn.AdaptiveMaxPool1d(1)

        self.tabular_encoder = nn.Sequential(
            nn.Linear(tabular_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 320),
            nn.LayerNorm(320),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.fusion = nn.Sequential(
            nn.Linear(320 + 320, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(384, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout * 0.25),
        )
        self.label_head = nn.Linear(256, 1)
        self.cwe_head = nn.Linear(256, num_cwe_classes)

    def forward(self, byte_x: torch.Tensor, tabular_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if byte_x.dim() == 3 and byte_x.size(1) == 1:
            byte_x = byte_x.squeeze(1)
        byte_x = byte_x.long().clamp(0, 255)
        byte_x = self.byte_embedding(byte_x).permute(0, 2, 1).contiguous()
        byte_feat = self.byte_encoder(byte_x)  # (B, 320, S)
        # Self-attention over sequence positions (v2.0)
        byte_feat_t = byte_feat.permute(0, 2, 1)  # (B, S, 320)
        attn_out, _ = self.byte_attn(byte_feat_t, byte_feat_t, byte_feat_t)
        byte_feat_t = self.byte_attn_norm(byte_feat_t + attn_out)
        byte_feat = byte_feat_t.permute(0, 2, 1)  # (B, 320, S)
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


class FusionMLP(nn.Module):
    """Meta-learner that fuses tree and neural probabilities per-class."""

    def __init__(self, num_cwe_classes: int, hidden: int = 192):
        super().__init__()
        input_dim = 1 + num_cwe_classes + 1 + num_cwe_classes
        self.label_net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(hidden // 2, 1),
        )
        self.cwe_net = nn.Sequential(
            nn.Linear(input_dim, hidden * 2),
            nn.BatchNorm1d(hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.40),
            nn.Linear(hidden * 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(hidden, num_cwe_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.label_net(x).squeeze(-1), self.cwe_net(x)


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


@torch.no_grad()
def predict_fusion_mlp(
    model: FusionMLP,
    tree_label: np.ndarray,
    tree_cwe: np.ndarray,
    neural_label: np.ndarray,
    neural_cwe: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    X = np.concatenate(
        [
            tree_label.reshape(-1, 1),
            tree_cwe,
            neural_label.reshape(-1, 1),
            neural_cwe,
        ],
        axis=1,
    ).astype(np.float32)
    label_out: List[np.ndarray] = []
    cwe_out: List[np.ndarray] = []
    for slc in _chunk_indices(len(X), batch_size):
        batch = torch.from_numpy(X[slc]).to(device)
        label_logits, cwe_logits = model(batch)
        label_out.append(torch.sigmoid(label_logits).cpu().numpy())
        cwe_out.append(torch.softmax(cwe_logits, dim=1).cpu().numpy())
    return np.concatenate(label_out), np.concatenate(cwe_out)
