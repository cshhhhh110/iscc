from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

VERSION = "v1.8"
DEFAULT_SEEDS = [20260504, 20260505, 20260506]


# ── FT-Transformer (v1.4 architecture) ──────────────────────────────────

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
    def __init__(self, d_token: int, n_heads: int, d_ffn: int,
                 attention_dropout: float, residual_dropout: float, ffn_dropout: float) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads,
                                          dropout=attention_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(residual_dropout)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn * 2), GEGLU(), nn.Dropout(ffn_dropout),
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
    def __init__(self, n_features: int, n_classes: int, d_token: int, n_blocks: int,
                 n_heads: int, d_ffn: int, attention_dropout: float, residual_dropout: float,
                 ffn_dropout: float, head_dropout: float) -> None:
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.Sequential(*[
            TransformerBlock(d_token=d_token, n_heads=n_heads, d_ffn=d_ffn,
                             attention_dropout=attention_dropout,
                             residual_dropout=residual_dropout,
                             ffn_dropout=ffn_dropout)
            for _ in range(n_blocks)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_token * 2), nn.Linear(d_token * 2, d_token * 2),
            nn.GELU(), nn.Dropout(head_dropout), nn.Linear(d_token * 2, n_classes),
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


# ── SWA utilities ───────────────────────────────────────────────────────

class SWA:
    """Stochastic Weight Averaging: maintains running average of model weights."""

    def __init__(self, model: nn.Module, start_epoch: int):
        self._model = model
        self._start_epoch = start_epoch
        self._swa_state: dict | None = None
        self._n_updates = 0

    @property
    def n_updates(self) -> int:
        return self._n_updates

    def step(self, epoch: int) -> None:
        if epoch < self._start_epoch:
            return
        current = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
        if self._swa_state is None:
            self._swa_state = current
            self._n_updates = 1
        else:
            self._n_updates += 1
            for k in self._swa_state:
                self._swa_state[k] = self._swa_state[k] + (current[k] - self._swa_state[k]) / self._n_updates

    def apply(self) -> dict:
        """Return SWA state dict (caller loads it into model)."""
        if self._swa_state is None:
            raise RuntimeError("SWA has not been updated yet (epoch < start_epoch)")
        return deepcopy(self._swa_state)


# ── Shared utilities ────────────────────────────────────────────────────

def parse_seeds(value: str) -> list[int]:
    seeds = [int(s.strip()) for s in value.split(",") if s.strip()]
    if not seeds:
        raise ValueError("at least one seed required")
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
        n_features=n_features, n_classes=n_classes,
        d_token=config["d_token"], n_blocks=config["n_blocks"],
        n_heads=config["n_heads"], d_ffn=config["d_ffn"],
        attention_dropout=config["attention_dropout"],
        residual_dropout=config["residual_dropout"],
        ffn_dropout=config["ffn_dropout"],
        head_dropout=config["head_dropout"],
    )


def predict_probs(model: nn.Module, x: np.ndarray, device: torch.device,
                  batch_size: int, num_workers: int) -> np.ndarray:
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)),
                        batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda")
    amp = device.type == "cuda"
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp):
                logits = model(batch_x)
            chunks.append(torch.softmax(logits, dim=1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def torch_load_bundle(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


# ── Key metrics logging ─────────────────────────────────────────────────

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
