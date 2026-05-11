from __future__ import annotations

import math
import random
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def infer_cardinalities(frame, features: Sequence[str]) -> list[int]:
    cardinalities: list[int] = []
    for column in features:
        values = frame[column].to_numpy()
        if np.isnan(values).any():
            raise ValueError(f"feature {column} contains missing values")
        if not np.issubdtype(values.dtype, np.integer):
            values = values.astype(np.int64)
        min_value = int(values.min())
        if min_value < 0:
            raise ValueError(f"feature {column} must be non-negative, got min={min_value}")
        max_value = int(values.max())
        cardinalities.append(max_value + 1)
    return cardinalities


def frame_to_categorical_array(frame, features: Sequence[str]) -> np.ndarray:
    matrix = frame[features].to_numpy(dtype=np.int64, copy=True)
    if (matrix < 0).any():
        raise ValueError("categorical matrix contains negative values")
    return matrix


def _prepare_pattern_counts(
    frame,
    features: Sequence[str],
    label_column: str,
    num_classes: int,
) -> pd.DataFrame:
    group_cols = list(features)
    counts = frame.groupby(group_cols, sort=False)[label_column].value_counts().unstack(fill_value=0)
    counts = counts.reindex(columns=list(range(num_classes)), fill_value=0)
    return counts.astype(np.float32)


def _pattern_stats(counts: pd.DataFrame, pattern_probs: pd.DataFrame) -> dict[str, float]:
    pattern_sizes = counts.sum(axis=1).astype(np.float32)
    target_counts = (counts > 0).sum(axis=1)
    ambiguous_patterns = int((target_counts > 1).sum())
    ambiguous_rows = int(pattern_sizes[target_counts > 1].sum())
    probs = pattern_probs.to_numpy(dtype=np.float32)
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
    return {
        "unique_patterns": int(len(counts)),
        "ambiguous_patterns": ambiguous_patterns,
        "ambiguous_rows": ambiguous_rows,
        "avg_pattern_size": float(pattern_sizes.mean()),
        "max_pattern_size": float(pattern_sizes.max()),
        "mean_pattern_entropy": float(entropy.mean()),
    }


def build_pattern_soft_targets(
    frame,
    features: Sequence[str],
    label_column: str = "label",
    num_classes: int = 3,
    alpha: float = 0.5,
    sample_weight_power: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    counts = _prepare_pattern_counts(frame, features, label_column, num_classes)
    smoothed = counts.astype(np.float32) + float(alpha)
    pattern_targets = smoothed.div(smoothed.sum(axis=1), axis=0).astype(np.float32)

    pattern_sizes = counts.sum(axis=1).astype(np.float32)
    weights = np.power(np.maximum(pattern_sizes.to_numpy(), 1.0), -float(sample_weight_power)).astype(np.float32)
    weights = weights / float(weights.mean())

    pattern_targets = pattern_targets.reset_index()
    pattern_targets["__pattern_weight"] = weights
    merged = frame[features].merge(pattern_targets, on=list(features), how="left", sort=False)
    target_matrix = merged[list(range(num_classes))].to_numpy(dtype=np.float32)
    row_weights = merged["__pattern_weight"].to_numpy(dtype=np.float32)

    stats = _pattern_stats(counts, pattern_targets)
    return target_matrix, row_weights, stats


def build_pattern_lookup_bundle(
    frame,
    features: Sequence[str],
    label_column: str = "label",
    num_classes: int = 3,
    alpha: float = 0.5,
) -> tuple[dict[tuple[int, ...], np.ndarray], np.ndarray, dict[str, float]]:
    counts = _prepare_pattern_counts(frame, features, label_column, num_classes)
    smoothed = counts + float(alpha)
    pattern_probs = smoothed.div(smoothed.sum(axis=1), axis=0).astype(np.float32)

    lookup = {
        tuple(int(v) for v in index): row.to_numpy(dtype=np.float32)
        for index, row in pattern_probs.iterrows()
    }

    global_counts = frame[label_column].value_counts().reindex(list(range(num_classes)), fill_value=0).astype(np.float32)
    default_proba = (
        (global_counts + float(alpha))
        / float(global_counts.sum() + float(alpha) * num_classes)
    ).to_numpy(dtype=np.float32)

    stats = _pattern_stats(counts, pattern_probs)
    stats.update(
        {
            "default_entropy": float(-(default_proba * np.log(np.clip(default_proba, 1e-12, 1.0))).sum()),
        }
    )
    return lookup, default_proba, stats


class CategoricalDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        labels: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> None:
        self.features = torch.as_tensor(features, dtype=torch.long)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        if weights is None:
            weights = np.ones(len(features), dtype=np.float32)
        self.weights = torch.as_tensor(weights, dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return (
            self.features[index],
            self.targets[index],
            self.labels[index],
            self.weights[index],
        )


class CategoricalInferenceDataset(Dataset):
    def __init__(self, features: np.ndarray) -> None:
        self.features = torch.as_tensor(features, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index]


class EmbeddingMLP(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        num_classes: int = 3,
        embed_dim: int = 12,
        hidden_dims: Sequence[int] = (192, 128),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), embed_dim) for cardinality in self.cardinalities]
        )
        input_dim = len(self.cardinalities) * embed_dim
        layers: list[nn.Module] = [nn.LayerNorm(input_dim)]
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(current_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = [embedding(x[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        representation = torch.cat(tokens, dim=-1)
        representation = self.backbone(representation)
        return self.head(representation)


class CrossLayer(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        interaction = torch.sum(x * self.weight, dim=-1, keepdim=True)
        return x0 * interaction + self.bias + x


class DeepCrossNetwork(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        num_classes: int = 3,
        embed_dim: int = 8,
        cross_layers: int = 3,
        deep_dims: Sequence[int] = (128, 64),
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), embed_dim) for cardinality in self.cardinalities]
        )
        self.input_dim = len(self.cardinalities) * embed_dim
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.cross_layers = nn.ModuleList([CrossLayer(self.input_dim) for _ in range(int(cross_layers))])

        deep_blocks: list[nn.Module] = []
        current_dim = self.input_dim
        for hidden_dim in deep_dims:
            hidden_dim = int(hidden_dim)
            deep_blocks.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        self.deep = nn.Sequential(*deep_blocks) if deep_blocks else nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(self.input_dim + current_dim),
            nn.Linear(self.input_dim + current_dim, max(self.input_dim, current_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(self.input_dim, current_dim), num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = [embedding(x[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        x0 = torch.cat(tokens, dim=-1)
        x0 = self.input_norm(x0)
        x0 = self.input_dropout(x0)
        cross = x0
        for layer in self.cross_layers:
            cross = layer(x0, cross)
        deep = self.deep(x0)
        representation = torch.cat([cross, deep], dim=-1)
        return self.head(representation)


class TabResidualBlock(nn.Module):
    def __init__(self, width: int, expansion: int = 2, dropout: float = 0.12) -> None:
        super().__init__()
        hidden_dim = int(width) * int(expansion)
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, width)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class TabResidualNet(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        num_classes: int = 3,
        embed_dim: int = 8,
        width: int = 192,
        num_blocks: int = 4,
        expansion: int = 2,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), embed_dim) for cardinality in self.cardinalities]
        )
        input_dim = len(self.cardinalities) * embed_dim
        self.stem = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [TabResidualBlock(width=width, expansion=expansion, dropout=dropout) for _ in range(int(num_blocks))]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = [embedding(x[:, idx]) for idx, embedding in enumerate(self.embeddings)]
        representation = torch.cat(tokens, dim=-1)
        representation = self.stem(representation)
        for block in self.blocks:
            representation = block(representation)
        return self.head(representation)


class TinyTransformer(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        num_classes: int = 3,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        ff_dim: int = 96,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.cardinalities = list(cardinalities)
        self.num_features = len(self.cardinalities)
        self.value_embeddings = nn.ModuleList(
            [nn.Embedding(int(cardinality), d_model) for cardinality in self.cardinalities]
        )
        self.feature_embeddings = nn.Embedding(self.num_features, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for idx, embedding in enumerate(self.value_embeddings):
            token = embedding(x[:, idx]) + self.feature_embeddings.weight[idx].unsqueeze(0)
            tokens.append(token)
        token_tensor = torch.stack(tokens, dim=1)
        cls_token = self.cls_token.expand(token_tensor.shape[0], -1, -1)
        token_tensor = torch.cat([cls_token, token_tensor], dim=1)
        encoded = self.encoder(token_tensor)
        representation = self.norm(encoded[:, 0])
        return self.head(representation)


def build_torch_model(
    arch: str,
    cardinalities: Sequence[int],
    num_classes: int,
    config: Mapping[str, object],
) -> nn.Module:
    arch = arch.lower().strip()
    if arch in {"dcn", "deep_cross_network"}:
        return DeepCrossNetwork(
            cardinalities=cardinalities,
            num_classes=num_classes,
            embed_dim=int(config.get("embed_dim", 8)),
            cross_layers=int(config.get("cross_layers", 3)),
            deep_dims=tuple(int(x) for x in config.get("deep_dims", (128, 64))),
            dropout=float(config.get("dropout", 0.12)),
        )
    if arch in {"tab_resnet", "tabresnet", "tab_residual_net"}:
        return TabResidualNet(
            cardinalities=cardinalities,
            num_classes=num_classes,
            embed_dim=int(config.get("embed_dim", 8)),
            width=int(config.get("width", 192)),
            num_blocks=int(config.get("num_blocks", 4)),
            expansion=int(config.get("expansion", 2)),
            dropout=float(config.get("dropout", 0.12)),
        )
    if arch == "mlp":
        return EmbeddingMLP(
            cardinalities=cardinalities,
            num_classes=num_classes,
            embed_dim=int(config.get("embed_dim", 12)),
            hidden_dims=tuple(int(x) for x in config.get("hidden_dims", (192, 128))),
            dropout=float(config.get("dropout", 0.15)),
        )
    if arch in {"transformer", "tiny_transformer"}:
        return TinyTransformer(
            cardinalities=cardinalities,
            num_classes=num_classes,
            d_model=int(config.get("d_model", 32)),
            num_heads=int(config.get("num_heads", 4)),
            num_layers=int(config.get("num_layers", 2)),
            ff_dim=int(config.get("ff_dim", 96)),
            dropout=float(config.get("dropout", 0.10)),
        )
    raise ValueError(f"unknown torch arch: {arch}")


def _move_state_dict_to_cpu(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def soft_cross_entropy(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = -(target_probs * log_probs).sum(dim=-1)
    if sample_weights is not None:
        loss = loss * sample_weights
    return loss.mean()


def predict_proba_model(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device | None,
    batch_size: int = 1024,
) -> np.ndarray:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    dataset = CategoricalInferenceDataset(features)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for batch_features in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            logits = model(batch_features)
            proba = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
            outputs.append(proba)
    return np.concatenate(outputs, axis=0)


def build_model_from_bundle(bundle: Mapping[str, object], device: torch.device | None) -> nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_torch_model(
        arch=str(bundle["arch"]),
        cardinalities=bundle["cardinalities"],
        num_classes=int(bundle.get("num_classes", 3)),
        config=bundle["config"],
    )
    state_dict = bundle["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def predict_torch_ensemble_proba(
    fold_bundles: Sequence[Mapping[str, object]],
    features: np.ndarray,
    device: torch.device | None,
) -> np.ndarray:
    if not fold_bundles:
        raise ValueError("torch ensemble has no folds")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proba = None
    for fold_bundle in fold_bundles:
        model = build_model_from_bundle(fold_bundle, device)
        fold_proba = predict_proba_model(
            model,
            features,
            device=device,
            batch_size=int(fold_bundle.get("eval_batch_size", 1024)),
        )
        if proba is None:
            proba = np.zeros_like(fold_proba, dtype=np.float32)
        proba += fold_proba
    return proba / float(len(fold_bundles))


def train_torch_fold(
    arch: str,
    cardinalities: Sequence[int],
    train_features: np.ndarray,
    train_targets: np.ndarray,
    train_labels: np.ndarray,
    train_weights: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    config: Mapping[str, object],
    device: torch.device,
    seed: int,
    desc: str,
) -> dict[str, object]:
    seed_everything(seed)
    model = build_torch_model(arch, cardinalities, num_classes=train_targets.shape[1], config=config)
    model.to(device)

    batch_size = int(config.get("batch_size", 1024))
    eval_batch_size = int(config.get("eval_batch_size", 2048))
    max_epochs = int(config.get("max_epochs", 120))
    patience = int(config.get("patience", 15))
    lr = float(config.get("lr", 2e-3))
    weight_decay = float(config.get("weight_decay", 1e-4))
    label_smoothing = float(config.get("label_smoothing", 0.05))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))

    dataset = CategoricalDataset(train_features, train_targets, train_labels, train_weights)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config.get("amp", True)))

    best_score = -1.0
    best_epoch = 0
    best_state = _move_state_dict_to_cpu(model.state_dict())
    best_val_proba = np.zeros((len(valid_features), train_targets.shape[1]), dtype=np.float32)
    best_val_loss = math.inf
    stale_epochs = 0

    epoch_iterator = tqdm(range(1, max_epochs + 1), desc=desc, leave=False)
    for epoch in epoch_iterator:
        model.train()
        for batch_features, batch_targets, _, batch_weights in loader:
            batch_features = batch_features.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            batch_weights = batch_weights.to(device, non_blocking=True)
            if label_smoothing > 0:
                batch_targets = batch_targets * (1.0 - label_smoothing) + label_smoothing / batch_targets.shape[-1]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                logits = model(batch_features)
                loss = soft_cross_entropy(logits, batch_targets, batch_weights)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

        model.eval()
        fold_proba = predict_proba_model(model, valid_features, device=device, batch_size=eval_batch_size)
        fold_pred = fold_proba.argmax(axis=1)
        macro_f1 = float(f1_score(valid_labels, fold_pred, average="macro"))
        val_loss = float(
            -(np.eye(train_targets.shape[1], dtype=np.float32)[valid_labels] * np.log(np.clip(fold_proba, 1e-12, 1.0))).sum(axis=1).mean()
        )
        if macro_f1 > best_score + 1e-12 or (abs(macro_f1 - best_score) <= 1e-12 and val_loss < best_val_loss):
            best_score = macro_f1
            best_epoch = epoch
            best_state = _move_state_dict_to_cpu(model.state_dict())
            best_val_proba = fold_proba.astype(np.float32)
            best_val_loss = val_loss
            stale_epochs = 0
            epoch_iterator.set_postfix({"macro_f1": f"{macro_f1:.4f}", "best": f"{best_score:.4f}"})
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return {
        "arch": arch,
        "config": dict(config),
        "cardinalities": list(cardinalities),
        "num_classes": int(train_targets.shape[1]),
        "state_dict": best_state,
        "best_epoch": int(best_epoch),
        "best_macro_f1": float(best_score),
        "best_val_loss": float(best_val_loss),
        "eval_batch_size": eval_batch_size,
        "val_proba": best_val_proba,
    }
