"""BiLSTM sequence labeling model for log anomaly detection.

Input: per-line dense features (SVD-reduced, ~300 dims)
Output: per-line class logits (11 classes: O + 10 anomaly types)
         + doc-level has_anomaly probability

Architecture:
  Linear(300→256) + LayerNorm → BiLSTM(2×256) → Dropout → Linear(512→11)
  Attention pooling → Linear(512→1) for doc-level prediction
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

# 11 labels: O + 10 anomaly types
NUM_LABELS = 11
O_LABEL = 0

TYPE_TO_LABEL = {
    "timeout_retry": 1,
    "resource_exhaustion": 2,
    "slow_burn_warning": 3,
    "state_conflict": 4,
    "parameter_drift": 5,
    "out_of_order": 6,
    "missing_step": 7,
    "duplicate_event": 8,
    "cross_component_mismatch": 9,
    "partial_recovery_loop": 10,
}
LABEL_TO_TYPE = {v: k for k, v in TYPE_TO_LABEL.items()}


class LogBiLSTM(nn.Module):
    """BiLSTM for log line sequence labeling + doc classification."""

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 256,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.35,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        lstm_out = hidden_dim * 2
        self.pre_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.line_head = nn.Linear(hidden_dim, num_labels)

        # Doc-level prediction via attention pooling
        self.attn = nn.Linear(lstm_out, 1)
        self.doc_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "bias" in name:
                nn.init.zeros_(param)
        # Initialize line_head bias toward O class
        nn.init.zeros_(self.line_head.bias)
        self.line_head.bias.data[O_LABEL] = 1.0

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """Forward pass.

        Args:
            features: (B, T, D) padded feature tensor
            mask: (B, T) boolean mask (True = real token)

        Returns:
            line_logits: (B, T, C) per-line class logits
            doc_logits: (B, 1) document-level logits
        """
        B, T, D = features.shape

        # Embed
        x = self.embed(features)  # (B, T, hidden_dim)

        # Pack for LSTM efficiency
        lengths = mask.sum(dim=1).cpu()
        if lengths.min() == 0:
            # Ensure at least one timestep
            lengths = torch.clamp(lengths, min=1)

        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths, batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=T)
        # lstm_out: (B, T, hidden_dim * 2)

        # Per-line predictions
        pre = self.pre_head(lstm_out)  # (B, T, hidden_dim)
        line_logits = self.line_head(pre)  # (B, T, num_labels)

        # Doc-level prediction via attention pooling
        attn_scores = self.attn(lstm_out)  # (B, T, 1)
        attn_scores = attn_scores.masked_fill(~mask.unsqueeze(-1), -6e4)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, T, 1)
        doc_vec = (lstm_out * attn_weights).sum(dim=1)  # (B, lstm_out)
        doc_logits = self.doc_head(doc_vec)  # (B, 1)

        return line_logits, doc_logits

    @torch.no_grad()
    def predict(self, features: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference: return line predictions, line probabilities, doc probability."""
        line_logits, doc_logits = self.forward(features, mask)
        line_probs = F.softmax(line_logits, dim=-1)
        line_preds = torch.argmax(line_logits, dim=-1) * mask.long()
        doc_prob = torch.sigmoid(doc_logits).squeeze(-1)
        return line_preds, line_probs, doc_prob


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate variable-length documents into a padded batch.

    Each item in batch is a dict with:
      - features: (n_lines, D) numpy array
      - labels: (n_lines,) numpy array (optional, for training)
      - has_anomaly: int (optional)
      - doc_id: str
    """
    features_list = [torch.from_numpy(item["features"]) for item in batch]
    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    max_len = int(lengths.max())

    # Pad features
    padded_features = torch.zeros(len(batch), max_len, features_list[0].shape[-1], dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        padded_features[i, :n] = f
        mask[i, :n] = True

    result: dict[str, torch.Tensor] = {
        "features": padded_features,
        "mask": mask,
        "lengths": lengths,
    }

    # Pad labels if available
    if "labels" in batch[0]:
        labels_list = [torch.from_numpy(item["labels"]) for item in batch]
        padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            padded_labels[i, :lab.shape[0]] = lab
        result["labels"] = padded_labels

    # Doc targets
    if "has_anomaly" in batch[0]:
        result["has_anomaly"] = torch.tensor([item["has_anomaly"] for item in batch], dtype=torch.float32)

    result["doc_ids"] = [item["doc_id"] for item in batch]
    return result


def _labels_to_spans(labels: np.ndarray) -> list[tuple[int, int, str]]:
    """Convert per-line label array to list of (start, end, type) spans."""
    n = len(labels)
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        if labels[i] == O_LABEL:
            i += 1
            continue
        label_id = labels[i]
        start = i
        while i < n and labels[i] == label_id:
            i += 1
        end = i - 1
        if label_id in LABEL_TO_TYPE:
            spans.append((int(start), int(end), LABEL_TO_TYPE[label_id]))
    return spans


def predictions_from_model_output(
    doc_ids: Sequence[str],
    line_preds: torch.Tensor,
    doc_probs: torch.Tensor,
    mask: torch.Tensor,
) -> list[dict]:
    """Convert model output to prediction dicts (compatible with common.Prediction format)."""
    results: list[dict] = []
    for i, doc_id in enumerate(doc_ids):
        n_valid = int(mask[i].sum())
        pred_labels = line_preds[i, :n_valid].cpu().numpy().astype(np.int64)
        doc_p = float(doc_probs[i])

        spans = _labels_to_spans(pred_labels)

        if not spans or doc_p < 0.5:
            results.append({
                "doc_id": str(doc_id),
                "has_anomaly": 0,
                "primary_start_idx": -1,
                "primary_end_idx": -1,
                "primary_anomaly_type": "none",
                "all_spans": "",
            })
        else:
            primary = spans[0]
            all_spans = ";".join(f"{s}|{e}|{t}" for s, e, t in spans)
            results.append({
                "doc_id": str(doc_id),
                "has_anomaly": 1,
                "primary_start_idx": primary[0],
                "primary_end_idx": primary[1],
                "primary_anomaly_type": primary[2],
                "all_spans": all_spans,
            })
    return results
