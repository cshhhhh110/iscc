"""BiLSTM with boundary-focused prediction (start/end/type heads).

Replaces 11-way per-line classification with:
  - start_head: per-line binary (is this line a span START?)
  - end_head:   per-line binary (is this line a span END?)
  - type_head:  per-line 11-way (only active within spans)

Decoder is threshold-based and searchable — boundaries can be tuned post-hoc.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_LABELS = 11
O_LABEL = 0

TYPE_TO_LABEL = {
    "timeout_retry": 1, "resource_exhaustion": 2, "slow_burn_warning": 3,
    "state_conflict": 4, "parameter_drift": 5, "out_of_order": 6,
    "missing_step": 7, "duplicate_event": 8, "cross_component_mismatch": 9,
    "partial_recovery_loop": 10,
}
LABEL_TO_TYPE = {v: k for k, v in TYPE_TO_LABEL.items()}


class LogBiLSTMv3(nn.Module):
    """BiLSTM with start/end/type boundary heads."""

    def __init__(
        self, input_dim: int = 300, hidden_dim: int = 256,
        num_labels: int = NUM_LABELS, dropout: float = 0.35,
        num_lstm_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        self.embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers=num_lstm_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0.0,
        )

        lstm_out = hidden_dim * 2

        # Boundary head: start + end logits per line
        self.boundary_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

        # Type head: per-line class logits
        self.type_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

        # Doc-level
        self.attn = nn.Linear(lstm_out, 1)
        self.doc_head = nn.Sequential(
            nn.Linear(lstm_out, hidden_dim // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if "weight" in name and param.ndim >= 2:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            elif "bias" in name:
                nn.init.zeros_(param)
        # Bias type head toward O
        self.type_head[-1].bias.data[O_LABEL] = 1.0

    def forward(self, features: torch.Tensor, mask: torch.Tensor):
        """Returns (boundary_logits, type_logits, doc_logits)."""
        B, T, D = features.shape
        x = self.embed(features)

        lengths = mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, total_length=T)

        boundary_logits = self.boundary_head(lstm_out)  # (B, T, 2)
        type_logits = self.type_head(lstm_out)           # (B, T, 11)

        # Doc via attention pooling
        attn_scores = self.attn(lstm_out).masked_fill(~mask.unsqueeze(-1), -1e10)
        attn_weights = F.softmax(attn_scores, dim=1)
        doc_vec = (lstm_out * attn_weights).sum(dim=1)
        doc_logits = self.doc_head(doc_vec)

        return boundary_logits, type_logits, doc_logits


def collate_batch(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Pad variable-length documents into a batch."""
    features_list = [torch.from_numpy(item["features"]) for item in batch]
    lengths = torch.tensor([len(f) for f in features_list], dtype=torch.long)
    max_len = int(lengths.max())

    padded_features = torch.zeros(len(batch), max_len, features_list[0].shape[-1], dtype=torch.float32)
    mask = torch.zeros(len(batch), max_len, dtype=torch.bool)
    for i, f in enumerate(features_list):
        n = f.shape[0]
        padded_features[i, :n] = f
        mask[i, :n] = True

    result: dict[str, torch.Tensor] = {"features": padded_features, "mask": mask, "lengths": lengths}

    if "labels" in batch[0]:
        labels_list = [torch.from_numpy(item["labels"]) for item in batch]
        padded_labels = torch.zeros(len(batch), max_len, dtype=torch.long)
        for i, lab in enumerate(labels_list):
            padded_labels[i, :lab.shape[0]] = lab
        result["labels"] = padded_labels

    if "has_anomaly" in batch[0]:
        result["has_anomaly"] = torch.tensor([item["has_anomaly"] for item in batch], dtype=torch.float32)

    result["doc_ids"] = [item["doc_id"] for item in batch]
    return result


def _labels_to_targets(labels: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert per-line label tensor to start/end/type targets.

    labels: (B, T) — 0=O, 1-10=type
    Returns:
      start_target: (B, T) — 1 at span starts
      end_target:   (B, T) — 1 at span ends
      type_target:  (B, T) — class label within spans, -100 elsewhere
    """
    B, T = labels.shape
    start_target = torch.zeros(B, T, dtype=torch.float32, device=labels.device)
    end_target = torch.zeros(B, T, dtype=torch.float32, device=labels.device)
    type_target = torch.full((B, T), -100, dtype=torch.long, device=labels.device)

    for b in range(B):
        n = int(mask[b].sum())
        lab = labels[b, :n].cpu().numpy()
        i = 0
        while i < n:
            if lab[i] == O_LABEL:
                i += 1
                continue
            lid = lab[i]
            start = i
            while i < n and lab[i] == lid:
                i += 1
            end = i - 1
            start_target[b, start] = 1.0
            end_target[b, end] = 1.0
            type_target[b, start:end + 1] = int(lid)

    return start_target, end_target, type_target


def decode_boundary_spans(
    start_probs: np.ndarray,
    end_probs: np.ndarray,
    type_probs: np.ndarray,
    start_threshold: float = 0.5,
    end_threshold: float = 0.5,
    min_span_len: int = 2,
    max_span_len: int = 15,
    max_spans: int = 3,
    type_span_lengths: dict[str, dict] | None = None,
) -> list[tuple[int, int, str, float]]:
    """Decode start/end/type predictions into scored spans.

    Returns list of (start, end, type_name, score), sorted by score descending.
    """
    n = len(start_probs)
    if n == 0:
        return []

    # Find candidate starts (local maxima above threshold)
    start_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if start_probs[i] < start_threshold:
            continue
        left_ok = (i == 0) or (start_probs[i] >= start_probs[i - 1])
        right_ok = (i == n - 1) or (start_probs[i] > start_probs[i + 1])
        if left_ok and right_ok:
            start_mask[i] = True

    end_mask = np.zeros(n, dtype=bool)
    for i in range(n):
        if end_probs[i] < end_threshold:
            continue
        left_ok = (i == 0) or (end_probs[i] > end_probs[i - 1])
        right_ok = (i == n - 1) or (end_probs[i] >= end_probs[i + 1])
        if left_ok and right_ok:
            end_mask[i] = True

    start_positions = np.where(start_mask)[0]
    end_positions = np.where(end_mask)[0]

    if len(start_positions) == 0 or len(end_positions) == 0:
        return []

    # Generate candidate spans
    candidates: list[tuple[int, int, str, float]] = []
    for s in start_positions:
        # Find best end for this start
        best_for_start: tuple[int, int, str, float] | None = None
        for e in end_positions:
            if e < s:
                continue
            length = e - s + 1
            if length < min_span_len or length > max_span_len:
                continue

            # Type: majority vote within span
            span_type_idx = int(np.argmax(type_probs[s:e + 1].mean(axis=0)))
            if span_type_idx == O_LABEL:
                span_type_idx = int(np.argmax(type_probs[s:e + 1, 1:].mean(axis=0))) + 1

            type_conf = float(type_probs[s:e + 1, span_type_idx].mean())
            score = float(start_probs[s]) * float(end_probs[e]) * type_conf

            # Type-specific length penalty
            type_name = LABEL_TO_TYPE.get(span_type_idx, "none")
            if type_span_lengths and type_name in type_span_lengths:
                cfg = type_span_lengths[type_name]
                target_len = cfg.get("target_len", length)
                score -= 0.01 * abs(length - target_len) / max(1, target_len)

            if best_for_start is None or score > best_for_start[3]:
                best_for_start = (int(s), int(e), type_name, score)

        if best_for_start is not None:
            candidates.append(best_for_start)

    if not candidates:
        return []

    # Sort by score, NMS to remove overlaps
    candidates.sort(key=lambda x: -x[3])
    selected: list[tuple[int, int, str, float]] = []
    occupied = set()

    for s, e, t, score in candidates:
        span_positions = set(range(s, e + 1))
        if span_positions & occupied:
            continue
        selected.append((s, e, t, score))
        occupied |= span_positions
        if len(selected) >= max_spans:
            break

    return selected


def predictions_from_boundary(
    doc_ids: Sequence[str],
    start_probs_batch: np.ndarray,
    end_probs_batch: np.ndarray,
    type_probs_batch: np.ndarray,
    doc_probs: np.ndarray,
    mask: torch.Tensor,
    thresholds: dict | None = None,
) -> list:
    """Convert boundary predictions to Prediction-compatible dicts.

    thresholds can contain:
      - start_threshold, end_threshold, min_span_len, max_span_len, max_spans
      - type_span_lengths
    """
    if thresholds is None:
        thresholds = {}
    st = thresholds.get("start_threshold", 0.5)
    et = thresholds.get("end_threshold", 0.5)
    min_len = thresholds.get("min_span_len", 2)
    max_len = thresholds.get("max_span_len", 15)
    max_s = thresholds.get("max_spans", 3)
    tsl = thresholds.get("type_span_lengths", None)
    doc_thr = thresholds.get("doc_threshold", 0.5)

    results = []
    for i, doc_id in enumerate(doc_ids):
        nv = int(mask[i].sum())
        spans = decode_boundary_spans(
            start_probs_batch[i, :nv], end_probs_batch[i, :nv],
            type_probs_batch[i, :nv], st, et, min_len, max_len, max_s, tsl,
        )

        doc_p = float(doc_probs[i])
        if not spans and doc_p < doc_thr:
            results.append({"doc_id": str(doc_id), "has_anomaly": 0,
                           "primary_start_idx": -1, "primary_end_idx": -1,
                           "primary_anomaly_type": "none", "all_spans": ""})
        elif not spans:
            # Fallback: use type probabilities argmax as in v2
            type_preds = np.argmax(type_probs_batch[i, :nv], axis=-1)
            lbls = type_preds.astype(np.int64)
            spans = []
            j = 0
            while j < len(lbls):
                if lbls[j] == O_LABEL:
                    j += 1
                    continue
                lid = lbls[j]
                s = j
                while j < len(lbls) and lbls[j] == lid:
                    j += 1
                e = j - 1
                if lid in LABEL_TO_TYPE:
                    spans.append((int(s), int(e), LABEL_TO_TYPE[lid], 0.5))
            if not spans:
                results.append({"doc_id": str(doc_id), "has_anomaly": 0,
                               "primary_start_idx": -1, "primary_end_idx": -1,
                               "primary_anomaly_type": "none", "all_spans": ""})
                continue
            primary = spans[0]
            all_spans_str = ";".join(f"{s}|{e}|{t}" for s, e, t, _ in spans)
            results.append({"doc_id": str(doc_id), "has_anomaly": 1,
                           "primary_start_idx": primary[0], "primary_end_idx": primary[1],
                           "primary_anomaly_type": primary[2], "all_spans": all_spans_str})
        else:
            primary = spans[0]
            all_spans_str = ";".join(f"{s}|{e}|{t}" for s, e, t, _ in spans)
            results.append({"doc_id": str(doc_id), "has_anomaly": 1,
                           "primary_start_idx": primary[0], "primary_end_idx": primary[1],
                           "primary_anomaly_type": primary[2], "all_spans": all_spans_str})

    return results
