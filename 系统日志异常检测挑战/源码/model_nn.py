"""BiLSTM + CRF model for log anomaly span detection.

Input: per-line features (44 numeric + 256 text hash = 300 dims)
Output: per-line label (O=0, or one of 10 anomaly types=1..10)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import HashingVectorizer


# Lightweight text vectorizer for dense line features (much smaller than the 524K used by SGD)
LINE_TEXT_VECTORIZER = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    n_features=256,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
    dtype=np.float32,
)

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


class CRF(nn.Module):
    """Minimal linear-chain CRF for token-level sequence labeling."""

    def __init__(self, num_labels: int):
        super().__init__()
        self.num_labels = num_labels
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels) * 0.01)
        # Strongly discourage transitions to/from non-O anomalies
        self.transitions.data[:, O_LABEL] = -5.0
        self.transitions.data[O_LABEL, :] = -5.0
        self.transitions.data[O_LABEL, O_LABEL] = 2.0
        self.transitions.data[range(1, num_labels), range(1, num_labels)] = 2.0

    def _forward_alg(self, emissions, mask):
        """Compute the partition function log Z."""
        B, T, C = emissions.shape
        log_alpha = torch.full((B, C), -1e10, device=emissions.device)
        log_alpha[:, O_LABEL] = 0.0

        for t in range(T):
            emit = emissions[:, t, :]  # (B, C)
            # log_alpha: (B, C) → (B, C, 1) + trans: (C, C) → logsumexp over dim=1 → (B, C)
            log_alpha = torch.logsumexp(log_alpha.unsqueeze(-1) + self.transitions, dim=1) + emit
            m = mask[:, t].unsqueeze(-1)
            log_alpha = log_alpha * m + (1 - m) * log_alpha.detach()

        return torch.logsumexp(log_alpha, dim=-1)  # (B,)

    def _score_sentence(self, emissions, tags, mask):
        """Score a given tag sequence."""
        B, T = tags.shape
        score = emissions.new_zeros(B)
        b_idx = torch.arange(B, device=emissions.device)
        # Start transition from O
        score = score + self.transitions[O_LABEL, tags[:, 0]] * mask[:, 0]
        # Add first emission
        score = score + emissions[b_idx, 0, tags[:, 0]] * mask[:, 0]
        # Add remaining transitions + emissions
        t_flat = self.transitions.view(-1)
        for t in range(1, T):
            idx = tags[:, t - 1] * self.num_labels + tags[:, t]
            score = score + t_flat[idx] * mask[:, t]
            score = score + emissions[b_idx, t, tags[:, t]] * mask[:, t]
        return score

    def forward(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute NLL. emissions: (B, T, C), tags: (B, T), mask: (B, T)"""
        assert tags.max() < self.num_labels, f"tag max {tags.max()} >= {self.num_labels}"
        log_z = self._forward_alg(emissions, mask)
        gold_score = self._score_sentence(emissions, tags, mask)
        return (log_z - gold_score).mean()

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding. Returns best tag sequence (B, T)."""
        B, T, C = emissions.shape
        log_delta = torch.full((B, C), -1e10, device=emissions.device)
        log_delta[:, O_LABEL] = 0.0
        backpointers = torch.zeros(B, T, C, dtype=torch.long, device=emissions.device)

        for t in range(T):
            # (B, C, 1) + (C, C) → max over dim=1 → (B, C)
            max_vals, max_idx = (log_delta.unsqueeze(-1) + self.transitions).max(dim=1)
            log_delta = max_vals + emissions[:, t, :]
            m = mask[:, t].unsqueeze(-1)
            log_delta = log_delta * m + (1 - m) * log_delta.detach()
            backpointers[:, t, :] = max_idx

        best_tags = torch.zeros(B, T, dtype=torch.long, device=emissions.device)
        best_last = log_delta.max(dim=1).indices
        b_idx = torch.arange(B, device=emissions.device)
        best_tags[b_idx, T - 1] = best_last

        b_idx = torch.arange(B, device=emissions.device)
        for t in range(T - 2, -1, -1):
            best_last = backpointers[b_idx, t + 1, best_last]
            best_tags[:, t] = best_last

        return best_tags * mask.long()


class LogBiLSTMCRF(nn.Module):
    """BiLSTM + CRF for log line-level sequence labeling."""

    def __init__(
        self,
        input_dim: int = 300,
        hidden_dim: int = 128,
        num_labels: int = NUM_LABELS,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if 2 > 1 else 0,
        )
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_labels)

    def forward(self, features: torch.Tensor, mask: torch.Tensor, tags: torch.Tensor | None = None):
        """features: (B, T, D), mask: (B, T), tags: (B, T) or None"""
        lstm_out, _ = self.lstm(features)
        lstm_out = self.dropout(lstm_out)
        emissions = self.classifier(lstm_out)

        if tags is not None:
            loss = self.crf(emissions, tags, mask)
            return loss, emissions
        else:
            preds = self.crf.decode(emissions, mask)
            return preds, emissions


def build_line_features(norm_texts: list[str], line_numeric: list[list[float]]) -> np.ndarray:
    """Build dense feature vectors for a document's lines."""
    if not norm_texts:
        return np.zeros((0, 300), dtype=np.float32)

    text_feats = LINE_TEXT_VECTORIZER.transform(norm_texts).toarray().astype(np.float32)
    num_feats = np.asarray(line_numeric, dtype=np.float32)
    return np.concatenate([text_feats, num_feats], axis=1)


def labels_from_doc(doc, n_lines: int) -> np.ndarray:
    """Convert document spans to per-line labels."""
    labels = np.zeros(n_lines, dtype=np.int64)
    for span in doc.spans:
        if span.label not in TYPE_TO_LABEL:
            continue
        label_id = TYPE_TO_LABEL[span.label]
        start = max(0, min(n_lines - 1, span.start))
        end = max(0, min(n_lines - 1, span.end))
        if start <= end:
            labels[start: end + 1] = label_id
    return labels


def tags_to_prediction(doc, tags: np.ndarray) -> tuple[int, int, int, str, str]:
    """Convert per-line tag sequence to prediction fields.
    Returns (has_anomaly, primary_start_idx, primary_end_idx, primary_type, all_spans).
    """
    n = len(tags)
    # Find contiguous non-O segments
    spans: list[tuple[int, int, str]] = []
    i = 0
    while i < n:
        if tags[i] == O_LABEL:
            i += 1
            continue
        label = tags[i]
        start = i
        while i < n and tags[i] == label:
            i += 1
        end = i - 1
        if label in LABEL_TO_TYPE:
            spans.append((start, end, LABEL_TO_TYPE[label]))

    if not spans:
        return 0, -1, -1, "none", ""

    primary = spans[0]
    all_spans = ";".join(f"{s}|{e}|{t}" for s, e, t in spans)
    return 1, primary[0], primary[1], primary[2], all_spans
