from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import f1_score
from tqdm.auto import tqdm


DEFAULT_SEED = 20260504
ID_COL = "id"
TEXT_COL = "log_text"
SUBMISSION_COLUMNS = [
    "id",
    "has_anomaly",
    "primary_start_idx",
    "primary_end_idx",
    "primary_anomaly_type",
    "all_spans",
]
ANOMALY_TYPES = [
    "timeout_retry",
    "resource_exhaustion",
    "slow_burn_warning",
    "state_conflict",
    "parameter_drift",
    "out_of_order",
    "missing_step",
    "duplicate_event",
    "cross_component_mismatch",
    "partial_recovery_loop",
]
TYPE_TO_INDEX = {label: idx for idx, label in enumerate(ANOMALY_TYPES)}
BINARY_CLASSES = np.array([0, 1], dtype=np.int32)
TYPE_SPAN_LENGTHS = {
    "timeout_retry": {"min_len": 4, "target_len": 6, "max_len": 9},
    "resource_exhaustion": {"min_len": 4, "target_len": 6, "max_len": 10},
    "slow_burn_warning": {"min_len": 5, "target_len": 7, "max_len": 10},
    "state_conflict": {"min_len": 4, "target_len": 5, "max_len": 8},
    "parameter_drift": {"min_len": 4, "target_len": 5, "max_len": 8},
    "out_of_order": {"min_len": 3, "target_len": 4, "max_len": 7},
    "missing_step": {"min_len": 3, "target_len": 4, "max_len": 7},
    "duplicate_event": {"min_len": 3, "target_len": 4, "max_len": 7},
    "cross_component_mismatch": {"min_len": 4, "target_len": 5, "max_len": 8},
    "partial_recovery_loop": {"min_len": 5, "target_len": 6, "max_len": 9},
}

TEXT_HASH_BITS = 18
FEATURE_CONFIG = {
    "char_hash_features": 2**TEXT_HASH_BITS,
    "word_hash_features": 2**TEXT_HASH_BITS,
    "char_ngrams": [3, 5],
    "word_ngrams": [1, 2],
    "line_context_window": 2,
}


_SEG_RE = re.compile(r"\bseg_[A-Za-z0-9]+\b")
_ID_RE = re.compile(r"\bid_[A-Za-z0-9]+\b")
_NUM_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?![A-Za-z_])")
_WORD_RE = re.compile(r"[A-Za-z_]+")

CHAR_VECTORIZER = HashingVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
    dtype=np.float32,
)
WORD_VECTORIZER = HashingVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    n_features=2**18,
    alternate_sign=False,
    norm="l2",
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
    dtype=np.float32,
)


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str


@dataclass(frozen=True)
class Document:
    doc_id: str
    lines: tuple[str, ...]
    has_anomaly: int = 0
    primary_start_idx: int = -1
    primary_end_idx: int = -1
    primary_anomaly_type: str = "none"
    spans: tuple[Span, ...] = ()


@dataclass(frozen=True)
class Prediction:
    doc_id: str
    has_anomaly: int
    primary_start_idx: int
    primary_end_idx: int
    primary_anomaly_type: str
    all_spans: str


def append_action_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {message}\n")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def parse_spans(value: str | None) -> tuple[Span, ...]:
    if not value:
        return ()
    spans: list[Span] = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        parts = item.split("|")
        if len(parts) != 3:
            raise ValueError(f"Bad span item: {item!r}")
        start, end, label = parts
        if label not in TYPE_TO_INDEX:
            raise ValueError(f"Unknown anomaly type in span: {label!r}")
        spans.append(Span(int(start), int(end), label))
    return tuple(spans)


def read_documents(path: Path, expect_labels: bool) -> list[Document]:
    docs: list[Document] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = {ID_COL, TEXT_COL} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for row in reader:
            text = row[TEXT_COL]
            lines = tuple(text.splitlines())
            if expect_labels:
                has_anomaly = int(row["has_anomaly"])
                spans = parse_spans(row.get("all_spans", ""))
                doc = Document(
                    doc_id=str(row[ID_COL]),
                    lines=lines,
                    has_anomaly=has_anomaly,
                    primary_start_idx=int(row["primary_start_idx"]),
                    primary_end_idx=int(row["primary_end_idx"]),
                    primary_anomaly_type=row["primary_anomaly_type"],
                    spans=spans,
                )
            else:
                doc = Document(doc_id=str(row[ID_COL]), lines=lines)
            docs.append(doc)
    return docs


def read_sample_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return next(reader)


def fast_timestamp(line: str) -> tuple[int | None, int, int, int]:
    if len(line) < 19:
        return None, 0, 0, 0
    try:
        if (
            line[4] != "-"
            or line[7] != "-"
            or line[10] != " "
            or line[13] != ":"
            or line[16] != ":"
        ):
            return None, 0, 0, 0
        year = int(line[0:4])
        month = int(line[5:7])
        day = int(line[8:10])
        hour = int(line[11:13])
        minute = int(line[14:16])
        second = int(line[17:19])
    except ValueError:
        return None, 0, 0, 0
    pseudo_day = year * 13 * 32 + month * 32 + day
    pseudo_seconds = ((pseudo_day * 24 + hour) * 60 + minute) * 60 + second
    return pseudo_seconds, hour, minute, second


def normalize_line(line: str) -> str:
    body = line[19:].lstrip() if fast_timestamp(line)[0] is not None else line
    body = _SEG_RE.sub("<seg>", body)
    body = _ID_RE.sub("<id>", body)
    body = _NUM_RE.sub("<num>", body)
    body = body.replace("<ADDR>", "<addr>").replace("<PATH>", "<path>")
    prefix = "<time> " if body is not line and len(line) >= 19 else ""
    return (prefix + body).lower()


def _safe_ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _clip_scale(value: float, limit: float) -> float:
    if limit <= 0:
        return 0.0
    return float(max(-limit, min(limit, value)) / limit)


def _flag(text: str, word: str) -> float:
    return 1.0 if word in text else 0.0


def parse_document(doc: Document) -> dict:
    n_lines = len(doc.lines)
    norm_lines: list[str] = []
    line_numeric: list[list[float]] = []
    prev_ts: int | None = None
    deltas: list[float] = []
    backward_count = 0
    timestamp_count = 0
    keyword_totals = {
        key: 0
        for key in [
            "info",
            "warn",
            "error",
            "retry",
            "timeout",
            "drift",
            "conflict",
            "duplicate",
            "missing",
            "resource",
            "exhaust",
            "slow",
            "recover",
            "partial",
            "order",
            "cross",
            "component",
            "mismatch",
        ]
    }
    seg_total = 0
    id_total = 0
    addr_total = 0
    lengths: list[int] = []

    for idx, raw in enumerate(doc.lines):
        norm = normalize_line(raw)
        norm_lines.append(norm)
        low = raw.lower()
        ts_value, hour, minute, second = fast_timestamp(raw)
        has_ts = 1.0 if ts_value is not None else 0.0
        timestamp_count += int(has_ts)
        delta = 0.0
        backward = 0.0
        if ts_value is not None:
            if prev_ts is not None:
                delta = float(ts_value - prev_ts)
                deltas.append(delta)
                if delta < 0:
                    backward = 1.0
                    backward_count += 1
            prev_ts = ts_value

        line_len = len(raw)
        lengths.append(line_len)
        word_count = len(_WORD_RE.findall(raw))
        digit_count = sum(ch.isdigit() for ch in raw)
        alpha_count = sum(ch.isalpha() for ch in raw)
        upper_count = sum(ch.isupper() for ch in raw)
        punct_count = sum((not ch.isalnum()) and (not ch.isspace()) for ch in raw)
        seg_count = low.count("seg_")
        id_count = low.count("id_")
        addr_count = low.count("<addr>")
        seg_total += seg_count
        id_total += id_count
        addr_total += addr_count

        flags = {key: _flag(low, key) for key in keyword_totals}
        for key, val in flags.items():
            keyword_totals[key] += int(val)

        denom = max(1, line_len)
        rel_pos = _safe_ratio(idx, max(1, n_lines - 1))
        rev_pos = _safe_ratio(n_lines - 1 - idx, max(1, n_lines - 1))
        row = [
            rel_pos,
            rel_pos * rel_pos,
            rev_pos,
            math.log1p(n_lines) / 6.0,
            math.log1p(idx + 1) / 5.0,
            math.log1p(max(1, n_lines - idx)) / 5.0,
            min(line_len, 500) / 500.0,
            min(word_count, 120) / 120.0,
            min(digit_count, 80) / 80.0,
            min(alpha_count, 300) / 300.0,
            _safe_ratio(upper_count, max(1, alpha_count)),
            _safe_ratio(punct_count, denom),
            has_ts,
            hour / 23.0 if has_ts else 0.0,
            minute / 59.0 if has_ts else 0.0,
            second / 59.0 if has_ts else 0.0,
            _clip_scale(delta, 7200.0),
            min(abs(delta), 7200.0) / 7200.0,
            backward,
            flags["info"],
            flags["warn"],
            flags["error"],
            flags["retry"],
            flags["timeout"],
            flags["drift"],
            flags["conflict"],
            flags["duplicate"],
            flags["missing"],
            flags["resource"],
            flags["exhaust"],
            flags["slow"],
            flags["recover"],
            flags["partial"],
            flags["order"],
            flags["cross"],
            flags["component"],
            flags["mismatch"],
            min(seg_count, 8) / 8.0,
            min(id_count, 5) / 5.0,
            min(addr_count, 5) / 5.0,
            # v1.2: typo and structure signals
            _safe_ratio(sum(1 for ch in raw if not (ch.isalnum() or ch.isspace() or ch in '.:-_/\\[](){}@#$,;\'\"')), denom),
            _safe_ratio(min(sum(1 for i in range(len(raw) - 2) if raw[i] == raw[i + 1] == raw[i + 2]), 20), 20.0),
            _safe_ratio(sum(1 for i in range(len(raw) - 1) if raw[i].isalpha() and raw[i + 1].isalpha() and raw[i].islower() != raw[i + 1].islower()), denom),
            _safe_ratio(abs(line_len - (lengths[-1] if idx > 0 else line_len)), denom),
        ]
        line_numeric.append(row)

    n = max(1, n_lines)
    length_array = np.array(lengths if lengths else [0], dtype=np.float32)
    delta_array = np.array(deltas if deltas else [0.0], dtype=np.float32)
    total_chars = float(sum(lengths))
    doc_numeric = [
        math.log1p(n_lines) / 6.0,
        math.log1p(total_chars) / 10.0,
        min(float(length_array.mean()), 500.0) / 500.0,
        min(float(length_array.std()), 250.0) / 250.0,
        min(float(length_array.max()), 700.0) / 700.0,
        min(float(length_array.min()), 250.0) / 250.0,
        _safe_ratio(timestamp_count, n),
        1.0 - _safe_ratio(timestamp_count, n),
        _safe_ratio(backward_count, max(1, len(deltas))),
        _clip_scale(float(delta_array.mean()), 7200.0),
        min(float(delta_array.std()), 7200.0) / 7200.0,
        min(float(np.abs(delta_array).max()), 7200.0) / 7200.0,
    ]
    for key in keyword_totals:
        doc_numeric.append(_safe_ratio(keyword_totals[key], n))
    doc_numeric.extend(
        [
            min(seg_total, n * 4) / max(1.0, n * 4.0),
            min(id_total, n * 2) / max(1.0, n * 2.0),
            min(addr_total, n * 2) / max(1.0, n * 2.0),
        ]
    )
    # v1.2: template diversity signals
    unique_templates = len(set(norm_lines))
    doc_numeric.extend(
        [
            min(unique_templates, 200) / 200.0,
            1.0 - _safe_ratio(unique_templates, n),
        ]
    )
    return {
        "doc": doc,
        "norm_lines": norm_lines,
        "line_numeric": line_numeric,
        "doc_numeric": doc_numeric,
    }


def parse_documents_batch(docs: Sequence[Document]) -> list[dict]:
    return [parse_document(doc) for doc in docs]


def vectorize_text_and_numeric(texts: Sequence[str], numeric_rows: Sequence[Sequence[float]]) -> sp.csr_matrix:
    text_char = CHAR_VECTORIZER.transform(texts)
    text_word = WORD_VECTORIZER.transform(texts)
    numeric = sp.csr_matrix(np.asarray(numeric_rows, dtype=np.float32))
    return sp.hstack([text_char, text_word, numeric], format="csr", dtype=np.float32)


def make_doc_feature_matrix(parsed_docs: Sequence[dict], y_override: np.ndarray | None = None) -> tuple[sp.csr_matrix, np.ndarray]:
    texts = [" <nl> ".join(item["norm_lines"]) for item in parsed_docs]
    numeric = [item["doc_numeric"] for item in parsed_docs]
    if y_override is not None:
        y = y_override
    else:
        y = np.array([item["doc"].has_anomaly for item in parsed_docs], dtype=np.int32)
    return vectorize_text_and_numeric(texts, numeric), y


def make_line_context(norm_lines: Sequence[str], idx: int, window: int = 2) -> str:
    pieces: list[str] = []
    for offset in range(-window, window + 1):
        pos = idx + offset
        tag = f"ctx{offset:+d}"
        if pos < 0:
            pieces.append(f"{tag} <bos>")
        elif pos >= len(norm_lines):
            pieces.append(f"{tag} <eos>")
        else:
            pieces.append(f"{tag} {norm_lines[pos]}")
    return " || ".join(pieces)


def line_targets_for_doc(doc: Document, n_lines: int) -> dict[str, np.ndarray]:
    targets = {label: np.zeros(n_lines, dtype=np.int32) for label in ANOMALY_TYPES}
    for span in doc.spans:
        if span.label not in targets:
            continue
        start = max(0, min(n_lines - 1, span.start)) if n_lines else 0
        end = max(0, min(n_lines - 1, span.end)) if n_lines else -1
        if start <= end:
            targets[span.label][start : end + 1] = 1
    return targets


def make_line_feature_matrix(
    parsed_docs: Sequence[dict],
    include_targets: bool,
    line_contexts: list[list[str]] | None = None,
    doc_targets_override: dict[str, list[np.ndarray]] | None = None,
) -> tuple[sp.csr_matrix, dict[str, np.ndarray] | None, list[int]]:
    texts: list[str] = []
    numeric: list[list[float]] = []
    lengths: list[int] = []
    targets: dict[str, list[np.ndarray]] | None = {label: [] for label in ANOMALY_TYPES} if include_targets else None

    for i, item in enumerate(parsed_docs):
        doc = item.get("doc")  # may be None for cached items
        norm_lines = item["norm_lines"]
        line_numeric = item["line_numeric"]
        n_lines = len(norm_lines)
        lengths.append(n_lines)
        if line_contexts is not None:
            texts.extend(line_contexts[i])
        else:
            for idx in range(n_lines):
                texts.append(make_line_context(norm_lines, idx))
        numeric.extend(line_numeric)
        if include_targets and targets is not None:
            if doc_targets_override is not None:
                for label in ANOMALY_TYPES:
                    targets[label].append(doc_targets_override[label][i])
            elif doc is not None:
                doc_targets = line_targets_for_doc(doc, n_lines)
                for label in ANOMALY_TYPES:
                    targets[label].append(doc_targets[label])

    matrix = vectorize_text_and_numeric(texts, numeric) if texts else sp.csr_matrix((0, 2 ** (TEXT_HASH_BITS + 1)))
    if not include_targets or targets is None:
        return matrix, None, lengths
    flat_targets = {
        label: np.concatenate(parts).astype(np.int32) if parts else np.zeros(0, dtype=np.int32)
        for label, parts in targets.items()
    }
    return matrix, flat_targets, lengths


def batches(items: Sequence, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def class_weights_from_counts(pos: int, neg: int, cap: float) -> tuple[float, float]:
    if pos <= 0 or neg <= 0:
        return 1.0, 1.0
    pos_weight = max(1.0, math.sqrt(neg / pos))
    neg_weight = max(1.0, math.sqrt(pos / neg))
    return min(neg_weight, cap), min(pos_weight, cap)


def estimate_training_weights(docs: Sequence[Document]) -> tuple[tuple[float, float], dict[str, tuple[float, float]]]:
    doc_pos = sum(doc.has_anomaly for doc in docs)
    doc_neg = len(docs) - doc_pos
    doc_weights = class_weights_from_counts(doc_pos, doc_neg, cap=5.0)

    total_lines = sum(len(doc.lines) for doc in docs)
    type_pos = {label: 0 for label in ANOMALY_TYPES}
    for doc in docs:
        seen = {label: set() for label in ANOMALY_TYPES}
        n_lines = len(doc.lines)
        for span in doc.spans:
            if span.label not in seen or n_lines == 0:
                continue
            start = max(0, min(n_lines - 1, span.start))
            end = max(0, min(n_lines - 1, span.end))
            if start <= end:
                seen[span.label].update(range(start, end + 1))
        for label in ANOMALY_TYPES:
            type_pos[label] += len(seen[label])

    type_weights = {
        label: class_weights_from_counts(pos, total_lines - pos, cap=25.0)
        for label, pos in type_pos.items()
    }
    return doc_weights, type_weights


def make_sample_weight(y: np.ndarray, weights: tuple[float, float]) -> np.ndarray:
    neg_weight, pos_weight = weights
    return np.where(y == 1, pos_weight, neg_weight).astype(np.float32)


def probability_of_positive(model, matrix: sp.csr_matrix) -> np.ndarray:
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(matrix)[:, 1]
    else:
        logits = model.decision_function(matrix)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35)))
    return np.asarray(probs, dtype=np.float32)


def predict_raw_scores(
    docs: Sequence[Document],
    doc_model,
    type_models: dict[str, object],
    batch_size: int,
    desc: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    doc_probs: list[float] = []
    line_scores_by_doc: list[np.ndarray] = []
    for batch_docs in tqdm(
        list(batches(list(docs), batch_size)),
        desc=desc,
        unit="batch",
        dynamic_ncols=True,
    ):
        parsed = parse_documents_batch(batch_docs)
        doc_matrix, _ = make_doc_feature_matrix(parsed)
        batch_doc_probs = probability_of_positive(doc_model, doc_matrix)
        doc_probs.extend(batch_doc_probs.tolist())

        line_matrix, _, lengths = make_line_feature_matrix(parsed, include_targets=False)
        if line_matrix.shape[0] == 0:
            batch_line_probs = np.zeros((0, len(ANOMALY_TYPES)), dtype=np.float32)
        else:
            columns = [
                probability_of_positive(type_models[label], line_matrix)
                for label in ANOMALY_TYPES
            ]
            batch_line_probs = np.column_stack(columns).astype(np.float32)
        offset = 0
        for length in lengths:
            line_scores_by_doc.append(batch_line_probs[offset : offset + length, :])
            offset += length
    return np.asarray(doc_probs, dtype=np.float32), line_scores_by_doc


def smooth_scores(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float32, copy=False)
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float32), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def fill_mask_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or mask.size == 0:
        return mask
    mask = mask.copy()
    n = len(mask)
    idx = 0
    while idx < n:
        if mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and not mask[idx]:
            idx += 1
        gap_len = idx - start
        if start > 0 and idx < n and gap_len <= max_gap:
            mask[start:idx] = True
    return mask


def mask_components(mask: np.ndarray) -> list[tuple[int, int]]:
    comps: list[tuple[int, int]] = []
    idx = 0
    n = len(mask)
    while idx < n:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and mask[idx]:
            idx += 1
        comps.append((start, idx - 1))
    return comps


def _span_length_profile(label: str, default_min_len: int, default_max_len: int, n_lines: int) -> tuple[int, int, int]:
    cfg = TYPE_SPAN_LENGTHS.get(label, {})
    min_len = max(1, min(int(cfg.get("min_len", default_min_len)), n_lines))
    max_len = max(min_len, min(int(cfg.get("max_len", default_max_len)), n_lines))
    target_len = int(cfg.get("target_len", max(min_len, min(max_len, default_min_len))))
    target_len = max(min_len, min(max_len, target_len))
    return min_len, max_len, target_len


def _best_subspan_in_range(
    line_scores: np.ndarray,
    start: int,
    end: int,
    min_len: int,
    max_len: int,
    target_len: int,
) -> tuple[int, int, float] | None:
    best: tuple[int, int, float] | None = None
    for s in range(start, end + 1):
        running = 0.0
        for e in range(s, min(end, s + max_len - 1) + 1):
            running += float(line_scores[e])
            length = e - s + 1
            if length < min_len:
                continue
            score = running / float(length)
            if target_len > 0:
                score -= 0.01 * abs(length - target_len) / float(target_len)
            if best is None or score > best[2] or (
                abs(score - best[2]) <= 1e-12 and (length > best[1] - best[0] + 1 or s < best[0])
            ):
                best = (s, e, score)
    return best


def best_fallback_span(
    line_scores: np.ndarray,
    smoothing_window: int,
    min_span_len: int,
    max_span_len: int,
    type_span_lengths: dict[str, dict[str, int]] | None = None,
) -> tuple[int, int, str, float] | None:
    n_lines = line_scores.shape[0]
    if n_lines == 0:
        return None
    best: tuple[int, int, str, float] | None = None
    for type_idx, label in enumerate(ANOMALY_TYPES):
        cfg = (type_span_lengths or {}).get(label, {})
        label_min, label_max, target_len = _span_length_profile(label, min_span_len, max_span_len, n_lines)
        if "min_len" in cfg:
            label_min = max(1, min(int(cfg["min_len"]), n_lines))
        if "max_len" in cfg:
            label_max = max(label_min, min(int(cfg["max_len"]), n_lines))
        if "target_len" in cfg:
            target_len = max(label_min, min(label_max, int(cfg["target_len"])))
        seq = smooth_scores(line_scores[:, type_idx], smoothing_window)
        candidate = _best_subspan_in_range(seq, 0, n_lines - 1, label_min, label_max, target_len)
        if candidate is None:
            continue
        s, e, score = candidate
        if best is None or score > best[3] or (
            abs(score - best[3]) <= 1e-12 and (s < best[0] or (s == best[0] and e < best[1]))
        ):
            best = (s, e, label, score)
    return best


def decode_one_prediction(
    doc: Document,
    doc_prob: float,
    line_scores: np.ndarray,
    thresholds: dict,
) -> Prediction:
    doc_threshold = float(thresholds["doc_threshold"])
    type_thresholds = thresholds["type_thresholds"]
    smoothing_window = int(thresholds.get("smoothing_window", 3))
    gap_merge = int(thresholds.get("gap_merge", 1))
    min_span_len = int(thresholds.get("min_span_len", 2))
    max_spans = int(thresholds.get("max_spans", 2))
    max_span_len = int(thresholds.get("max_span_len", 12))
    type_span_lengths = thresholds.get("type_span_lengths", {})
    n_lines = len(doc.lines)
    raw_spans: list[tuple[int, int, str, float]] = []

    for type_idx, label in enumerate(ANOMALY_TYPES):
        if n_lines == 0:
            continue
        scores = smooth_scores(line_scores[:, type_idx], smoothing_window)
        mask = scores >= float(type_thresholds[label])
        mask = fill_mask_gaps(mask, gap_merge)
        label_min, label_max, target_len = _span_length_profile(label, min_span_len, max_span_len, n_lines)
        for start, end in mask_components(mask):
            candidate = _best_subspan_in_range(scores, start, end, label_min, label_max, target_len)
            if candidate is None:
                continue
            s, e, score = candidate
            raw_spans.append((s, e, label, float(score)))

    dedup: dict[tuple[int, int], tuple[int, int, str, float]] = {}
    for span in raw_spans:
        key = (span[0], span[1])
        if key not in dedup or span[3] > dedup[key][3]:
            dedup[key] = span
    spans = list(dedup.values())

    if not spans and doc_prob >= doc_threshold:
        fallback = best_fallback_span(
            line_scores,
            smoothing_window,
            min_span_len,
            max_span_len,
            type_span_lengths=type_span_lengths,
        )
        if fallback is not None:
            spans = [fallback]

    if not spans and doc_prob < doc_threshold:
        return Prediction(str(doc.doc_id), 0, -1, -1, "none", "")

    spans = sorted(spans, key=lambda item: (item[0], -item[3], item[1], item[2]))[:max_spans]
    if not spans:
        return Prediction(str(doc.doc_id), 0, -1, -1, "none", "")
    primary = spans[0]
    span_text = ";".join(f"{s}|{e}|{label}" for s, e, label, _ in spans)
    return Prediction(
        doc_id=str(doc.doc_id),
        has_anomaly=1,
        primary_start_idx=int(primary[0]),
        primary_end_idx=int(primary[1]),
        primary_anomaly_type=primary[2],
        all_spans=span_text,
    )


def decode_predictions(
    docs: Sequence[Document],
    doc_probs: np.ndarray,
    line_scores_by_doc: Sequence[np.ndarray],
    thresholds: dict,
) -> list[Prediction]:
    return [
        decode_one_prediction(doc, float(doc_prob), line_scores, thresholds)
        for doc, doc_prob, line_scores in zip(docs, doc_probs, line_scores_by_doc)
    ]


def write_submission(path: Path, predictions: Sequence[Prediction]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_COLUMNS)
        writer.writeheader()
        for pred in predictions:
            writer.writerow(
                {
                    "id": pred.doc_id,
                    "has_anomaly": pred.has_anomaly,
                    "primary_start_idx": pred.primary_start_idx,
                    "primary_end_idx": pred.primary_end_idx,
                    "primary_anomaly_type": pred.primary_anomaly_type,
                    "all_spans": pred.all_spans,
                }
            )


def read_submission(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def validate_submission_file(path: Path, test_docs: Sequence[Document], expected_columns: Sequence[str]) -> dict:
    columns, rows = read_submission(path)
    if list(columns) != list(expected_columns):
        raise ValueError(f"submission columns mismatch: got {columns}, expected {list(expected_columns)}")
    if len(rows) != len(test_docs):
        raise ValueError(f"submission row count {len(rows)} != test row count {len(test_docs)}")
    for idx, (row, doc) in enumerate(zip(rows, test_docs)):
        if str(row["id"]) != str(doc.doc_id):
            raise ValueError(f"id mismatch at row {idx}: got {row['id']!r}, expected {doc.doc_id!r}")
        if row["has_anomaly"] not in {"0", "1"}:
            raise ValueError(f"bad has_anomaly at id={doc.doc_id}: {row['has_anomaly']!r}")
        has_anomaly = int(row["has_anomaly"])
        try:
            start = int(row["primary_start_idx"])
            end = int(row["primary_end_idx"])
        except ValueError as exc:
            raise ValueError(f"bad primary indices at id={doc.doc_id}") from exc
        label = row["primary_anomaly_type"]
        all_spans = row.get("all_spans", "")
        if has_anomaly == 0:
            if start != -1 or end != -1 or label != "none" or all_spans not in {"", None}:
                raise ValueError(f"normal row has non-empty anomaly fields at id={doc.doc_id}")
            continue
        if start < 0 or end < start or end >= len(doc.lines):
            raise ValueError(f"primary span out of bounds at id={doc.doc_id}: {start}|{end}")
        if label not in TYPE_TO_INDEX:
            raise ValueError(f"bad primary type at id={doc.doc_id}: {label!r}")
        if not all_spans:
            raise ValueError(f"anomaly row has empty all_spans at id={doc.doc_id}")
        parsed = parse_spans(all_spans)
        if not parsed:
            raise ValueError(f"anomaly row has unparsable all_spans at id={doc.doc_id}")
        first = parsed[0]
        if (first.start, first.end, first.label) != (start, end, label):
            raise ValueError(f"primary fields do not match first all_spans item at id={doc.doc_id}")
        for span in parsed:
            if span.start < 0 or span.end < span.start or span.end >= len(doc.lines):
                raise ValueError(f"span out of bounds at id={doc.doc_id}: {span}")
    return {"rows": len(rows), "columns": columns}


def span_iou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    if a_start < 0 or b_start < 0 or a_end < a_start or b_end < b_start:
        return 0.0
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    if inter == 0:
        return 0.0
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return float(inter / union)


def evaluate_predictions(docs: Sequence[Document], predictions: Sequence[Prediction]) -> dict:
    y_true = np.array([doc.has_anomaly for doc in docs], dtype=np.int32)
    y_pred = np.array([pred.has_anomaly for pred in predictions], dtype=np.int32)
    detect_f1 = float(f1_score(y_true, y_pred, labels=[0, 1], average="macro", zero_division=0))

    ious: list[float] = []
    true_types: list[str] = []
    pred_types: list[str] = []
    for doc, pred in zip(docs, predictions):
        if doc.has_anomaly == 1 and pred.has_anomaly == 1:
            ious.append(
                span_iou(
                    pred.primary_start_idx,
                    pred.primary_end_idx,
                    doc.primary_start_idx,
                    doc.primary_end_idx,
                )
            )
            true_types.append(doc.primary_anomaly_type)
            pred_types.append(pred.primary_anomaly_type)

    loc_iou = float(np.mean(ious)) if ious else 0.0
    type_f1 = (
        float(f1_score(true_types, pred_types, labels=ANOMALY_TYPES, average="macro", zero_division=0))
        if true_types
        else 0.0
    )
    score = 0.15 * detect_f1 + 0.50 * loc_iou + 0.35 * type_f1
    return {
        "score": float(score),
        "f1_detect": detect_f1,
        "iou_loc": loc_iou,
        "f1_type": type_f1,
        "eligible_anomaly_rows": len(ious),
    }


def best_threshold_for_f1(
    y_true: np.ndarray,
    probs: np.ndarray,
    average: str = "binary",
) -> tuple[float, float]:
    if y_true.size == 0:
        return 0.5, 0.0
    candidates = list(np.linspace(0.05, 0.95, 19))
    finite_probs = probs[np.isfinite(probs)]
    if finite_probs.size:
        candidates.extend(np.quantile(finite_probs, [0.5, 0.75, 0.9, 0.95, 0.98, 0.99]).tolist())
    candidates = sorted({float(min(0.98, max(0.02, x))) for x in candidates})
    best_thr = 0.5
    best_score = -1.0
    labels = [0, 1] if average == "macro" else None
    for threshold in candidates:
        pred = (probs >= threshold).astype(np.int32)
        score = float(f1_score(y_true, pred, labels=labels, average=average, zero_division=0))
        if score > best_score:
            best_score = score
            best_thr = float(threshold)
    return best_thr, best_score


def flatten_line_labels_by_type(docs: Sequence[Document]) -> dict[str, np.ndarray]:
    parts = {label: [] for label in ANOMALY_TYPES}
    for doc in docs:
        targets = line_targets_for_doc(doc, len(doc.lines))
        for label in ANOMALY_TYPES:
            parts[label].append(targets[label])
    return {
        label: np.concatenate(chunks).astype(np.int32) if chunks else np.zeros(0, dtype=np.int32)
        for label, chunks in parts.items()
    }


def flatten_line_scores_for_type(line_scores_by_doc: Sequence[np.ndarray], type_idx: int) -> np.ndarray:
    chunks = [scores[:, type_idx] for scores in line_scores_by_doc if scores.size]
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
