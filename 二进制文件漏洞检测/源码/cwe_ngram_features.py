"""Per-CWE byte n-gram dictionary features for vulnerability classification (v3.1).

Segmented by PE section (.text / .rdata / .idata / .data / other).
Each section independently tracked for per-CWE TF-IDF top-K n-grams.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pefile

from dataset import binary_path, read_csv_rows

NGRAM_SIZES = (2, 3, 4)
TOP_K_PER_CLASS = 50
MIN_NGRAM_FREQ = 3

SECTION_CLASSES = ["text", "rdata", "idata", "data", "other"]

# Map PE section names to classes
_SECTION_MAP = {
    ".text": "text", ".code": "text",
    ".rdata": "rdata", ".rodata": "rdata",
    ".idata": "idata",
    ".data": "data", ".sdata": "data", ".sbss": "data", ".bss": "data",
}


def _classify_section(name: str) -> str:
    """Classify a PE section name into one of the 5 section classes."""
    normalized = name.lower().strip()
    return _SECTION_MAP.get(normalized, "other")


def _parse_pe_sections(raw: bytes) -> Dict[str, bytes]:
    """Parse PE sections and return a dict mapping section class -> concatenated bytes.

    Returns empty dict if PE parsing fails.
    """
    result: Dict[str, bytes] = {cls: b"" for cls in SECTION_CLASSES}
    try:
        pe = pefile.PE(data=raw, fast_load=True)
    except Exception:
        # Can't parse PE — all bytes go to other
        result["other"] = raw
        return result

    for section in pe.sections:
        try:
            data = section.get_data()
        except Exception:
            continue
        cls = _classify_section(
            getattr(section, "Name", b"").decode("utf-8", errors="ignore").strip("\x00").strip()
        )
        result[cls] += data

    # If no sections mapped to a class, it stays empty string
    return result


def _bytes_to_ngram_set(data: bytes) -> Set[str]:
    """Extract all byte n-grams (sizes 2,3,4) from raw bytes into a hex set."""
    ngrams: Set[str] = set()
    n = len(data)
    for k in NGRAM_SIZES:
        if n < k:
            continue
        for i in range(n - k + 1):
            ngrams.add(data[i:i + k].hex())
    return ngrams


def _file_to_section_ngram_sets(raw: bytes) -> Dict[str, Set[str]]:
    """Extract per-section n-gram sets from a PE binary's raw bytes."""
    section_bytes = _parse_pe_sections(raw)
    return {cls: _bytes_to_ngram_set(section_bytes[cls]) for cls in SECTION_CLASSES}


def _build_section_dict(
    cwe_classes: List[str],
    cwe_binaries: Dict[str, List[str]],
    binaries_dir: Path,
    top_k: int,
) -> Dict[str, Dict[str, List[str]]]:
    """Build per-CWE, per-section n-gram dictionaries using TF-IDF.

    Returns: {cwe_id: {section_class: [hex_ngram_str, ...]}}
    """
    # Collect per-class, per-section ngram frequencies
    cwe_section_counts: Dict[str, Dict[str, Counter]] = {}
    for cwe_id in cwe_classes:
        section_counters: Dict[str, Counter] = {cls: Counter() for cls in SECTION_CLASSES}
        for bid in cwe_binaries[cwe_id]:
            path = binary_path(binaries_dir, bid)
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            section_ngrams = _file_to_section_ngram_sets(raw)
            for cls in SECTION_CLASSES:
                section_counters[cls].update(section_ngrams[cls])
        # Filter low-frequency
        for cls in SECTION_CLASSES:
            section_counters[cls] = Counter(
                {k: v for k, v in section_counters[cls].items() if v >= MIN_NGRAM_FREQ}
            )
        cwe_section_counts[cwe_id] = section_counters
        total_ngrams = sum(len(c[cls]) for cls in SECTION_CLASSES for c in [section_counters[cls]])
        print(f"  {cwe_id}: {len(cwe_binaries[cwe_id])} files, {total_ngrams} total unique ngrams (freq>={MIN_NGRAM_FREQ})")

    # TF-IDF per section
    num_classes = len(cwe_classes)
    cwe_ngram_dict: Dict[str, Dict[str, List[str]]] = {}

    for cls in SECTION_CLASSES:
        # Document frequency for this section
        df: Counter = Counter()
        for cwe_id in cwe_classes:
            df.update(cwe_section_counts[cwe_id][cls].keys())

        for cwe_id in cwe_classes:
            class_total = sum(cwe_section_counts[cwe_id][cls].values())
            if cwe_ngram_dict.get(cwe_id) is None:
                cwe_ngram_dict[cwe_id] = {}

            if class_total == 0:
                cwe_ngram_dict[cwe_id][cls] = []
                continue

            tfidf_scores: List[Tuple[str, float]] = []
            for ngram, freq in cwe_section_counts[cwe_id][cls].items():
                tf = freq / class_total
                idf = np.log(num_classes / (1 + df[ngram]))
                tfidf_scores.append((ngram, tf * idf))

            tfidf_scores.sort(key=lambda x: -x[1])
            cwe_ngram_dict[cwe_id][cls] = [ngram for ngram, _ in tfidf_scores[:top_k]]

    return cwe_ngram_dict


def build_cwe_ngram_dict(
    train_csv: Path,
    binaries_dir: Path,
    output_path: Path,
    top_k: int = TOP_K_PER_CLASS,
) -> Dict[str, Dict[str, List[str]]]:
    """Build per-CWE, per-section byte n-gram dictionaries from training data.

    Returns: {cwe_id: {section: [hex_ngram_str, ...]}}
    Saved to output_path as JSON.
    """
    rows = read_csv_rows(train_csv)
    pos_rows = [r for r in rows if r.get("label") == "1" and r.get("cwe_id")]
    print(f"Building segmented ngram dict from {len(pos_rows)} positive samples ({len(rows)} total)")

    cwe_binaries: Dict[str, List[str]] = {}
    for r in pos_rows:
        cwe_binaries.setdefault(r["cwe_id"], []).append(r["binary_id"])

    cwe_classes = sorted(cwe_binaries.keys())
    print(f"CWE classes: {len(cwe_classes)}, sections: {SECTION_CLASSES}")

    cwe_ngram_dict = _build_section_dict(cwe_classes, cwe_binaries, binaries_dir, top_k)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cwe_ngram_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved segmented ngram dict to {output_path} ({len(cwe_ngram_dict)} classes)")
    return cwe_ngram_dict


def build_cwe_ngram_dict_for_rows(
    rows: List[Dict[str, str]],
    binaries_dir: Path,
    top_k: int = TOP_K_PER_CLASS,
) -> Dict[str, Dict[str, List[str]]]:
    """Build per-CWE n-gram dictionary from explicit rows (for fold-safe OOF).

    Only uses rows with label=1 and non-empty cwe_id.
    Does NOT save to disk — caller decides.
    """
    pos_rows = [r for r in rows if r.get("label") == "1" and r.get("cwe_id")]
    cwe_binaries: Dict[str, List[str]] = {}
    for r in pos_rows:
        cwe_binaries.setdefault(r["cwe_id"], []).append(r["binary_id"])
    cwe_classes = sorted(cwe_binaries.keys())
    if not cwe_classes:
        return {}
    return _build_section_dict(cwe_classes, cwe_binaries, binaries_dir, top_k)


def load_cwe_ngram_dict(dict_path: Path) -> Dict[str, Dict[str, List[str]]]:
    """Load a previously built CWE n-gram dictionary from JSON."""
    with dict_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_cwe_ngram_features(
    binary_path: Path,
    cwe_ngram_dict: Dict[str, Dict[str, List[str]]],
) -> Dict[str, float]:
    """Extract per-section, per-CWE n-gram match features for a single binary.

    Returns: {cwe_ngram_{section}_{count|ratio|unique}_{cwe_id}: float}
    """
    feats: Dict[str, float] = {}
    for cwe_id in cwe_ngram_dict:
        for cls in SECTION_CLASSES:
            feats[f"cwe_ngram_{cls}_count_{cwe_id}"] = 0.0
            feats[f"cwe_ngram_{cls}_ratio_{cwe_id}"] = 0.0
            feats[f"cwe_ngram_{cls}_unique_{cwe_id}"] = 0.0

    try:
        raw = binary_path.read_bytes()
    except OSError:
        return feats

    section_ngrams = _file_to_section_ngram_sets(raw)

    for cwe_id, section_dicts in cwe_ngram_dict.items():
        for cls in SECTION_CLASSES:
            dict_ngrams = section_dicts.get(cls, [])
            if not dict_ngrams:
                continue
            dict_set = set(dict_ngrams)
            file_set = section_ngrams.get(cls, set())
            matched = file_set & dict_set
            hit_count = len(matched)
            dict_len = len(dict_set)
            feats[f"cwe_ngram_{cls}_count_{cwe_id}"] = float(hit_count)
            feats[f"cwe_ngram_{cls}_ratio_{cwe_id}"] = hit_count / max(dict_len, 1)
            feats[f"cwe_ngram_{cls}_unique_{cwe_id}"] = float(hit_count)
    return feats


def get_cwe_ngram_feature_names(cwe_classes: List[str]) -> List[str]:
    """Return ordered feature column names for segmented n-gram features."""
    names: List[str] = []
    for cwe_id in cwe_classes:
        for cls in SECTION_CLASSES:
            names.append(f"cwe_ngram_{cls}_count_{cwe_id}")
            names.append(f"cwe_ngram_{cls}_ratio_{cwe_id}")
            names.append(f"cwe_ngram_{cls}_unique_{cwe_id}")
    return names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build CWE segmented n-gram dictionaries (v3.1)")
    parser.add_argument("--build", action="store_true", help="Build dictionary from training data")
    parser.add_argument("--train-csv", default="data/train.csv", help="Path to training CSV")
    parser.add_argument("--binaries-dir", default="../binaries", help="Path to binaries directory")
    parser.add_argument("--output", default="models/cwe_ngram_dict_v3.1.json",
                        help="Output path for dictionary JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_CLASS,
                        help="Number of ngrams per class per section")
    args = parser.parse_args()

    if args.build:
        build_cwe_ngram_dict(
            train_csv=Path(args.train_csv),
            binaries_dir=Path(args.binaries_dir),
            output_path=Path(args.output),
            top_k=args.top_k,
        )
    else:
        print("Usage: python cwe_ngram_features.py --build [--train-csv ...] [--binaries-dir ...]")
