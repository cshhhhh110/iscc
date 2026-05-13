"""Per-CWE byte n-gram dictionary features for vulnerability classification (v3.0).

Builds class-specific byte n-gram dictionaries from training data (TF-IDF top-K),
then extracts match features per-file.  Analogous to the "keyword dictionary"
approach proven effective on the system-log challenge (0.95).
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np

from dataset import binary_path, read_csv_rows

NGRAM_SIZES = (2, 3, 4)
TOP_K_PER_CLASS = 50
MIN_NGRAM_FREQ = 3  # minimum within-class frequency to be considered
FEATURES_PER_CLASS = 3  # count, ratio, unique


def _file_to_ngram_set(raw: bytes) -> Set[str]:
    """Extract all byte n-grams (sizes 2,3,4) from raw bytes into a hex set."""
    ngrams: Set[str] = set()
    n = len(raw)
    for k in NGRAM_SIZES:
        if n < k:
            continue
        for i in range(n - k + 1):
            ngrams.add(raw[i:i + k].hex())
    return ngrams


def build_cwe_ngram_dict(
    train_csv: Path,
    binaries_dir: Path,
    output_path: Path,
    top_k: int = TOP_K_PER_CLASS,
) -> Dict[str, List[str]]:
    """Build per-CWE byte n-gram dictionaries from training data using TF-IDF.

    Returns dict: {cwe_id: [hex_ngram_str, ...]} saved to output_path as JSON.
    """
    rows = read_csv_rows(train_csv)
    # Only use label=1 samples for dictionary building
    pos_rows = [r for r in rows if r.get("label") == "1" and r.get("cwe_id")]
    print(f"Building ngram dict from {len(pos_rows)} positive samples ({len(rows)} total)")

    # Group by CWE
    cwe_binaries: Dict[str, List[str]] = {}
    for r in pos_rows:
        cwe_id = r["cwe_id"]
        cwe_binaries.setdefault(cwe_id, []).append(r["binary_id"])

    cwe_classes = sorted(cwe_binaries.keys())
    print(f"CWE classes: {len(cwe_classes)}")

    # Collect per-class ngram frequencies
    cwe_ngram_counts: Dict[str, Counter] = {}
    for cwe_id in cwe_classes:
        counts: Counter = Counter()
        for bid in cwe_binaries[cwe_id]:
            path = binary_path(binaries_dir, bid)
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            ngram_set = _file_to_ngram_set(raw)
            counts.update(ngram_set)
        # Filter low-frequency
        cwe_ngram_counts[cwe_id] = Counter({k: v for k, v in counts.items() if v >= MIN_NGRAM_FREQ})
        print(f"  {cwe_id}: {len(cwe_binaries[cwe_id])} files, "
              f"{len(cwe_ngram_counts[cwe_id])} unique ngrams (freq>={MIN_NGRAM_FREQ})")

    # Compute TF-IDF
    num_classes = len(cwe_classes)
    # Document frequency: how many classes contain each ngram
    df: Counter = Counter()
    for cwe_id in cwe_classes:
        df.update(cwe_ngram_counts[cwe_id].keys())

    cwe_ngram_dict: Dict[str, List[str]] = {}
    for cwe_id in cwe_classes:
        class_total = sum(cwe_ngram_counts[cwe_id].values())
        if class_total == 0:
            cwe_ngram_dict[cwe_id] = []
            continue
        tfidf_scores: List[Tuple[str, float]] = []
        for ngram, freq in cwe_ngram_counts[cwe_id].items():
            tf = freq / class_total
            idf = np.log(num_classes / (1 + df[ngram]))
            tfidf_scores.append((ngram, tf * idf))
        # Top-K by TF-IDF
        tfidf_scores.sort(key=lambda x: -x[1])
        cwe_ngram_dict[cwe_id] = [ngram for ngram, _ in tfidf_scores[:top_k]]
        top_tfidf = tfidf_scores[0][1] if tfidf_scores else 0.0
        print(f"  {cwe_id}: kept {len(cwe_ngram_dict[cwe_id])}/{len(tfidf_scores)} ngrams, "
              f"top TF-IDF={top_tfidf:.4f}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(cwe_ngram_dict, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Saved ngram dict to {output_path} ({len(cwe_ngram_dict)} classes)")
    return cwe_ngram_dict


def load_cwe_ngram_dict(dict_path: Path) -> Dict[str, List[str]]:
    """Load a previously built CWE n-gram dictionary from JSON."""
    with dict_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_cwe_ngram_features(
    binary_path: Path,
    cwe_ngram_dict: Dict[str, List[str]],
) -> Dict[str, float]:
    """Extract per-CWE n-gram match features for a single binary.

    Returns a dict with keys like:
      cwe_ngram_count_{cwe_id}  — total ngram hits
      cwe_ngram_ratio_{cwe_id}  — hits / dictionary size
      cwe_ngram_unique_{cwe_id} — unique ngrams matched
    """
    feats: Dict[str, float] = {}

    # Initialize all features to zero
    for cwe_id in cwe_ngram_dict:
        feats[f"cwe_ngram_count_{cwe_id}"] = 0.0
        feats[f"cwe_ngram_ratio_{cwe_id}"] = 0.0
        feats[f"cwe_ngram_unique_{cwe_id}"] = 0.0

    # Read file and extract ngrams
    try:
        raw = binary_path.read_bytes()
    except OSError:
        return feats

    file_ngrams = _file_to_ngram_set(raw)

    # Match against each CWE dictionary
    for cwe_id, dict_ngrams in cwe_ngram_dict.items():
        dict_set = set(dict_ngrams)
        matched = file_ngrams & dict_set
        hit_count = len(matched)  # total overlap
        dict_len = len(dict_set)
        feats[f"cwe_ngram_count_{cwe_id}"] = float(hit_count)
        feats[f"cwe_ngram_ratio_{cwe_id}"] = hit_count / max(dict_len, 1)
        # Unique is same as count here since each ngram appears at most once in the file set
        feats[f"cwe_ngram_unique_{cwe_id}"] = float(hit_count)
    return feats


def get_cwe_ngram_feature_names(cwe_classes: List[str]) -> List[str]:
    """Return ordered feature column names for n-gram features."""
    names: List[str] = []
    for cwe_id in cwe_classes:
        names.append(f"cwe_ngram_count_{cwe_id}")
        names.append(f"cwe_ngram_ratio_{cwe_id}")
        names.append(f"cwe_ngram_unique_{cwe_id}")
    return names


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build or inspect CWE n-gram dictionaries")
    parser.add_argument("--build", action="store_true", help="Build dictionary from training data")
    parser.add_argument("--train-csv", default="train.csv", help="Path to training CSV")
    parser.add_argument("--binaries-dir", default="binaries", help="Path to binaries directory")
    parser.add_argument("--output", default="模型/cwe_ngram_dict_v3.0.json",
                        help="Output path for dictionary JSON")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_CLASS,
                        help="Number of ngrams per class")
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
