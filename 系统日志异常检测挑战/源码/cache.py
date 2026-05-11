"""Feature cache: pre-compute document parsing and line context strings.

Usage:
    python cache.py --data-dir E:/赛题数据/系统日志异常检测挑战

Outputs:
    - 缓存/parsed_docs.joblib     — list of parsed doc dicts (norm_lines, line_numeric, doc_numeric)
    - 缓存/line_contexts.pkl       — list of lists of context strings per document
    - 缓存/doc_ids.joblib          — list of doc IDs (for alignment)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from common import make_line_context, parse_documents_batch, read_documents


def build_cache(data_dir: Path, train_file: str = "train.csv") -> None:
    cache_dir = data_dir / "缓存"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use cache names based on train file
    prefix = Path(train_file).stem  # "train" or "pseudo_train"
    parsed_path = cache_dir / f"{prefix}_parsed.joblib"
    contexts_path = cache_dir / f"{prefix}_contexts.pkl"
    ids_path = cache_dir / f"{prefix}_ids.joblib"

    print("Loading documents...")
    docs = read_documents(data_dir / train_file, expect_labels=True)
    print(f"  {len(docs)} documents")

    print("Parsing documents (normalizing text, extracting numeric features)...")
    parsed = parse_documents_batch(docs)
    print(f"  {len(parsed)} parsed")

    print("Building line context strings (this is the heavy part)...")
    line_contexts: list[list[str]] = []
    total_lines = 0
    for item in parsed:
        norm_lines = item["norm_lines"]
        contexts = [make_line_context(norm_lines, idx) for idx in range(len(norm_lines))]
        line_contexts.append(contexts)
        total_lines += len(contexts)

    print(f"  {total_lines} context strings built")

    # Remove Document objects from parsed (not serializable cleanly)
    parsed_clean = []
    for item in parsed:
        doc = item["doc"]
        parsed_clean.append({
            "has_anomaly": doc.has_anomaly,
            "num_lines": len(item["norm_lines"]),
            "norm_lines": item["norm_lines"],
            "line_numeric": item["line_numeric"],
            "doc_numeric": item["doc_numeric"],
        })

    print("Saving cache files...")
    joblib.dump(parsed_clean, parsed_path, compress=3)
    joblib.dump(line_contexts, contexts_path, compress=3)
    joblib.dump([item["doc"].doc_id for item in parsed], ids_path, compress=3)

    print(f"Cache built successfully in {cache_dir}")
    print(f"  {parsed_path.name}  — {len(parsed_clean)} documents")
    print(f"  {contexts_path.name}   — {len(line_contexts)} documents, {total_lines} lines")
    print(f"  {ids_path.name}      — ID alignment")


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Build feature cache for log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=default_root)
    parser.add_argument("--train-file", type=str, default="train.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_cache(args.data_dir.resolve(), args.train_file)
