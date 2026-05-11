"""Build dense feature representations via TruncatedSVD on sparse HashingVectorizer features.

Reduces 262K-dim char + 262K-dim word sparse features → 128 + 128 dense,
then concatenates with 44-dim numeric features → 300 dims per line.

Output (saved to 缓存/):
  - dense_train.joblib: list of dicts with doc_id, features(n_lines×300), labels, has_anomaly
  - dense_test.joblib: list of dicts with doc_id, features(n_lines×300)
  - svd_char.joblib: fitted TruncatedSVD for char features
  - svd_word.joblib: fitted TruncatedSVD for word features
  - scaler_numeric.joblib: fitted StandardScaler for numeric features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common import (
    ANOMALY_TYPES,
    CHAR_VECTORIZER,
    WORD_VECTORIZER,
    TYPE_TO_INDEX,
    parse_documents_batch,
    read_documents,
)

SEED = 20260504


def _build_labels_from_doc(doc, n_lines: int) -> np.ndarray:
    """Convert span annotations to per-line label array (0=O, 1-10=type index)."""
    labels = np.zeros(n_lines, dtype=np.int64)
    for span in doc.spans:
        if span.label not in TYPE_TO_INDEX:
            continue
        type_idx = TYPE_TO_INDEX[span.label] + 1  # 1..10
        start = max(0, min(n_lines - 1, span.start))
        end = max(0, min(n_lines - 1, span.end))
        if start <= end:
            labels[start: end + 1] = type_idx
    return labels


def _batch_transform(vectorizer, texts: list[str], batch_size: int = 50000) -> sp.csr_matrix:
    """Transform texts in batches to avoid memory spikes."""
    parts: list[sp.csr_matrix] = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"vectorizing ({len(texts)} texts)", unit="batch", dynamic_ncols=True):
        batch = texts[i: i + batch_size]
        parts.append(vectorizer.transform(batch))
    return sp.vstack(parts, format="csr")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build SVD-reduced dense features for log anomaly detection.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--train-file", type=str, default="train.csv")
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--svd-components", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    cache_dir = data_dir / "缓存"
    cache_dir.mkdir(parents=True, exist_ok=True)
    n_comp = args.svd_components

    print("=" * 60)
    print("Step 1/6: Loading documents...")
    train_docs = read_documents(data_dir / args.train_file, expect_labels=True)
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    print(f"  Train: {len(train_docs)} docs, Test: {len(test_docs)} docs")

    # Use parsed cache for train docs if available (much faster than re-parsing)
    train_parsed_path = cache_dir / "train_parsed.joblib"
    if train_parsed_path.exists():
        print("  Using cached train parsed data")
        train_parsed = joblib.load(train_parsed_path)
        # Verify order by checking doc_ids
        train_ids_path = cache_dir / "train_ids.joblib"
        if train_ids_path.exists():
            cached_ids = joblib.load(train_ids_path)
            for i, (cid, doc) in enumerate(zip(cached_ids, train_docs)):
                if str(cid) != str(doc.doc_id):
                    raise ValueError(f"Cache order mismatch at index {i}: cached={cid}, doc={doc.doc_id}")
            print(f"  Cache order verified: {len(cached_ids)} docs match")
    else:
        print("  Parsing train docs (no cache found)...")
        train_parsed = parse_documents_batch(train_docs)

    print("  Parsing test docs...")
    test_parsed = parse_documents_batch(test_docs)

    print("\nStep 2/6: Collecting line texts and numeric features...")
    train_n_lines = sum(len(item["norm_lines"]) for item in train_parsed)
    test_n_lines = sum(len(item["norm_lines"]) for item in test_parsed)
    total_lines = train_n_lines + test_n_lines
    print(f"  Train lines: {train_n_lines}, Test lines: {test_n_lines}, Total: {total_lines}")

    # Collect all line texts (for vectorization) and numeric features
    all_texts: list[str] = []
    all_numeric: list[list[float]] = []

    # Per-document metadata
    train_features_meta: list[dict] = []
    test_features_meta: list[dict] = []

    # Process train docs
    for i, (doc, parsed) in enumerate(tqdm(zip(train_docs, train_parsed), total=len(train_docs), desc="processing train docs", unit="doc", dynamic_ncols=True)):
        norm_lines = parsed["norm_lines"]
        line_numeric = parsed["line_numeric"]
        n = len(norm_lines)
        labels = _build_labels_from_doc(doc, n)
        doc_start = len(all_texts)
        all_texts.extend(norm_lines)
        all_numeric.extend(line_numeric)
        train_features_meta.append({
            "doc_id": str(doc.doc_id),
            "start": doc_start,
            "n_lines": n,
            "labels": labels,
            "has_anomaly": doc.has_anomaly,
        })

    train_text_count = len(all_texts)

    # Process test docs
    for i, (doc, parsed) in enumerate(tqdm(zip(test_docs, test_parsed), total=len(test_docs), desc="processing test docs", unit="doc", dynamic_ncols=True)):
        norm_lines = parsed["norm_lines"]
        line_numeric = parsed["line_numeric"]
        n = len(norm_lines)
        doc_start = len(all_texts)
        all_texts.extend(norm_lines)
        all_numeric.extend(line_numeric)
        test_features_meta.append({
            "doc_id": str(doc.doc_id),
            "start": doc_start,
            "n_lines": n,
        })

    assert len(all_texts) == total_lines
    assert len(all_numeric) == total_lines

    print("\nStep 3/6: Building sparse feature matrices...")
    char_sparse = _batch_transform(CHAR_VECTORIZER, all_texts)
    print(f"  Char features: {char_sparse.shape}")
    word_sparse = _batch_transform(WORD_VECTORIZER, all_texts)
    print(f"  Word features: {word_sparse.shape}")

    # Numeric to sparse
    numeric_dense = np.asarray(all_numeric, dtype=np.float32)
    print(f"  Numeric features: {numeric_dense.shape}")

    # Split into train/test
    char_train = char_sparse[:train_text_count]
    char_test = char_sparse[train_text_count:]
    word_train = word_sparse[:train_text_count]
    word_test = word_sparse[train_text_count:]
    numeric_train = numeric_dense[:train_text_count]
    numeric_test = numeric_dense[train_text_count:]

    print(f"  Train: char={char_train.shape}, word={word_train.shape}, numeric={numeric_train.shape}")
    print(f"  Test:  char={char_test.shape}, word={word_test.shape}, numeric={numeric_test.shape}")

    print(f"\nStep 4/6: Fitting SVD (n_components={n_comp})...")
    svd_char = TruncatedSVD(n_components=n_comp, random_state=SEED, algorithm="randomized")
    svd_word = TruncatedSVD(n_components=n_comp, random_state=SEED + 1, algorithm="randomized")

    print("  Fitting char SVD...")
    char_train_svd = svd_char.fit_transform(char_train)
    print(f"    Explained variance: {svd_char.explained_variance_ratio_.sum():.4f}")
    print("  Transforming char test...")
    char_test_svd = svd_char.transform(char_test)

    print("  Fitting word SVD...")
    word_train_svd = svd_word.fit_transform(word_train)
    print(f"    Explained variance: {svd_word.explained_variance_ratio_.sum():.4f}")
    print("  Transforming word test...")
    word_test_svd = svd_word.transform(word_test)

    print("\nStep 5/6: Scaling numeric features and concatenating...")
    scaler = StandardScaler()
    numeric_train_scaled = scaler.fit_transform(numeric_train)
    numeric_test_scaled = scaler.transform(numeric_test)

    # Concatenate: char_svd + word_svd + numeric_scaled
    train_dense = np.concatenate(
        [char_train_svd.astype(np.float32), word_train_svd.astype(np.float32), numeric_train_scaled.astype(np.float32)],
        axis=1,
    )
    test_dense = np.concatenate(
        [char_test_svd.astype(np.float32), word_test_svd.astype(np.float32), numeric_test_scaled.astype(np.float32)],
        axis=1,
    )
    total_dim = n_comp * 2 + numeric_train.shape[1]
    print(f"  Dense feature dim: {total_dim} ({n_comp} char + {n_comp} word + {numeric_train.shape[1]} numeric)")
    print(f"  Train dense: {train_dense.shape}, Test dense: {test_dense.shape}")

    print("\nStep 6/6: Assembling per-document arrays and saving...")
    # Assemble train features
    dense_train: list[dict] = []
    for meta in tqdm(train_features_meta, desc="assembling train", unit="doc", dynamic_ncols=True):
        s, n = meta["start"], meta["n_lines"]
        dense_train.append({
            "doc_id": meta["doc_id"],
            "features": np.asarray(train_dense[s:s + n], dtype=np.float32),
            "labels": np.asarray(meta["labels"], dtype=np.int64),
            "has_anomaly": int(meta["has_anomaly"]),
        })

    # Assemble test features
    dense_test: list[dict] = []
    # Test features start after train features, so offset by train_text_count
    for meta in tqdm(test_features_meta, desc="assembling test", unit="doc", dynamic_ncols=True):
        s = meta["start"] - train_text_count  # relative to test_dense
        n = meta["n_lines"]
        dense_test.append({
            "doc_id": meta["doc_id"],
            "features": np.asarray(test_dense[s:s + n], dtype=np.float32),
        })

    print("  Saving...")
    joblib.dump(dense_train, cache_dir / "dense_train.joblib", compress=3)
    joblib.dump(dense_test, cache_dir / "dense_test.joblib", compress=3)
    joblib.dump(svd_char, cache_dir / "svd_char.joblib", compress=3)
    joblib.dump(svd_word, cache_dir / "svd_word.joblib", compress=3)
    joblib.dump(scaler, cache_dir / "scaler_numeric.joblib", compress=3)

    print("\nDone!")
    print(f"  Train: {len(dense_train)} docs, dim={total_dim}")
    print(f"  Test:  {len(dense_test)} docs")
    print(f"  Files saved to: {cache_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
