"""Ensemble prediction v2: multi-model + per-type independent thresholds + long-doc tuning.

Key improvements over v1:
  - Per-type independent threshold search (10 types × 20 values)
  - Long-document specialized decoding (docs > 100 lines)
  - Global decoder param search (smooth, gap, spans) after per-type thresholds fixed
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    ANOMALY_TYPES, DEFAULT_SEED, TYPE_SPAN_LENGTHS,
    Prediction, evaluate_predictions,
    read_documents, read_sample_columns, validate_submission_file, write_submission,
)
from model_nn_v2 import LABEL_TO_TYPE, NUM_LABELS, O_LABEL, LogBiLSTM, collate_batch

# ── threshold decoder ──

def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) <= 1:
        return values.astype(np.float64)
    window = max(1, int(window))
    if window % 2 == 0: window += 1
    pad = window // 2
    padded = np.pad(values.astype(np.float64), (pad, pad), mode="edge")
    kernel = np.ones(window) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def _fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    if max_gap <= 0 or mask.size == 0:
        return mask.copy()
    mask = mask.copy(); n = len(mask); i = 0
    while i < n:
        if mask[i]: i += 1; continue
        s = i
        while i < n and not mask[i]: i += 1
        if s > 0 and i < n and (i - s) <= max_gap: mask[s:i] = True
    return mask


def _components(mask: np.ndarray) -> list[tuple[int, int]]:
    comps = []; i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        s = i
        while i < n and mask[i]: i += 1
        comps.append((s, i - 1))
    return comps


def _best_subspan(scores: np.ndarray, s: int, e: int,
                  min_len: int, max_len: int, target_len: int) -> tuple[int, int, float] | None:
    best = None
    for a in range(s, e + 1):
        acc = 0.0
        for b in range(a, min(e, a + max_len - 1) + 1):
            acc += float(scores[b]); length = b - a + 1
            if length < min_len: continue
            avg = acc / float(length)
            if target_len > 0: avg -= 0.01 * abs(length - target_len) / float(target_len)
            if best is None or avg > best[2] or (
                abs(avg - best[2]) <= 1e-12 and length > best[1] - best[0] + 1):
                best = (a, b, avg)
    return best


def decode_one(line_probs: np.ndarray, doc_prob: float, params: dict) -> Prediction:
    """Decode per-line probabilities into Prediction.

    params = {
        type_thresholds: {label: float},   # per-type thresholds
        smoothing_window: int, gap_merge: int,
        min_span_len: int, max_span_len: int, max_spans: int,
        doc_threshold: float,
        long_doc_thr_offset: float,  # subtract from threshold for long docs
        long_doc_max_len: int,       # max span len for long docs
        type_span_lengths: dict | None,
    }
    """
    n_lines = line_probs.shape[0]
    if n_lines == 0:
        return Prediction("unknown", 0, -1, -1, "none", "")

    is_long = n_lines > 100
    type_thr = params["type_thresholds"]
    sw = int(params.get("smoothing_window", 3))
    gm = int(params.get("gap_merge", 1))
    min_len = int(params.get("min_span_len", 2))
    max_len = int(params.get("max_span_len", 15))
    if is_long:
        max_len = params.get("long_doc_max_len", max_len)
    max_s = int(params.get("max_spans", 3))
    doc_thr = float(params.get("doc_threshold", 0.5))
    thr_offset = params.get("long_doc_thr_offset", 0.0)
    tsl = params.get("type_span_lengths", None)

    candidates: list[tuple[int, int, str, float]] = []

    for type_idx, label in enumerate(ANOMALY_TYPES):
        col = type_idx + 1  # skip O
        scores = _smooth(line_probs[:, col], sw)
        thr = float(type_thr.get(label, 0.5))
        if is_long and thr_offset > 0:
            thr = max(0.05, thr - thr_offset)

        cfg = (tsl or {}).get(label, {})
        lmin = max(1, int(cfg.get("min_len", min_len)))
        lmax = max(lmin, int(cfg.get("max_len", max_len)))
        tgt = max(lmin, min(lmax, int(cfg.get("target_len", (lmin + lmax) // 2))))

        mask = scores >= thr
        mask = _fill_gaps(mask, gm)
        for c_start, c_end in _components(mask):
            result = _best_subspan(scores, c_start, c_end, lmin, lmax, tgt)
            if result is not None:
                a, b, score = result
                candidates.append((a, b, label, score))

    # Deduplicate
    dedup: dict[tuple[int, int], tuple[int, int, str, float]] = {}
    for a, b, label, score in candidates:
        k = (a, b)
        if k not in dedup or score > dedup[k][3]:
            dedup[k] = (a, b, label, score)
    spans = list(dedup.values())

    # Boundary refinement ±1
    refined = []
    for a, b, label, score in spans:
        ti = ANOMALY_TYPES.index(label) + 1
        raw = line_probs[:, ti]
        best_a, best_b, best_score = a, b, score
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                na, nb = max(0, a + da), min(n_lines - 1, b + db)
                if na > nb: continue
                ln = nb - na + 1
                cfg2 = (tsl or {}).get(label, {})
                lmin2 = max(1, int(cfg2.get("min_len", min_len)))
                lmax2 = max(lmin2, int(cfg2.get("max_len", max_len)))
                if ln < lmin2 or ln > lmax2: continue
                avg = float(raw[na:nb + 1].mean())
                tgt2 = max(lmin2, min(lmax2, int(cfg2.get("target_len", (lmin2 + lmax2) // 2))))
                avg -= 0.005 * abs(ln - tgt2) / max(1, tgt2)
                if avg > best_score:
                    best_a, best_b, best_score = na, nb, avg
        refined.append((best_a, best_b, label, best_score))

    refined.sort(key=lambda x: (x[0], -x[3]))
    spans = refined[:max_s]

    if not spans and doc_prob < doc_thr:
        return Prediction("unknown", 0, -1, -1, "none", "")
    if not spans:
        preds = np.argmax(line_probs, axis=-1).astype(np.int64)
        spans = []; i = 0
        while i < n_lines:
            if preds[i] == O_LABEL: i += 1; continue
            lid, s = preds[i], i
            while i < n_lines and preds[i] == lid: i += 1
            if lid in LABEL_TO_TYPE: spans.append((int(s), int(i-1), LABEL_TO_TYPE[lid], float(line_probs[s:i, lid].mean())))
        if not spans: return Prediction("unknown", 0, -1, -1, "none", "")
    primary = spans[0]
    all_spans = ";".join(f"{a}|{b}|{t}" for a, b, t, _ in spans)
    return Prediction("unknown", 1, primary[0], primary[1], primary[2], all_spans)


# ── threshold search ──

def _default_params():
    return {
        "type_thresholds": {label: 0.35 for label in ANOMALY_TYPES},
        "smoothing_window": 3, "gap_merge": 1,
        "min_span_len": 2, "max_span_len": 15, "max_spans": 3,
        "doc_threshold": 0.5,
        "long_doc_thr_offset": 0.05, "long_doc_max_len": 20,
        "type_span_lengths": TYPE_SPAN_LENGTHS,
    }


def _score_params(params, search_docs, search_probs, search_doc_probs):
    preds = []
    for sp, sdp, d in zip(search_probs, search_doc_probs, search_docs):
        raw = decode_one(sp, sdp, params)
        preds.append(Prediction(str(d.doc_id), raw.has_anomaly, raw.primary_start_idx,
                                raw.primary_end_idx, raw.primary_anomaly_type, raw.all_spans))
    return evaluate_predictions(search_docs, preds)["score"]


def tune_per_type_thresholds(docs, probs, doc_probs, rng):
    """Line-search best threshold for each type independently."""
    type_thr = {}
    for label in ANOMALY_TYPES:
        best_thr, best_score = 0.35, -1.0
        for thr in np.linspace(0.10, 0.65, 20):
            params = _default_params()
            params["type_thresholds"] = {l: thr if l == label else 0.35 for l in ANOMALY_TYPES}
            score = _score_params(params, docs, probs, doc_probs)
            if score > best_score:
                best_score, best_thr = score, float(thr)
        type_thr[label] = best_thr
    return type_thr


def tune_global_params(docs, probs, doc_probs, type_thr, rng, n=150):
    """Random search over global decoder params with per-type thresholds fixed."""
    best_params = _default_params()
    best_params["type_thresholds"] = type_thr
    best_score = _score_params(best_params, docs, probs, doc_probs)

    for _ in tqdm(range(n), desc="tune global params", unit="trial", dynamic_ncols=True):
        params = {
            "type_thresholds": type_thr,
            "smoothing_window": int(rng.integers(1, 6)),
            "gap_merge": int(rng.integers(0, 3)),
            "min_span_len": int(rng.integers(1, 5)),
            "max_span_len": int(rng.integers(6, 20)),
            "max_spans": int(rng.integers(1, 4)),
            "doc_threshold": float(rng.uniform(0.2, 0.7)),
            "long_doc_thr_offset": float(rng.uniform(0.0, 0.15)),
            "long_doc_max_len": int(rng.integers(12, 30)),
            "type_span_lengths": TYPE_SPAN_LENGTHS,
        }
        score = _score_params(params, docs, probs, doc_probs)
        if score > best_score:
            best_score, best_params = score, params

    return best_params, best_score


def tune_all(docs, probs, doc_probs, n_global=150):
    """Full tuning: per-type thresholds first, then global params."""
    rng = np.random.default_rng(DEFAULT_SEED + 777)

    # Sample for speed
    y = np.array([d.has_anomaly for d in docs])
    pos_idx = np.flatnonzero(y == 1); neg_idx = np.flatnonzero(y == 0)
    if len(docs) > 5000:
        sampled = np.concatenate([
            rng.choice(pos_idx, size=min(2500, len(pos_idx)), replace=False),
            rng.choice(neg_idx, size=min(2500, len(neg_idx)), replace=False),
        ])
        idx = np.sort(sampled.astype(np.int64))
    else:
        idx = np.arange(len(docs))

    sd = [docs[int(i)] for i in idx]; sp = [probs[int(i)] for i in idx]; sdp = doc_probs[idx]

    print("  Stage 1: per-type thresholds...")
    type_thr = tune_per_type_thresholds(sd, sp, sdp, rng)
    print(f"    {', '.join(f'{k}={v:.3f}' for k, v in sorted(type_thr.items(), key=lambda x: -x[1]))}")

    print(f"  Stage 2: global params ({n_global} trials)...")
    best_params, best_score = tune_global_params(sd, sp, sdp, type_thr, rng, n_global)
    print(f"    Score={best_score:.6f}, sw={best_params['smoothing_window']}, gm={best_params['gap_merge']}, "
          f"msl={best_params['min_span_len']}, mxl={best_params['max_span_len']}, ms={best_params['max_spans']}, "
          f"dt={best_params['doc_threshold']:.3f}, ld_off={best_params['long_doc_thr_offset']:.3f}, ld_mxl={best_params['long_doc_max_len']}")

    return best_params, best_score


# ── Main ──

def load_models(paths: list[Path], device: torch.device) -> list:
    models = []
    for p in paths:
        b = joblib.load(p)
        m = LogBiLSTM(input_dim=b["input_dim"], hidden_dim=b["hidden_dim"],
                       num_labels=b["num_labels"], dropout=b["dropout"],
                       num_lstm_layers=b.get("lstm_layers", 2)).to(device)
        m.load_state_dict(b["state_dict"]); m.eval()
        models.append(m)
        print(f"  {p.name}: seed={b.get('seed','?')}")
    return models


@torch.no_grad()
def ensemble_predict(models, loader, device, params):
    preds = []
    for batch in tqdm(loader, desc="ensemble predict", unit="batch", dynamic_ncols=True):
        features = batch["features"].to(device); mask = batch["mask"].to(device)
        dids = batch["doc_ids"]

        alp, adp = None, None
        for model in models:
            ll, dl = model(features, mask)
            lp = F.softmax(ll, dim=-1).detach().cpu().numpy()
            dp = torch.sigmoid(dl).squeeze(-1).detach().cpu().numpy()
            if alp is None: alp = lp; adp = dp
            else: alp += lp; adp += dp
        alp /= len(models); adp /= len(models)

        for i, did in enumerate(dids):
            nv = int(mask[i].sum())
            raw = decode_one(alp[i, :nv], float(adp[i]), params)
            preds.append(Prediction(str(did), raw.has_anomaly, raw.primary_start_idx,
                                     raw.primary_end_idx, raw.primary_anomaly_type, raw.all_spans))
    return preds


def main() -> int:
    parser = argparse.ArgumentParser(description="Ensemble prediction v2 (per-type thresholds).")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--tune-trials", type=int, default=150)
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if args.output_path is None:
        args.output_path = data_dir / "提交结果" / "submission_ensemble_v2.csv"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {len(args.models)} models...")
    models = load_models(args.models, device)

    dense_test = joblib.load(data_dir / "缓存" / "dense_test.joblib")
    test_docs = read_documents(data_dir / "test.csv", expect_labels=False)
    for i, (f, d) in enumerate(zip(dense_test, test_docs)):
        if str(f["doc_id"]) != str(d.doc_id): raise ValueError(f"Mismatch at {i}")
    print(f"  Test: {len(dense_test)} docs")

    class DS:
        def __init__(self, d): self.d = d
        def __len__(self): return len(self.d)
        def __getitem__(self, i): return self.d[i]

    test_loader = DataLoader(DS(dense_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    params = _default_params()
    if args.tune:
        print("Tuning thresholds on training data...")
        dense_train = joblib.load(data_dir / "缓存" / "dense_train.joblib")
        train_docs = read_documents(data_dir / "train.csv", expect_labels=True)
        train_loader = DataLoader(DS(dense_train), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

        # Collect ensemble probabilities on training data
        doc_id_to_idx = {str(d.doc_id): i for i, d in enumerate(train_docs)}
        ensemble_probs = [None] * len(train_docs)
        ensemble_doc_probs = np.zeros(len(train_docs), dtype=np.float32)
        for batch in tqdm(train_loader, desc="ensemble on train", unit="batch", dynamic_ncols=True):
            features = batch["features"].to(device); mask = batch["mask"].to(device)
            dids = batch["doc_ids"]
            alp, adp = None, None
            for model in models:
                ll, dl = model(features, mask)
                lp = F.softmax(ll, dim=-1).detach().cpu().numpy()
                dp = torch.sigmoid(dl).squeeze(-1).detach().cpu().numpy()
                if alp is None: alp = lp; adp = dp
                else: alp += lp; adp += dp
            alp /= len(models); adp /= len(models)
            for i, did in enumerate(dids):
                nv = int(mask[i].sum())
                idx = doc_id_to_idx[str(did)]
                ensemble_probs[idx] = alp[i, :nv]
                ensemble_doc_probs[idx] = float(adp[i])

        params, best_score = tune_all(train_docs, ensemble_probs, ensemble_doc_probs, args.tune_trials)
        print(f"  Best ensemble OOF score: {best_score:.6f}")

    print("Predicting ensemble on test...")
    predictions = ensemble_predict(models, test_loader, device, params)
    write_submission(args.output_path, predictions)
    sample_columns = read_sample_columns(data_dir / "sample_submission.csv")
    validate_submission_file(args.output_path, test_docs, sample_columns)
    print(f"Submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
