"""Prediction script for BiLSTM v3 boundary model."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import (
    Prediction,
    read_documents,
    read_sample_columns,
    validate_submission_file,
    write_submission,
)
from model_nn_v3 import (
    LABEL_TO_TYPE,
    NUM_LABELS,
    O_LABEL,
    LogBiLSTMv3,
    collate_batch,
    decode_boundary_spans,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict with BiLSTM v3 boundary model.")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--test-file", type=str, default="test.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    if args.model_path is None:
        args.model_path = data_dir / "模型" / "model_bundle_nn_v3.joblib"
    if args.output_path is None:
        args.output_path = data_dir / "提交结果" / "submission_nn_v3.csv"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    bundle = joblib.load(args.model_path)
    if bundle.get("model") != "BiLSTM_v3_boundary":
        raise ValueError(f"Expected BiLSTM_v3_boundary, got {bundle.get('model')}")

    model = LogBiLSTMv3(
        input_dim=bundle["input_dim"], hidden_dim=bundle["hidden_dim"],
        num_labels=bundle["num_labels"], dropout=bundle["dropout"],
        num_lstm_layers=bundle["lstm_layers"],
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    thresholds = bundle.get("thresholds", {})
    print(f"  Thresholds: start={thresholds.get('start_threshold', 0.5):.3f}, end={thresholds.get('end_threshold', 0.5):.3f}")

    cache_dir = data_dir / "缓存"
    dense_test = joblib.load(cache_dir / "dense_test.joblib")
    test_docs = read_documents(data_dir / args.test_file, expect_labels=False)
    for i, (f, d) in enumerate(zip(dense_test, test_docs)):
        if str(f["doc_id"]) != str(d.doc_id):
            raise ValueError(f"Order mismatch at {i}")
    print(f"  {len(dense_test)} test documents")

    class DS:
        def __init__(self, data): self.data = data
        def __len__(self): return len(self.data)
        def __getitem__(self, i): return self.data[i]

    loader = DataLoader(DS(dense_test), batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    st = thresholds.get("start_threshold", 0.5)
    et = thresholds.get("end_threshold", 0.5)
    min_len = thresholds.get("min_span_len", 2)
    max_len = thresholds.get("max_span_len", 15)
    max_s = thresholds.get("max_spans", 3)
    tsl = thresholds.get("type_span_lengths", None)
    doc_thr = thresholds.get("doc_threshold", 0.5)

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predict test", unit="batch", dynamic_ncols=True):
            features = batch["features"].to(device); mask = batch["mask"].to(device)
            doc_ids = batch["doc_ids"]
            boundary_logits, type_logits, doc_logits = model(features, mask)

            sp = torch.sigmoid(boundary_logits[:,:,0]).cpu().numpy()
            ep = torch.sigmoid(boundary_logits[:,:,1]).cpu().numpy()
            tp = F.softmax(type_logits, dim=-1).cpu().numpy()
            dp = torch.sigmoid(doc_logits).squeeze(-1).cpu().numpy()

            for i, did in enumerate(doc_ids):
                nv = int(mask[i].sum())
                spans = decode_boundary_spans(sp[i,:nv], ep[i,:nv], tp[i,:nv], st, et, min_len, max_len, max_s, tsl)
                doc_p = float(dp[i])

                if not spans and doc_p < doc_thr:
                    predictions.append(Prediction(str(did), 0, -1, -1, "none", ""))
                elif not spans:
                    type_preds = np.argmax(tp[i,:nv], axis=-1).astype(np.int64)
                    spans = []
                    j = 0
                    while j < len(type_preds):
                        if type_preds[j] == O_LABEL: j += 1; continue
                        lid = type_preds[j]; s = j
                        while j < len(type_preds) and type_preds[j] == lid: j += 1
                        if lid in LABEL_TO_TYPE: spans.append((int(s), int(j-1), LABEL_TO_TYPE[lid], 0.5))
                    if not spans:
                        predictions.append(Prediction(str(did), 0, -1, -1, "none", ""))
                    else:
                        p = spans[0]
                        predictions.append(Prediction(str(did), 1, p[0], p[1], p[2],
                                          ";".join(f"{s}|{e}|{t}" for s,e,t,_ in spans)))
                else:
                    p = spans[0]
                    predictions.append(Prediction(str(did), 1, p[0], p[1], p[2],
                                      ";".join(f"{s}|{e}|{t}" for s,e,t,_ in spans)))

    write_submission(args.output_path, predictions)
    sample_columns = read_sample_columns(data_dir / args.sample_file)
    validate_submission_file(args.output_path, test_docs, sample_columns)
    print(f"Submission: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
