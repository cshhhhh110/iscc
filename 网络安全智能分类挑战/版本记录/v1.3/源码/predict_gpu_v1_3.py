from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    ID_COL,
    LABEL_COL,
    append_action_log,
    ensure_feature_columns,
    make_feature_frame,
    read_table,
    validate_prediction_frame,
)
from gpu_v1_3_core import (
    VERSION,
    build_candidate_probs_from_bundle,
    predict_probs_from_bundle,
    torch_load_bundle,
)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=f"Predict with the {VERSION} compact ResMLP bundle.")
    parser.add_argument("--data-dir", type=Path, default=root)
    parser.add_argument("--train-file", type=str, default="train_data.csv")
    parser.add_argument("--test-file", type=str, default="test_data.csv")
    parser.add_argument("--sample-file", type=str, default="sample_submission.csv")
    parser.add_argument("--bundle-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--metadata-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--action-log", type=Path, default=root / "ACTION_LOG.md")
    parser.add_argument("--no-candidates", action="store_true", help="Only write the selected official submission.")
    parser.add_argument("--smoke", action="store_true", help="Use smoke_* v1.3 paths.")
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()

    prefix = "smoke_" if args.smoke else ""
    if args.bundle_path is None:
        args.bundle_path = root / "模型" / f"{prefix}gpu_model_bundle_{VERSION}.pt"
    if args.output_path is None:
        args.output_path = root / "提交结果" / f"{prefix}submission_gpu_{VERSION}.csv"
    if args.metadata_path is None:
        args.metadata_path = root / "模型" / f"{prefix}gpu_prediction_metadata_{VERSION}.json"
    return args


def write_csv_with_retry(frame: pd.DataFrame, path: Path, retries: int = 5, delay: float = 1.0) -> None:
    last_error: PermissionError | None = None
    for attempt in range(1, retries + 1):
        try:
            with path.open("w", encoding="utf-8", newline="") as handle:
                frame.to_csv(handle, index=False)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(delay)
    assert last_error is not None
    raise last_error


def source_to_filename_part(source: str) -> str:
    if source.startswith("seed_"):
        return "seed" + source.split("_", 1)[1]
    return source


def candidate_output_path(base_path: Path, source: str) -> Path:
    return base_path.with_name(f"{base_path.stem}_{source_to_filename_part(source)}{base_path.suffix}")


def make_submission(test_df: pd.DataFrame, label_names: list[str], probs: np.ndarray) -> pd.DataFrame:
    labels = np.asarray(label_names)[np.argmax(probs, axis=1)]
    return pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})


def main() -> int:
    args = parse_args()
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_float32_matmul_precision("high")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA is not available. Use the iscc-gpu environment or pass --allow-cpu.")

    bundle = torch_load_bundle(args.bundle_path)
    data_dir = args.data_dir.resolve()
    train_df = read_table(data_dir / args.train_file)
    test_df = read_table(data_dir / args.test_file)
    sample_df = read_table(data_dir / args.sample_file)
    feature_cols = ensure_feature_columns(train_df, test_df)
    if feature_cols != list(bundle["feature_columns"]):
        raise ValueError("feature columns do not match the saved bundle")

    x_test = make_feature_frame(test_df, feature_cols).to_numpy(dtype=np.float32)
    all_probs, seed_probs = predict_probs_from_bundle(
        bundle,
        x_test,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    candidates = build_candidate_probs_from_bundle(bundle, all_probs, seed_probs)
    selected_source = bundle.get("selected", {}).get("source", "all_average")
    if selected_source not in candidates:
        selected_source = "all_average"

    label_names = list(bundle["label_names"])
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    output_records: dict[str, str] = {}
    selected_submission = make_submission(test_df, label_names, candidates[selected_source])
    validate_prediction_frame(selected_submission, test_df, label_names, list(sample_df.columns))
    write_csv_with_retry(selected_submission, args.output_path)
    output_records[selected_source] = str(args.output_path)

    if not args.no_candidates:
        for source, probs in sorted(candidates.items(), key=lambda item: item[0]):
            if source == selected_source:
                continue
            path = candidate_output_path(args.output_path, source)
            submission = make_submission(test_df, label_names, probs)
            validate_prediction_frame(submission, test_df, label_names, list(sample_df.columns))
            write_csv_with_retry(submission, path)
            output_records[source] = str(path)

    metadata = {
        "version": VERSION,
        "bundle_path": str(args.bundle_path),
        "selected_source": selected_source,
        "default_output_path": str(args.output_path),
        "candidate_outputs": output_records,
        "bundle_selected": bundle.get("selected", {}),
        "candidate_reports": bundle.get("candidate_reports", {}),
        "rows": int(len(test_df)),
    }
    with args.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    append_action_log(
        args.action_log,
        f"{VERSION} predict completed: selected={selected_source}, output={args.output_path}, "
        f"candidates={len(output_records)}.",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
