from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from common import ARTIFACT_VERSION, MODEL_DIR, RESULT_DIR, TEST_PATH, append_log, load_test, predict_from_bundle, validate_features, write_csv_atomic


def bundle_uses_torch(bundle: dict) -> bool:
    selected = bundle.get("selected_model_bundle", bundle)
    model_type = selected.get("model_type")
    if model_type == "torch_ensemble":
        return True
    if model_type == "fusion":
        return any(bundle_uses_torch(component) for component in selected.get("components", {}).values())
    return False


def resolve_device(spec: str):
    import torch

    spec = spec.lower().strip()
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if spec == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict PowerShell test labels.")
    parser.add_argument("--model", type=Path, default=MODEL_DIR / f"model_bundle_{ARTIFACT_VERSION}.joblib")
    parser.add_argument("--test", type=Path, default=TEST_PATH)
    parser.add_argument("--output", type=Path, default=RESULT_DIR / f"submission_{ARTIFACT_VERSION}.csv")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    parser.add_argument("--no-log", action="store_true", help="Do not append project log.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    features = bundle["feature_columns"]

    device = None
    if bundle_uses_torch(bundle):
        device = resolve_device(args.device)

    test_df = load_test(args.test)
    validate_features(test_df, features)
    X = test_df[features].astype("int16")
    pred = predict_from_bundle(bundle, X, device=device).astype(int)

    submission = pd.DataFrame({"name": test_df["name"], "label": pred})
    write_csv_atomic(submission, args.output, index=False, encoding="utf-8")
    if not args.no_log:
        append_log(f"generated PowerShell submission with {len(submission)} rows: {args.output}")
    print(f"Saved submission: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
