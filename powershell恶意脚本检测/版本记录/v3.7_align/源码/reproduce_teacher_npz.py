from __future__ import annotations

import argparse
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import default_model_dir, feature_columns, key_tuple, normalize, package_root, resolve_data_path

warnings.filterwarnings("ignore")

LABELS = [0, 1, 2]
DEFAULT_FOLDS = 5
PAIR_BASE = 10
TRIPLE_BASE = 10

RAW_COLUMNS = [
    "function_scope_level",
    "branch_scope_level",
    "loop_scope_level",
    "parameter_block_presence",
    "pipeline_usage_level",
    "decode_activity_profile",
    "network_command_profile",
    "task_registry_profile",
    "credential_runtime_profile",
    "structure_rhythm_profile",
    "layout_variation_profile",
    "identifier_variation_profile",
    "content_encoding_profile",
    "command_surface_profile",
    "extension_import_profile",
]

SCOPE_COLS = [
    "function_scope_level",
    "branch_scope_level",
    "loop_scope_level",
    "parameter_block_presence",
    "pipeline_usage_level",
]

BEHAVIOR_COLS = [
    "decode_activity_profile",
    "network_command_profile",
    "task_registry_profile",
    "credential_runtime_profile",
    "content_encoding_profile",
    "extension_import_profile",
]

STRUCTURE_COLS = [
    "structure_rhythm_profile",
    "command_surface_profile",
]

VARIATION_COLS = [
    "layout_variation_profile",
    "identifier_variation_profile",
]

SELECTED_FEATURES = [
    "pair__function_scope_level__command_surface_profile",
    "rule_complex_surface",
    "triple__function_scope_level__structure_rhythm_profile__command_surface_profile",
    "triple__function_scope_level__parameter_block_presence__pipeline_usage_level",
    "freq_pair__decode_activity_profile__identifier_variation_profile_train",
    "exact_key_freq_test",
    "triple__function_scope_level__pipeline_usage_level__structure_rhythm_profile",
    "exact_key_freq_train",
    "freq_pair__credential_runtime_profile__identifier_variation_profile_train",
    "pair__loop_scope_level__decode_activity_profile",
    "freq_pair__identifier_variation_profile__command_surface_profile_train",
    "pair__identifier_variation_profile__command_surface_profile",
    "freq_pair__credential_runtime_profile__structure_rhythm_profile_test",
    "pair__function_scope_level__identifier_variation_profile",
    "freq_pair__identifier_variation_profile__command_surface_profile_test",
    "freq_pair__credential_runtime_profile__identifier_variation_profile_test",
    "pair__decode_activity_profile__layout_variation_profile",
    "pair__identifier_variation_profile__extension_import_profile",
    "exact_key_freq_all",
    "pair__decode_activity_profile__identifier_variation_profile",
    "freq_pair__layout_variation_profile__identifier_variation_profile_train",
    "pair__loop_scope_level__content_encoding_profile",
    "rule_decode_and_encoded",
    "pair__decode_activity_profile__content_encoding_profile",
    "triple__decode_activity_profile__credential_runtime_profile__identifier_variation_profile",
    "pair__function_scope_level__extension_import_profile",
    "freq_pair__decode_activity_profile__identifier_variation_profile_test",
    "exact_key_logfreq_test",
    "freq_pair__loop_scope_level__pipeline_usage_level_train",
    "freq_pair__branch_scope_level__decode_activity_profile_test",
    "triple__function_scope_level__credential_runtime_profile__command_surface_profile",
    "freq_pair__decode_activity_profile__structure_rhythm_profile_train",
    "exact_key_logfreq_train",
    "freq_pair__loop_scope_level__structure_rhythm_profile_train",
    "freq_pair__loop_scope_level__decode_activity_profile_train",
    "triple__function_scope_level__parameter_block_presence__credential_runtime_profile",
    "pair__decode_activity_profile__task_registry_profile",
    "triple__pipeline_usage_level__structure_rhythm_profile__identifier_variation_profile",
    "freq_pair__pipeline_usage_level__identifier_variation_profile_train",
    "agg_all_one_count",
    "freq_pair__function_scope_level__identifier_variation_profile_train",
    "agg_all_sum",
    "freq_pair__identifier_variation_profile__content_encoding_profile_train",
    "freq_pair__branch_scope_level__identifier_variation_profile_test",
    "triple__pipeline_usage_level__credential_runtime_profile__structure_rhythm_profile",
    "pair__parameter_block_presence__identifier_variation_profile",
    "freq_pair__branch_scope_level__parameter_block_presence_train",
    "freq_pair__loop_scope_level__content_encoding_profile_train",
    "pair__identifier_variation_profile__content_encoding_profile",
    "freq_pair__task_registry_profile__identifier_variation_profile_train",
    "triple__parameter_block_presence__decode_activity_profile__structure_rhythm_profile",
    "pair__layout_variation_profile__content_encoding_profile",
    "freq_pair__structure_rhythm_profile__identifier_variation_profile_train",
    "pair__parameter_block_presence__structure_rhythm_profile",
    "pair__credential_runtime_profile__content_encoding_profile",
    "freq_pair__branch_scope_level__loop_scope_level_train",
    "freq_pair__structure_rhythm_profile__extension_import_profile_train",
    "freq_pair__structure_rhythm_profile__command_surface_profile_test",
    "agg_all_zero_count",
    "agg_behavior_sum",
    "freq_pair__credential_runtime_profile__content_encoding_profile_train",
    "freq_pair__function_scope_level__content_encoding_profile_train",
    "freq_pair__task_registry_profile__command_surface_profile_train",
    "pair__branch_scope_level__identifier_variation_profile",
    "freq_pair__branch_scope_level__identifier_variation_profile_train",
    "pair__loop_scope_level__identifier_variation_profile",
    "agg_scope_behavior_sum",
    "pair__task_registry_profile__content_encoding_profile",
    "agg_variation_sum",
    "freq_pair__layout_variation_profile__identifier_variation_profile_test",
    "triple__parameter_block_presence__pipeline_usage_level__identifier_variation_profile",
    "triple__decode_activity_profile__structure_rhythm_profile__command_surface_profile",
    "pair__pipeline_usage_level__identifier_variation_profile",
    "triple__function_scope_level__decode_activity_profile__identifier_variation_profile",
    "freq_pair__branch_scope_level__decode_activity_profile_train",
    "triple__pipeline_usage_level__decode_activity_profile__credential_runtime_profile",
    "pair__loop_scope_level__credential_runtime_profile",
    "freq_pair__structure_rhythm_profile__content_encoding_profile_train",
    "agg_variation_one_count",
    "pair__loop_scope_level__extension_import_profile",
    "freq_pair__parameter_block_presence__content_encoding_profile_train",
    "freq_pair__credential_runtime_profile__command_surface_profile_train",
    "freq_pair__parameter_block_presence__task_registry_profile_train",
    "freq_pair__decode_activity_profile__structure_rhythm_profile_test",
    "pair__layout_variation_profile__identifier_variation_profile",
    "freq_pair__function_scope_level__parameter_block_presence_train",
    "layout_variation_profile",
    "triple__parameter_block_presence__decode_activity_profile__identifier_variation_profile",
    "agg_scope_behavior_one_count",
    "freq_pair__loop_scope_level__task_registry_profile_test",
    "freq_pair__branch_scope_level__content_encoding_profile_train",
    "freq_pair__branch_scope_level__pipeline_usage_level_train",
    "freq_pair__loop_scope_level__identifier_variation_profile_train",
    "freq_pair__loop_scope_level__parameter_block_presence_train",
    "agg_behavior_one_count",
    "triple__function_scope_level__pipeline_usage_level__identifier_variation_profile",
    "pair__parameter_block_presence__extension_import_profile",
    "freq_pair__pipeline_usage_level__structure_rhythm_profile_train",
    "pair__pipeline_usage_level__credential_runtime_profile",
    "freq_pair__content_encoding_profile__extension_import_profile_train",
    "triple__function_scope_level__parameter_block_presence__structure_rhythm_profile",
    "pair__structure_rhythm_profile__content_encoding_profile",
    "freq_pair__decode_activity_profile__layout_variation_profile_train",
    "freq_pair__parameter_block_presence__structure_rhythm_profile_train",
    "freq_pair__identifier_variation_profile__extension_import_profile_train",
    "triple__parameter_block_presence__credential_runtime_profile__structure_rhythm_profile",
    "freq_pair__decode_activity_profile__credential_runtime_profile_train",
    "triple__decode_activity_profile__structure_rhythm_profile__identifier_variation_profile",
    "freq_pair__branch_scope_level__structure_rhythm_profile_train",
    "agg_struct_sum",
    "triple__parameter_block_presence__structure_rhythm_profile__identifier_variation_profile",
    "freq_pair__pipeline_usage_level__extension_import_profile_train",
    "freq_pair__decode_activity_profile__content_encoding_profile_train",
    "triple__structure_rhythm_profile__identifier_variation_profile__command_surface_profile",
    "freq_pair__credential_runtime_profile__extension_import_profile_train",
    "triple__credential_runtime_profile__structure_rhythm_profile__identifier_variation_profile",
    "freq_pair__loop_scope_level__structure_rhythm_profile_test",
    "freq_pair__pipeline_usage_level__identifier_variation_profile_test",
    "triple__decode_activity_profile__credential_runtime_profile__structure_rhythm_profile",
    "triple__parameter_block_presence__credential_runtime_profile__identifier_variation_profile",
]


@dataclass(frozen=True)
class ArtifactSpec:
    file_name: str
    seed: int
    params: Dict[str, object]


ARTIFACTS = {
    "tuned_smoke_group_lgbm_tuned_00.npz": ArtifactSpec(
        file_name="tuned_smoke_group_lgbm_tuned_00.npz",
        seed=42,
        params=dict(
            n_estimators=360,
            learning_rate=0.04,
            num_leaves=63,
            min_child_samples=30,
            subsample=0.97,
            colsample_bytree=0.95,
            reg_alpha=0.0,
            reg_lambda=5.0,
            min_split_gain=0.03,
        ),
    ),
    "tuned_smoke_group_lgbm_tuned_01.npz": ArtifactSpec(
        file_name="tuned_smoke_group_lgbm_tuned_01.npz",
        seed=73,
        params=dict(
            n_estimators=260,
            learning_rate=0.04,
            num_leaves=15,
            min_child_samples=45,
            subsample=0.75,
            colsample_bytree=0.75,
            reg_alpha=0.03,
            reg_lambda=0.3,
            min_split_gain=0.03,
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the two reference teacher npz files.")
    parser.add_argument("--data-dir", default=None, help="Directory containing data_train.csv and data_test.csv.")
    parser.add_argument("--train", default=None, help="Explicit path to data_train.csv.")
    parser.add_argument("--test", default=None, help="Explicit path to data_test.csv.")
    parser.add_argument("--model-dir", default=None, help="Directory where npz files will be written.")
    parser.add_argument(
        "--artifact",
        default="all",
        choices=["all", "tuned_smoke_group_lgbm_tuned_00.npz", "tuned_smoke_group_lgbm_tuned_01.npz"],
        help="Which artifact to generate.",
    )
    parser.add_argument(
        "--verify-against",
        default=None,
        help="Path to a reference npz file or a directory containing the reference npz files.",
    )
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help="Number of CV folds.")
    parser.add_argument("--n-jobs", type=int, default=1, help="LightGBM threads. Default 1 for reproducibility.")
    return parser.parse_args()


def resolve_input_paths(root: Path, args: argparse.Namespace) -> tuple[Path, Path]:
    if args.train:
        train_path = Path(args.train).expanduser().resolve()
    elif args.data_dir:
        train_path = (Path(args.data_dir).expanduser().resolve() / "data_train.csv")
    else:
        train_path = resolve_data_path(root, "train", None)

    if args.test:
        test_path = Path(args.test).expanduser().resolve()
    elif args.data_dir:
        test_path = (Path(args.data_dir).expanduser().resolve() / "data_test.csv")
    else:
        test_path = resolve_data_path(root, "test", None)

    if not train_path.exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found: {test_path}")
    return train_path, test_path


def exact_group_splits(train: pd.DataFrame, y: np.ndarray, features: list[str], n_folds: int):
    group_ids: Dict[Tuple[int, ...], int] = {}
    groups = []
    for row in train[features].itertuples(index=False, name=None):
        key = key_tuple(row)
        if key not in group_ids:
            group_ids[key] = len(group_ids)
        groups.append(group_ids[key])
    cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    return list(cv.split(train[features], y, np.asarray(groups, dtype=int)))


def align_proba(classes: np.ndarray, proba: np.ndarray) -> np.ndarray:
    aligned = np.zeros((proba.shape[0], len(LABELS)), dtype=float)
    for i, cls in enumerate(classes):
        aligned[:, int(cls)] = proba[:, i]
    return aligned


def build_count_map(frame: pd.DataFrame, cols: Sequence[str]) -> Dict[Tuple[int, ...], int]:
    mapping: Dict[Tuple[int, ...], int] = {}
    grouped = frame.groupby(list(cols), sort=False).size()
    for key, value in grouped.items():
        if not isinstance(key, tuple):
            key = (key,)
        mapping[tuple(int(v) for v in key)] = int(value)
    return mapping


def map_count_values(frame: pd.DataFrame, cols: Sequence[str], mapping: Dict[Tuple[int, ...], int]) -> np.ndarray:
    values = [mapping.get(tuple(int(v) for v in row), 0) for row in frame[list(cols)].itertuples(index=False, name=None)]
    return np.asarray(values, dtype=float)


def encode_pair(frame: pd.DataFrame, a: str, b: str) -> np.ndarray:
    return frame[a].to_numpy(dtype=np.int64) * PAIR_BASE + frame[b].to_numpy(dtype=np.int64)


def encode_triple(frame: pd.DataFrame, a: str, b: str, c: str) -> np.ndarray:
    return (
        frame[a].to_numpy(dtype=np.int64) * (TRIPLE_BASE * TRIPLE_BASE)
        + frame[b].to_numpy(dtype=np.int64) * TRIPLE_BASE
        + frame[c].to_numpy(dtype=np.int64)
    )


class FeatureRecipe:
    def __init__(self, train: pd.DataFrame, test: pd.DataFrame):
        self.train = train
        self.test = test
        self.raw_columns = RAW_COLUMNS
        all_frame = pd.concat([train[self.raw_columns], test[self.raw_columns]], ignore_index=True)
        self.frames = {
            "train": train[self.raw_columns].copy(),
            "test": test[self.raw_columns].copy(),
            "all": all_frame,
        }
        self.count_cache: Dict[tuple[str, tuple[str, ...]], Dict[Tuple[int, ...], int]] = {}

    def count_map(self, split: str, cols: Sequence[str]) -> Dict[Tuple[int, ...], int]:
        key = (split, tuple(cols))
        if key not in self.count_cache:
            self.count_cache[key] = build_count_map(self.frames[split], cols)
        return self.count_cache[key]

    def count_feature(self, frame: pd.DataFrame, split: str, cols: Sequence[str]) -> np.ndarray:
        return map_count_values(frame, cols, self.count_map(split, cols))

    def build_column(self, frame: pd.DataFrame, name: str) -> np.ndarray:
        if name in self.raw_columns:
            return frame[name].to_numpy(dtype=float)

        if name.startswith("pair__"):
            left = name[len("pair__") :]
            a, b = left.split("__")
            return encode_pair(frame, a, b).astype(float)

        if name.startswith("triple__"):
            left = name[len("triple__") :]
            a, b, c = left.split("__")
            return encode_triple(frame, a, b, c).astype(float)

        if name.startswith("freq_pair__"):
            rest = name[len("freq_pair__") :]
            cols_part, split = rest.rsplit("_", 1)
            a, b = cols_part.split("__")
            return self.count_feature(frame, split, [a, b])

        if name == "exact_key_freq_train":
            return self.count_feature(frame, "train", self.raw_columns)
        if name == "exact_key_freq_test":
            return self.count_feature(frame, "test", self.raw_columns)
        if name == "exact_key_freq_all":
            return self.count_feature(frame, "all", self.raw_columns)
        if name == "exact_key_logfreq_train":
            return np.log1p(self.count_feature(frame, "train", self.raw_columns))
        if name == "exact_key_logfreq_test":
            return np.log1p(self.count_feature(frame, "test", self.raw_columns))

        if name == "agg_all_zero_count":
            return (frame[self.raw_columns] == 0).sum(axis=1).to_numpy(dtype=float)
        if name == "agg_all_one_count":
            return (frame[self.raw_columns] == 1).sum(axis=1).to_numpy(dtype=float)
        if name == "agg_all_sum":
            return frame[self.raw_columns].sum(axis=1).to_numpy(dtype=float)

        if name == "agg_behavior_sum":
            return frame[BEHAVIOR_COLS].sum(axis=1).to_numpy(dtype=float)
        if name == "agg_behavior_one_count":
            return (frame[BEHAVIOR_COLS] == 1).sum(axis=1).to_numpy(dtype=float)
        if name == "agg_scope_behavior_sum":
            return frame[SCOPE_COLS + BEHAVIOR_COLS].sum(axis=1).to_numpy(dtype=float)
        if name == "agg_scope_behavior_one_count":
            return (frame[SCOPE_COLS + BEHAVIOR_COLS] == 1).sum(axis=1).to_numpy(dtype=float)
        if name == "agg_variation_sum":
            return frame[VARIATION_COLS].sum(axis=1).to_numpy(dtype=float)
        if name == "agg_variation_one_count":
            return (frame[VARIATION_COLS] == 1).sum(axis=1).to_numpy(dtype=float)
        if name == "agg_struct_sum":
            return frame[STRUCTURE_COLS].sum(axis=1).to_numpy(dtype=float)

        if name == "rule_complex_surface":
            score = (
                frame["structure_rhythm_profile"]
                + frame["command_surface_profile"]
                + frame["layout_variation_profile"]
                + frame["identifier_variation_profile"]
            )
            return (score >= 4).to_numpy(dtype=float)
        if name == "rule_decode_and_encoded":
            return (
                (frame["decode_activity_profile"] > 0)
                & (frame["content_encoding_profile"] > 0)
            ).to_numpy(dtype=float)

        raise KeyError(f"Unsupported feature: {name}")

    def build_matrix(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [self.build_column(frame, name) for name in SELECTED_FEATURES]
        return np.column_stack(columns).astype(np.float64, copy=False)


def train_one_artifact(
    spec: ArtifactSpec,
    x_train: np.ndarray,
    x_test: np.ndarray,
    y: np.ndarray,
    splits,
    n_jobs: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    oof = np.zeros((len(y), len(LABELS)), dtype=float)
    test_probs = np.zeros((len(x_test), len(LABELS)), dtype=float)
    fold_scores: list[float] = []

    for fold_id, (tr_idx, va_idx) in enumerate(splits, start=1):
        model = LGBMClassifier(
            objective="multiclass",
            num_class=len(LABELS),
            random_state=spec.seed,
            n_jobs=n_jobs,
            verbosity=-1,
            **spec.params,
        )
        model.fit(x_train[tr_idx], y[tr_idx])
        valid = align_proba(model.classes_, model.predict_proba(x_train[va_idx]))
        test_fold = align_proba(model.classes_, model.predict_proba(x_test))
        oof[va_idx] = valid
        test_probs += test_fold / len(splits)
        score = float(f1_score(y[va_idx], valid.argmax(axis=1), average="macro"))
        fold_scores.append(score)
        print(f"  fold {fold_id}/{len(splits)} macro_f1={score:.6f}")

    return normalize(oof), normalize(test_probs), fold_scores


def save_artifact(
    out_path: Path,
    oof: np.ndarray,
    test_probs: np.ndarray,
    summary: dict,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        oof=oof,
        test_probs=test_probs,
        summary_json=np.array(json.dumps(summary, ensure_ascii=False, indent=2)),
    )


def verify_artifact(generated: Path, reference: Path) -> dict:
    gen = np.load(generated, allow_pickle=True)
    ref = np.load(reference, allow_pickle=True)
    gen_summary = json.loads(gen["summary_json"].item())
    ref_summary = json.loads(ref["summary_json"].item())

    def array_stats(a: np.ndarray, b: np.ndarray) -> dict:
        diff = np.abs(a - b)
        return {
            "shape_match": list(a.shape) == list(b.shape),
            "mean_abs_diff": float(diff.mean()),
            "p95_abs_diff": float(np.quantile(diff, 0.95)),
            "max_abs_diff": float(diff.max()),
            "argmax_agreement": float((a.argmax(axis=1) == b.argmax(axis=1)).mean()),
        }

    report = {
        "generated": str(generated),
        "reference": str(reference),
        "keys_match": gen.files == ref.files,
        "summary": {
            "selected_feature_count_match": gen_summary["selected_feature_count"] == ref_summary["selected_feature_count"],
            "selected_features_match": gen_summary["selected_features"] == ref_summary["selected_features"],
            "fold_count_match": len(gen_summary["fold_scores"]) == len(ref_summary["fold_scores"]),
            "fold_scores_abs_diff_mean": float(
                np.mean(np.abs(np.asarray(gen_summary["fold_scores"]) - np.asarray(ref_summary["fold_scores"])))
            ),
            "oof_macro_f1_abs_diff": float(abs(gen_summary["oof_macro_f1"] - ref_summary["oof_macro_f1"])),
        },
        "oof": array_stats(gen["oof"], ref["oof"]),
        "test_probs": array_stats(gen["test_probs"], ref["test_probs"]),
    }
    return report


def main() -> None:
    args = parse_args()
    root = package_root(__file__)
    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else default_model_dir(root) / "teacher_npz_repro"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_path, test_path = resolve_input_paths(root, args)
    print(f"train: {train_path}")
    print(f"test : {test_path}")
    print(f"model_dir: {model_dir}")
    print(f"n_jobs: {args.n_jobs}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if "label" not in train.columns:
        raise ValueError("Training data must contain a 'label' column.")
    if "name" not in train.columns or "name" not in test.columns:
        raise ValueError("Both train and test data must contain a 'name' column.")

    y = train["label"].astype(int).to_numpy()
    raw_cols = feature_columns(train)
    if raw_cols != RAW_COLUMNS:
        print("warning: raw column order differs from the expected reference schema")
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    splits = exact_group_splits(train, y, raw_cols, args.folds)
    recipe = FeatureRecipe(train, test)
    x_train = recipe.build_matrix(train)
    x_test = recipe.build_matrix(test)

    print(f"rows: train={len(train)}, test={len(test)}")
    print(f"feature_dim: {x_train.shape[1]}")
    print(f"selected_feature_count: {len(SELECTED_FEATURES)}")
    print(f"label_distribution: {pd.Series(y).value_counts().sort_index().to_dict()}")
    print(f"cv_folds: {len(splits)}")

    chosen = list(ARTIFACTS.values())
    if args.artifact != "all":
        chosen = [ARTIFACTS[args.artifact]]

    verification_root = Path(args.verify_against).expanduser().resolve() if args.verify_against else None
    all_reports = []

    for spec in chosen:
        print(f"training {spec.file_name}")
        oof, test_probs, fold_scores = train_one_artifact(spec, x_train, x_test, y, splits, args.n_jobs)
        oof_f1 = float(f1_score(y, oof.argmax(axis=1), average="macro"))
        summary = {
            "family": "lgbm",
            "params": spec.params,
            "selected_feature_count": len(SELECTED_FEATURES),
            "selected_features": SELECTED_FEATURES,
            "fold_scores": [float(v) for v in fold_scores],
            "oof_macro_f1": oof_f1,
        }
        out_path = model_dir / spec.file_name
        save_artifact(out_path, oof, test_probs, summary)
        print(f"  saved: {out_path}")
        print(f"  oof_macro_f1={oof_f1:.6f}")

        if verification_root is not None:
            reference_path = verification_root if verification_root.is_file() else verification_root / spec.file_name
            if not reference_path.exists():
                raise FileNotFoundError(f"Reference artifact not found: {reference_path}")
            report = verify_artifact(out_path, reference_path)
            report_path = model_dir / f"{spec.file_name}.verify.json"
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            all_reports.append(report)
            print(f"  verify report: {report_path}")
            print(
                "  verify oof: "
                f"mean_abs_diff={report['oof']['mean_abs_diff']:.6f} "
                f"p95={report['oof']['p95_abs_diff']:.6f} "
                f"argmax={report['oof']['argmax_agreement']:.6f}"
            )
            print(
                "  verify test: "
                f"mean_abs_diff={report['test_probs']['mean_abs_diff']:.6f} "
                f"p95={report['test_probs']['p95_abs_diff']:.6f} "
                f"argmax={report['test_probs']['argmax_agreement']:.6f}"
            )

    if all_reports:
        combined_path = model_dir / "verification_report.json"
        combined_path.write_text(json.dumps(all_reports, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"verification summary: {combined_path}")

    print("done")


if __name__ == "__main__":
    main()
