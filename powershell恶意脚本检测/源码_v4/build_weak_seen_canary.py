"""Build weak-seen canary: low-count (<30), stratified by bucket + entropy."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = PKG_ROOT / "data_train.csv"
SPLIT_DIR = PKG_ROOT / "模型" / "v4_splits"
RNG = np.random.default_rng(42)


def entropy(p):
    p = np.clip(p, 1e-12, 1); return -np.sum(p * np.log(p))


def main():
    train = pd.read_csv(TRAIN_PATH)
    features = [c for c in train.columns if c not in ("name", "label")]
    y = train["label"].astype(int).to_numpy()

    keys = defaultdict(list)
    for i, row in enumerate(train[features].itertuples(index=False, name=None)):
        k = tuple(int(v) for v in row)
        keys[k].append(i)

    weak_seen = []
    for k, idxs in keys.items():
        cnt = len(idxs)
        if cnt >= 30: continue
        lbls = y[idxs]
        cc = np.bincount(lbls, minlength=3) / cnt
        ent = entropy(cc)
        bkt = "1-2" if cnt <= 2 else ("3-9" if cnt <= 9 else "10-29")
        weak_seen.append({"key": k, "count": cnt, "entropy": ent, "bucket": bkt})

    strata = defaultdict(list)
    for wk in weak_seen:
        label = "high_ent" if wk["entropy"] > 0.5 else "low_ent"
        strata[f"{wk['bucket']}_{label}"].append(wk)

    print("Strata:")
    for s, items in sorted(strata.items()):
        print(f"  {s}: {len(items)} keys")

    canary_ws_keys = []
    total = len(weak_seen)
    for s, items in strata.items():
        n = max(1, int(150 * len(items) / total))
        chosen = RNG.choice([w["key"] for w in items], size=min(n, len(items)), replace=False)
        canary_ws_keys.extend(chosen)

    canary_ws_indices = []
    for k in canary_ws_keys:
        canary_ws_indices.extend(keys[tuple(k)])
    canary_ws_indices = sorted(set(int(i) for i in canary_ws_indices))

    print(f"\nWeak-seen canary: {len(canary_ws_keys)} keys, {len(canary_ws_indices)} samples")

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)): return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super().default(obj)

    with open(SPLIT_DIR / "canary_splits.json", encoding="utf-8") as f:
        meta = json.load(f)
    meta["weak_seen_indices"] = [int(x) for x in canary_ws_indices]
    meta["weak_seen_keys"] = [list(k) for k in canary_ws_keys]
    with open(SPLIT_DIR / "canary_splits.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, cls=NpEncoder, ensure_ascii=False, indent=2)
    print("Updated canary_splits.json")


if __name__ == "__main__":
    main()
