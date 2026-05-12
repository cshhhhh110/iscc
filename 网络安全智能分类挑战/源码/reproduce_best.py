"""
复现 best submission (blend_top2_ft = 0.709)

原理: 训练两组 FT-Transformer 模型, 对测试集概率做 50:50 平均
  Group A: seed=[20260504], 5 folds → 模拟 v1.1 (平台 0.708)
  Group B: seed=[20260504, 20260505], 5 folds each → 模拟 v1.4 (平台 0.707)

用法:
  训练:  python 源码/reproduce_best.py train
  预测:  python 源码/reproduce_best.py predict
  一键:  python 源码/reproduce_best.py all

  输出: 提交结果/blend_top2_ft.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── Model definition (same as v1.1/v1.4) ───────────────────────────────

class GEGLU(nn.Module):
    def forward(self, x): return x.chunk(2, dim=-1)[0] * torch.nn.functional.gelu(x.chunk(2, dim=-1)[1])

class NumericFeatureTokenizer(nn.Module):
    def __init__(self, n_features, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.normal_(self.weight, 0, 0.02); nn.init.normal_(self.bias, 0, 0.02)

    def forward(self, x): return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

class TransformerBlock(nn.Module):
    def __init__(self, d_token, n_heads, d_ffn, attn_drop, res_drop, ffn_drop):
        super().__init__()
        self.attn_norm = nn.LayerNorm(d_token)
        self.attn = nn.MultiheadAttention(d_token, n_heads, dropout=attn_drop, batch_first=True)
        self.attn_dropout = nn.Dropout(res_drop)
        self.ffn_norm = nn.LayerNorm(d_token)
        self.ffn = nn.Sequential(nn.Linear(d_token, d_ffn*2), GEGLU(), nn.Dropout(ffn_drop), nn.Linear(d_ffn, d_token))
        self.ffn_dropout = nn.Dropout(res_drop)

    def forward(self, x):
        h = self.attn_norm(x); h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.attn_dropout(h); h = self.ffn(self.ffn_norm(x)); return x + self.ffn_dropout(h)

class FTTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_token=64, n_blocks=4, n_heads=8, d_ffn=192,
                 attn_drop=0.10, res_drop=0.05, ffn_drop=0.10, head_drop=0.15):
        super().__init__()
        self.tokenizer = NumericFeatureTokenizer(n_features, d_token)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks = nn.Sequential(*[TransformerBlock(d_token, n_heads, d_ffn, attn_drop, res_drop, ffn_drop) for _ in range(n_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d_token*2), nn.Linear(d_token*2, d_token*2), nn.GELU(), nn.Dropout(head_drop), nn.Linear(d_token*2, n_classes))
        nn.init.normal_(self.cls_token, 0, 0.02)

    def forward(self, x):
        tokens = self.tokenizer(x); cls = self.cls_token.expand(len(x), -1, -1)
        tokens = torch.cat([cls, tokens], dim=1); tokens = self.blocks(tokens)
        return self.head(torch.cat([tokens[:, 0], tokens[:, 1:].mean(dim=1)], dim=1))


# ── Utilities ───────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
LABEL_COL = "label"; ID_COL = "id"
CONFIG = {"d_token": 64, "n_blocks": 4, "n_heads": 8, "d_ffn": 192,
          "attn_drop": 0.10, "res_drop": 0.05, "ffn_drop": 0.10, "head_drop": 0.15}

GROUP_A_SEEDS = [20260504]                                    # v1.1: 1 seed, 5 folds
GROUP_B_SEEDS = [20260504, 20260505]                          # v1.4: 2 seeds, 5 folds each
FOLDS = 5; EPOCHS = 90; BATCH_SIZE = 1024; PATIENCE = 14
LR = 8e-4; WEIGHT_DECAY = 2e-4; LABEL_SMOOTHING = 0.03


def set_seed(s): import random; random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def fit_robust_stats(x):
    c = np.nanmedian(x, 0, keepdims=True).astype(np.float32)
    s = (np.nanpercentile(x, 75, 0, keepdims=True) - np.nanpercentile(x, 25, 0, keepdims=True)).astype(np.float32)
    fb = np.nanstd(x, 0, keepdims=True).astype(np.float32)
    s = np.where(s < 1e-6, fb, s); s = np.where(s < 1e-6, 1.0, s); return c, s


def apply_robust(x, c, s):
    x = (x - c) / s; x = np.nan_to_num(x, nan=0, posinf=8, neginf=-8); return np.clip(x, -8, 8).astype(np.float32)


def predict_probs(model, x, device):
    model.eval(); loader = DataLoader(TensorDataset(torch.from_numpy(x)), BATCH_SIZE*2, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
    chunks = []
    with torch.inference_mode():
        for (bx,) in loader:
            with torch.amp.autocast(device.type, enabled=device.type=="cuda"):
                logits = model(bx.to(device))
            chunks.append(torch.softmax(logits, -1).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)


def train_group(seeds, x_train, y, x_test, n_classes, device):
    """Train FT-Transformer for a group of seeds. Returns test probs average."""
    n_features = x_train.shape[1]
    total = len(seeds) * FOLDS
    oof = np.zeros((len(x_train), n_classes), dtype=np.float32)
    test = np.zeros((len(x_test), n_classes), dtype=np.float32)

    for seed in seeds:
        cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for fold_idx, (tr, va) in enumerate(cv.split(x_train, y), 1):
            set_seed(seed + fold_idx * 1000)
            c, s = fit_robust_stats(x_train[tr])
            x_tr = apply_robust(x_train[tr], c, s); x_va = apply_robust(x_train[va], c, s)
            x_te = apply_robust(x_test, c, s)
            y_tr = y[tr].astype(np.int64); y_va = y[va].astype(np.int64)

            loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), BATCH_SIZE, shuffle=True, pin_memory=device.type=="cuda")

            model = FTTransformer(n_features, n_classes, **CONFIG).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")

            best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            best_acc, best_mf1, bad, best_ep = -1.0, -1.0, 0, 0

            bar = tqdm(range(1, EPOCHS+1), desc=f"  fold {fold_idx}/{FOLDS}", unit="ep", leave=False)
            for ep in bar:
                model.train(); seen = 0
                for bx, by in loader:
                    bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device.type, enabled=device.type=="cuda"):
                        loss = crit(model(bx), by)
                    scaler.scale(loss).backward(); scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); seen += len(by)
                sch.step()

                vp = predict_probs(model, x_va, device)
                va_pred = np.argmax(vp, 1)
                va_acc = float(accuracy_score(y_va, va_pred))
                va_mf1 = float(f1_score(y_va, va_pred, average="macro"))

                if va_acc > best_acc + 1e-12 or (abs(va_acc - best_acc) <= 1e-12 and va_mf1 > best_mf1 + 1e-12):
                    best_acc, best_mf1, best_ep = va_acc, va_mf1, ep
                    best_state = deepcopy({k: v.cpu() for k, v in model.state_dict().items()}); bad = 0
                else:
                    bad += 1
                bar.set_postfix(acc=f"{va_acc:.4f}", mf1=f"{va_mf1:.4f}", best=f"{best_mf1:.4f}", pat=f"{bad}/{PATIENCE}")
                if bad >= PATIENCE: break

            model.load_state_dict(best_state)
            oof[va] += predict_probs(model, x_va, device) / len(seeds)
            test += predict_probs(model, x_te, device) / total
            tqdm.write(f"  seed={seed} fold={fold_idx}: mf1={best_mf1:.4f} acc={best_acc:.4f} ep={best_ep}")

    return oof, test


def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision("high")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    else:
        device = torch.device("cpu")

    data_dir = (ROOT / args.data_dir).resolve()
    train_df = pd.read_csv(data_dir / "train_data.csv")
    test_df = pd.read_csv(data_dir / "test_data.csv")
    feat_cols = [c for c in train_df.columns if c not in (ID_COL, LABEL_COL)]
    le = __import__('sklearn.preprocessing', fromlist=['LabelEncoder']).LabelEncoder()
    y = le.fit_transform(train_df[LABEL_COL].astype(str))
    label_names = le.classes_.tolist(); n_classes = len(label_names)

    x_train = train_df[feat_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32).to_numpy()
    x_test = test_df[feat_cols].apply(pd.to_numeric, errors='coerce').astype(np.float32).to_numpy()

    print(f"Train: {len(x_train)} rows, Test: {len(x_test)} rows, Features: {len(feat_cols)}, Classes: {n_classes}")
    print(f"Group A seeds={GROUP_A_SEEDS} (simulates v1.1, platform 0.708)")
    print(f"Group B seeds={GROUP_B_SEEDS} (simulates v1.4, platform 0.707)")

    t0 = time.perf_counter()

    print("\n=== Group A (v1.1-style) ===")
    oof_a, test_a = train_group(GROUP_A_SEEDS, x_train, y, x_test, n_classes, device)
    oof_a_mf1 = float(f1_score(y, np.argmax(oof_a, 1), average="macro"))
    print(f"Group A OOF macro_f1 = {oof_a_mf1:.4f}")

    print("\n=== Group B (v1.4-style) ===")
    oof_b, test_b = train_group(GROUP_B_SEEDS, x_train, y, x_test, n_classes, device)
    oof_b_mf1 = float(f1_score(y, np.argmax(oof_b, 1), average="macro"))
    print(f"Group B OOF macro_f1 = {oof_b_mf1:.4f}")

    # ── Blend: 50/50 ──
    test_blend = (test_a + test_b) / 2.0
    labels = np.asarray(label_names)[np.argmax(test_blend, 1)]
    submission = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: labels})

    out_dir = ROOT / "提交结果"; out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "blend_top2_ft.csv"
    with out_path.open("w", encoding="utf-8", newline="") as f:
        submission.to_csv(f, index=False)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed/60:.1f} min")
    print(f"Wrote: {out_path}")
    print(f"Validate: python 源码/validate_submission.py --submission-path {out_path}")


def predict(args):
    """Not needed — train() already outputs the final submission. This is a no-op for CLI consistency."""
    print("Prediction is already done during training. See 提交结果/blend_top2_ft.csv")


def main():
    p = argparse.ArgumentParser(description="Reproduce blend_top2_ft (0.709)")
    p.add_argument("action", nargs="?", default="all", choices=["train", "predict", "all"])
    p.add_argument("--data-dir", default=".", help="Data directory (default: project root)")
    args = p.parse_args()

    if args.action in ("train", "all"):
        train(args)

    if args.action in ("predict",):
        predict(args)


if __name__ == "__main__":
    main()
