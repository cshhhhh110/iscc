"""
v2.0 — 可复现 FT-Transformer 多 seed 集成

确定性训练: cudnn.benchmark=False, 固定 seed → 任何人运行得到完全相同的提交文件

用法:
  python 源码/train_v2_0.py train    # 训练 + 保存模型
  python 源码/train_v2_0.py predict  # 预测 + 输出提交文件
  python 源码/train_v2_0.py all      # 一键完成
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ── 超参数 (v1.1/v1.4 同款) ─────────────────────────────────────────

SEEDS = [20260504, 20260505, 20260506]   # 3 seeds × 5 folds = 15 模型
FOLDS = 5
EPOCHS = 90
BATCH = 1024
PATIENCE = 14
LR = 8e-4
WEIGHT_DECAY = 2e-4
LABEL_SMOOTH = 0.03

# FT-Transformer 架构
D_TOKEN = 64
N_BLOCKS = 4
N_HEADS = 8
D_FFN = 192
ATTN_DROP = 0.10
RES_DROP = 0.05
FFN_DROP = 0.10
HEAD_DROP = 0.15

ROOT = Path(__file__).resolve().parents[1]
ID_COL, LABEL_COL = "id", "label"

# ── 确定性环境 ────────────────────────────────────────────────────────

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

def fix_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ── 模型 ───────────────────────────────────────────────────────────

class GEGLU(nn.Module):
    def forward(self, x): v, g = x.chunk(2, -1); return v * torch.nn.functional.gelu(g)

class FeatureTokenizer(nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.w = nn.Parameter(torch.empty(n, d)); self.b = nn.Parameter(torch.empty(n, d))
        nn.init.normal_(self.w, 0, 0.02); nn.init.normal_(self.b, 0, 0.02)
    def forward(self, x): return x.unsqueeze(-1) * self.w + self.b

class Block(nn.Module):
    def __init__(self, d, h, ff, ad, rd, fd):
        super().__init__()
        self.an = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=ad, batch_first=True)
        self.ad = nn.Dropout(rd)
        self.fn = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ff*2), GEGLU(), nn.Dropout(fd), nn.Linear(ff, d))
        self.fd = nn.Dropout(rd)
    def forward(self, x):
        h = self.an(x); h, _ = self.attn(h, h, h, need_weights=False); x = x + self.ad(h)
        return x + self.fd(self.ffn(self.fn(x)))

class FTTransformer(nn.Module):
    def __init__(self, n_feat, n_cls):
        super().__init__()
        self.tok = FeatureTokenizer(n_feat, D_TOKEN)
        self.cls = nn.Parameter(torch.zeros(1, 1, D_TOKEN))
        self.blocks = nn.Sequential(*[Block(D_TOKEN, N_HEADS, D_FFN, ATTN_DROP, RES_DROP, FFN_DROP) for _ in range(N_BLOCKS)])
        self.head = nn.Sequential(nn.LayerNorm(D_TOKEN*2), nn.Linear(D_TOKEN*2, D_TOKEN*2), nn.GELU(), nn.Dropout(HEAD_DROP), nn.Linear(D_TOKEN*2, n_cls))
        nn.init.normal_(self.cls, 0, 0.02)
    def forward(self, x):
        t = self.tok(x); c = self.cls.expand(len(x), -1, -1)
        t = torch.cat([c, t], 1); t = self.blocks(t)
        return self.head(torch.cat([t[:,0], t[:,1:].mean(1)], 1))

# ── 预处理 ──────────────────────────────────────────────────────────

def robust_stats(x):
    c = np.nanmedian(x, 0, keepdims=True).astype(np.float32)
    s = (np.nanpercentile(x, 75, 0, keepdims=True) - np.nanpercentile(x, 25, 0, keepdims=True)).astype(np.float32)
    fb = np.nanstd(x, 0, keepdims=True).astype(np.float32)
    s = np.where(s < 1e-6, fb, s); s = np.where(s < 1e-6, 1.0, s)
    return c, s

def apply_robust(x, c, s):
    x = (x - c) / s; x = np.nan_to_num(x, nan=0, posinf=8, neginf=-8)
    return np.clip(x, -8, 8).astype(np.float32)

# ── 推理 ───────────────────────────────────────────────────────────

@torch.inference_mode()
def get_probs(model, x, device):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(x)), BATCH*2, shuffle=False, num_workers=0, pin_memory=device.type=="cuda")
    chunks = []
    for (bx,) in loader:
        with torch.amp.autocast(device.type, enabled=device.type=="cuda"):
            chunks.append(torch.softmax(model(bx.to(device)), -1).cpu().numpy().astype(np.float32))
    return np.concatenate(chunks)

# ── 单折训练 ────────────────────────────────────────────────────────

def train_fold(seed, fold, x_tr, x_va, x_te, y_tr, y_va, n_cls, device):
    fix_seed(seed)
    model = FTTransformer(x_tr.shape[1], n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    crit = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type=="cuda")

    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)),
                        BATCH, shuffle=True, num_workers=0, pin_memory=device.type=="cuda")

    best, best_ep, best_acc, best_mf1, bad = None, 0, -1.0, -1.0, 0

    for ep in tqdm(range(1, EPOCHS+1), desc=f"  fold {fold}", unit="ep", leave=False):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=device.type=="cuda"):
                loss = crit(model(bx), by)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        sch.step()

        vp = get_probs(model, x_va, device); vp = np.argmax(vp, 1)
        acc = float(accuracy_score(y_va, vp))
        mf1 = float(f1_score(y_va, vp, average="macro"))

        if acc > best_acc + 1e-12 or (abs(acc-best_acc) <= 1e-12 and mf1 > best_mf1 + 1e-12):
            best_acc, best_mf1, best_ep = acc, mf1, ep
            best = deepcopy({k: v.cpu() for k, v in model.state_dict().items()}); bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break

    model.load_state_dict(best)
    va_p = get_probs(model, x_va, device)
    te_p = get_probs(model, x_te, device)
    del model; return va_p, te_p, best_mf1, best_acc, best_ep

# ── 主流程 ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", nargs="?", default="all", choices=["train", "predict", "all"])
    p.add_argument("--data-dir", default=".")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.set_float32_matmul_precision("high")

    data_dir = (ROOT / args.data_dir).resolve()

    if args.action in ("train", "all"):
        train_df = pd.read_csv(data_dir / "train_data.csv")
        test_df = pd.read_csv(data_dir / "test_data.csv")
        feats = [c for c in train_df.columns if c not in (ID_COL, LABEL_COL)]

        le = LabelEncoder(); y = le.fit_transform(train_df[LABEL_COL].astype(str))
        lbls = le.classes_.tolist(); n_cls = len(lbls)

        x_all = train_df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy()
        x_test = test_df[feats].apply(pd.to_numeric, errors="coerce").astype(np.float32).to_numpy()

        n_runs = len(SEEDS) * FOLDS
        oof = np.zeros((len(x_all), n_cls), dtype=np.float32)
        tst = np.zeros((len(x_test), n_cls), dtype=np.float32)

        print(f"v2.0 | train={len(x_all)} test={len(x_test)} feat={len(feats)} class={n_cls}")
        print(f"seeds={SEEDS} folds={FOLDS} total_models={n_runs}")
        print(f"deterministic={torch.backends.cudnn.deterministic}")
        t0 = time.perf_counter()

        for seed in SEEDS:
            cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
            for fold, (tr, va) in enumerate(cv.split(x_all, y), 1):
                c, s = robust_stats(x_all[tr])
                x_tr = apply_robust(x_all[tr], c, s)
                x_va = apply_robust(x_all[va], c, s)
                x_te = apply_robust(x_test, c, s)
                y_tr = y[tr].astype(np.int64); y_va = y[va].astype(np.int64)

                fold_seed = seed + fold * 1000
                va_p, te_p, mf1, acc, ep = train_fold(fold_seed, fold, x_tr, x_va, x_te, y_tr, y_va, n_cls, device)
                oof[va] += va_p / len(SEEDS)
                tst += te_p / n_runs
                tqdm.write(f"  seed={seed} fold={fold}: mf1={mf1:.4f} acc={acc:.4f} ep={ep}")

        elapsed = time.perf_counter() - t0
        oof_mf1 = float(f1_score(y, np.argmax(oof, 1), average="macro"))
        oof_acc = float(accuracy_score(y, np.argmax(oof, 1)))
        per_class = {l: float(s) for l, s in zip(lbls, f1_score(y, np.argmax(oof, 1), average=None))}
        print(f"\nDone {elapsed/60:.1f}min | OOF mf1={oof_mf1:.4f} acc={oof_acc:.4f} | weak={min(per_class, key=per_class.get)}:{min(per_class.values()):.4f}")

        # 保存 test probs 和模型参数
        model_dir = ROOT / "模型"; model_dir.mkdir(parents=True, exist_ok=True)
        np.save(model_dir / "test_probs_v2_0.npy", tst)
        torch.save({
            "version": "v2.0", "seeds": SEEDS, "folds": FOLDS, "feats": feats, "labels": lbls,
            "oof_mf1": oof_mf1, "oof_acc": oof_acc, "per_class_f1": per_class,
        }, model_dir / "bundle_v2_0.pt")
        print(f"Saved: {model_dir / 'test_probs_v2_0.npy'}, {model_dir / 'bundle_v2_0.pt'}")

    if args.action in ("predict", "all"):
        test_df = pd.read_csv(data_dir / "test_data.csv")
        bundle = torch.load(ROOT / "模型" / "bundle_v2_0.pt", map_location="cpu", weights_only=False)
        tst = np.load(ROOT / "模型" / "test_probs_v2_0.npy")

        lbls = bundle["labels"]
        pred = np.asarray(lbls)[np.argmax(tst, 1)]
        sub = pd.DataFrame({ID_COL: test_df[ID_COL], LABEL_COL: pred})

        out_dir = ROOT / "提交结果"; out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "submission_v2_0.csv"
        sub.to_csv(out_path, index=False)
        print(f"Wrote: {out_path} | rows={len(sub)} | classes={len(set(pred))}")

        # 校验
        from sklearn.preprocessing import LabelEncoder as LE
        allowed = set(bundle["labels"])
        assert list(sub.columns) == ["id", "label"], "列名不对"
        assert len(sub) == len(test_df), "行数不对"
        assert sub["id"].tolist() == test_df["id"].tolist(), "id 不对"
        assert not sub["label"].isna().any(), "有空标签"
        invalid = set(sub["label"]) - allowed
        assert not invalid, f"非法标签: {invalid}"
        print("验证通过")


if __name__ == "__main__":
    main()
