# 二进制文件漏洞检测

ISCC 数据安全赛道 — 给定 PE 二进制文件，判断漏洞标签 (0/1) 和 CWE 类别 (86 类)。  
评分：87 类 Macro-F1（含 label=0 "无漏洞"）。

---

## 架构

```
PE Binary (.exe)
    │
    ├── Tabular 分支 (674 + 87×3 = 935 维)
    │   ├── PE 头统计 (56维)
    │   ├── 字节直方图 (256维)
    │   ├── 节/DLL/API/关键字 (194维)
    │   ├── Capstone 反汇编 — opcode/寄存器/函数/栈帧 (228维)
    │   └── CWE 字节 N-gram 词典匹配 (87×3=261维) ← v3.0 新增
    │   → LightGBM GBDT (多分类)
    │
    ├── Byte 分支 (8192 字节窗口)
    │   → 5层 Conv1D (64→128→192→256→320) + residual
    │   → ByteMetaMultiTaskNet (双头: label + CWE)
    │
    └── 3-seed [42, 123, 202] 加权集成
            │
            ▼
    Scalar 融合 → 轻量元模型修正 (v3.0)
            │
            ▼
    submission.csv
```

## 关键设计

### 特征工程

| 类别 | 维度 | 内容 |
|------|------|------|
| PE 基础 | 59 | 文件大小、熵、字节统计、PE 头字段 |
| 字节直方图 | 256 | 0-255 字节频率分布 |
| 节/DLL/API/关键字 | 194 | 节信息、导入 DLL/API、字符串关键词 |
| Capstone 反汇编 | 228 | opcode unigram/bigram、寄存器热图、函数边界、操作数/指令统计 |
| **CWE 字节 N-gram 词典** | **261** | **每类 CWE 50 个 TF-IDF 高频字节模式匹配计数** ← v3.0 |

### 神经网络
- **ByteMetaMultiTaskNet**: 5 层 Conv1D + residual + 双头输出
- **FocalLoss** (γ=1.5) 处理 CWE 类不平衡
- **CosineAnnealingWarmRestarts** + Warmup + SWA

### 元模型 (v3.0)
- **小型 LightGBM** (num_leaves=15, max_depth=4, strong reg)
- **输入**: OOF 集成预测 (label_prob + cwe_probs) + N-gram 词典特征 + 文件大小
- **训练**: 仅用 seed-wise OOF 样本 + 原始训练集 (排除伪标签)
- **作用**: 用领域知识 (字节模式) 修正集成预测的 CWE 错误

### 伪标签
- 17,343 条 0.9 阈值伪标签，扩大训练集 (39K → 57K)

---

## 成绩

| Version | Platform (Macro-F1) | 关键改动 |
|---------|---------------------|----------|
| v1.3 | 0.849 | 基线：LightGBM + NGram |
| v1.4 | 0.901 | Scalar fusion (修复阈值 bug) |
| v1.5 | 0.923 | 3-seed 集成 + 伪标签 + TTA |
| v1.9 | 0.921 | CosineAnnealing + warmup + SWA |
| v2.3 | 0.923 | **Capstone 反汇编 (192维)** |
| **v2.4** | **0.925** 🏆 | **扩展反汇编 (228维) + 种子 456→202** |
| v2.5 | 0.912 | Per-class fusion (过拟合，失败) |
| v2.6 | 0.599 | Trigram hash 用 Python hash() → 非确定性 (已修复) |
| **v3.0** | **待提交** | **CWE 字节 N-gram 词典 + 轻量元模型修正** |

---

## 运行

```bash
# 1. 构建 CWE N-gram 词典 (新)
cd 二进制文件漏洞检测
python 源码/cwe_ngram_features.py --build

# 2. 训练 (3 seeds + 元模型)
python 源码/train.py
# → 模型/cwe_ngram_dict_v3.0.json
# → 模型/cwe_meta_model_v3.0.joblib
# → 模型/*_v3.0_*.joblib / .pt

# 3. 预测
python 源码/test.py
# → 提交结果/submission_v3.0.csv
```

训练时间 (RTX 4060 8GB): 首次 ~6h (含特征提取 + N-gram 词典); 有缓存后 ~1.5h。

---

## 踩过的坑

| 坑 | 教训 |
|----|------|
| Python `hash()` 非确定性 | FNV-1a 替代; `PYTHONHASHSEED` 设置无效 (必须在进程启动前) |
| Per-class fusion 过拟合 | 稀有类 1-5 样本下权重是噪声 → 全局 scalar 更好 |
| Seed 456 | 6 版未超 0.873 → 换 202 |
| 伪标签阈值一刀切 | 后续可选 per-class margin 过滤 |
| Capstone SSL 安装 | `--trusted-host pypi.tuna.tsinghua.edu.cn` |
| .cmd GBK 编码 | 中文路径需 GBK 或 `chcp 65001` |

---

## 文件清单

```
二进制文件漏洞检测/
├── 源码/
│   ├── train.py              # 训练主流程
│   ├── test.py               # 预测入口
│   ├── features.py           # 特征提取 (PE + 字节 + 反汇编 + N-gram)
│   ├── cwe_ngram_features.py # CWE 字节 N-gram 词典 (v3.0)
│   ├── disasm_features.py    # Capstone 反汇编特征
│   ├── nn_models.py          # 神经网络模型
│   ├── byte_features.py      # 字节窗口提取
│   ├── dataset.py            # 数据加载
│   ├── models.py             # 命名规范
│   └── utils.py              # 工具函数
├── 模型/                     # 训练产出
├── 提交结果/                 # 预测产出
├── 版本记录/                 # 历史版本归档
└── README.md
```
