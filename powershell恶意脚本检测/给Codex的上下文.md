# PowerShell 恶意脚本检测 — 全量上下文（给 Codex/Grok）

> 这是 ISCC 竞赛的一道题。以下包含所有版本迭代、成绩、代码架构、发现和教训。
> 请完整阅读后再给建议。

---

## 1. 题目定义

- **任务**: PowerShell 脚本三分类
- **类别**: 0=正常脚本, 1=一般恶意脚本, 2=混淆恶意脚本
- **训练集**: data_train.csv, 48,065 条
- **测试集**: data_test.csv, 20,000 条
- **特征**: 15 个低基数离散特征（每个 2-3 个取值），具体列名：
  ```
  function_scope_level, branch_scope_level, loop_scope_level,
  parameter_block_presence, pipeline_usage_level, decode_activity_profile,
  network_command_profile, task_registry_profile, credential_runtime_profile,
  structure_rhythm_profile, layout_variation_profile, identifier_variation_profile,
  content_encoding_profile, command_surface_profile, extension_import_profile
  ```
- **指标**: 平台 Macro-F1（本地只能测 OOF）
- **特殊约束**: 15 维特征所有可能取值组合只有 **1,593 种实际出现**——大量样本共享完全相同的特征模式

---

## 2. 全部版本成绩

| 版本 | OOF | 平台 | gap | 方案 | 状态 |
|------|-----|------|-----|------|------|
| v1.5 | 0.7555 | 0.6910 | 0.064 | 7候选 fusion (ET+HGB+DCN+TabResNet+Pattern+CatBoost+XGB+LGB) | 基线 |
| v1.6 | 0.7555 | 0.6884 | 0.067 | +交互特征(product/ratio/diff)+KMeans+SMOTE+Focal Loss | 全无效 |
| v1.7 | 0.7554 | 0.6978 | 0.058 | +对抗验证(区分train/test) | 无效,但重提交分变高 |
| v1.8 | 0.7519 | 0.6978 | 0.054 | tree-only (ET+HGB) + pseudo-label 自训练 | 持平最高 |
| v1.9 | 0.7554 | 0.6915 | 0.064 | full fusion + pseudo-label | NN是负资产 |
| v1.9 AB | - | 0.690/0.689 | - | fusion裸 vs tree裸 对比实验 | 都不行 |
| v1.10 | 0.7551 | 0.688 | - | tree + pseudo + class2单独降阈值 | 阈值越降越差 |
| v1.11 | 0.7526 | ? | - | hybrid: LGB(40%)+XGB(40%)+RF(20%) 固定权重 | 未提交 |
| v2.0 | - | - | - | mega: 10seed×5fold×2model=100模型集成 | 未提交 |
| v3.0 | 0.7614 | 0.6940 | 0.067 | 参考0.713方案的KD复现(抄歪了) | 复现失败 |
| **参考** | **0.7392** | **0.713** | **0.026** | **别人的0.713方案(朋友给的)** | **目标** |

---

## 3. 核心发现: OOF 高 ≠ 平台高

OOF→平台 gap 是所有问题的根源。

- 我们的 gap 始终在 **0.054~0.067**
- 参考方案的 gap 只有 **0.026**
- 原因: 我们一直用 `StratifiedKFold`（随机切分），同一个特征模式的样本可能同时出现在 train 和 valid 中
- 模型在 valid 上看到的是"训练集里见过的模式"，不是真正的泛化
- 平台测试集有训练集没见过的模式——于是塌陷

**所有试图抬高 OOF 的操作（交互特征、SMOTE、Focal Loss、对抗验证、pseudo-label、NN架构）在平台面前全部失效。我们的 OOF 从 0.75 涨到 0.76，平台纹丝不动。**

---

## 4. 什么有效、什么无效

### 有效
| 方法 | 效果 | 原理 |
|------|------|------|
| StratifiedGroupKFold by 特征key | gap -40% | 同模式整组切，OOF诚实 |
| 模式统计特征(频率/冲突/分布) | +0.01~0.02 | 模型学到"模式可信度"而非死背类别 |
| Knowledge Distillation 软标签 | +0.01~0.02 | 软标签比硬标签平滑，防过拟合 |
| Per-class LGBMRegressor | 稳定 | 3个独立回归器，避免softmax零和 |
| 特征 pairwise 交叉 (c1*10+c2) | +0.005 | 105个二阶组合 |
| GPU加速(CatBoost/XGB/LGB) | 提速3-5x | task_type="GPU", device="cuda", device="gpu" |
| 树模型(LGBM/XGB) | 基线 | 15维类别数据最适合树模型 |
| teacher训练完立刻checkpoint | 省时间 | 崩了不白跑 |
| SafeEmbedding(OOB裁剪) | 零风险 | 防CUDA embedding崩溃 |

### 无效
| 方法 | 原因 |
|------|------|
| 交互特征(product/ratio/diff) | 离散值乘除无意义 |
| KMeans聚类特征 | 离散数据上聚类无信息增益 |
| 对抗验证权重 | train/test分布差异不大 |
| Focal Loss | 不是极度不平衡 |
| NN(DCN/TabResNet/Transformer) | 15维低基数，embedding过参数化 |
| pseudo-label(低阈值,<0.9) | 噪声反噬 |
| 100模型简单平均 | 无结构多样性，只加方差 |
| 融合权重暴力搜索(462×1377=63万组合) | OOF过拟合 |
| class2单独降阈值 | 低置信度样本本身就是错的 |

---

## 5. 参考方案完整分析 (0.713)

### 5.1 架构

```
第0步（离线，代码未提供）: 训练4个Teacher
  3×LGBM + 1×XGB
  ├── SMOTE过采样（每折内）
  ├── StratifiedGroupKFold by 15维特征key
  ├── 超参调优（文件名带"tuned"）
  └── 保存4个.npz OOF文件

第1步（train.py, 190行）: 知识蒸馏
  加载4个.npz → 加权集成 → 温度软化 → 训练Student
  ├── Teacher权重: {"lgbm_tuned_00":0.14273, "lgbm_tuned_01":0.12394, 
  │                  "lgbm_tuned_02":0.34562, "xgb_tuned_00":0.38771}
  ├── 软目标公式: 0.05×exp(log_p/1.0) + 0.95×exp(log_p/3.0) → row-normalize
  ├── 3 Student × 5 Fold × 3 Class = 45 LGBMRegressor
  └── 每个模型存为txt (booster.model_to_string())

第2步（test.py, 65行）: 预测
  加载45个student模型 → 平均 → bias校正 → normalize → CSV
```

### 5.2 特征工程 (common.py, 190行)

```python
15 维原始特征
+ 105 维 pairwise 交叉 (c1*10 + c2)  # 不是product/ratio!
+ pattern count (该特征组合在训练集中出现次数)
+ log1p(pattern count)
+ row sum (该样本各特征值之和)
+ nonzero count
+ label_nunique (该模式对应几个不同类别)
+ is_deterministic (该模式是否100%对应一个类)
+ is_ambiguous (该模式是否对应≥2个类)
+ class_0_fraction (该模式中class0占比)
+ class_1_fraction
+ class_2_fraction
+ pattern_total_count
= ~132 维
```

关键：模式统计特征需要 `key_stats.json`（训练集上预计算的 groupby 统计）。

### 5.3 关键参数

```python
LGBM Teacher: n_estimators=300, lr=0.04, num_leaves=31, min_child_samples=30,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0

XGB Teacher:  n_estimators=300, lr=0.04, max_depth=6, 类似正则

LGBM Student: 同上参数，但用 LGBMRegressor（不是Classifier!）

N_STUDENTS=3, N_FOLDS=5
Final bias: [1.0, 1.0791, 1.0007]
StratifiedGroupKFold random_state=42, shuffle=True
```

### 5.4 为什么它成功

1. **OOF 诚实**: StratifiedGroupKFold 按特征key整组切 → OOF 测的是"没见过的新模式"的泛化能力
2. **KD软标签**: teacher的0.7-0.2-0.1比硬标签1-0-0更平滑 → student不overfit noisy pattern
3. **Per-class回归**: 3个独立LGBMRegressor各自回答"像不像"，避免softmax零和博弈
4. **参数少**: 只搜了4个teacher权重和3个bias值，没做63万组合暴力搜索
5. **模型存文本**: booster.model_to_string() 存为txt，45个模型只占几MB

### 5.5 代码缺失部分

朋友给的代码不全，缺少：
- 另外2个teacher .npz文件 (lgbm_tuned_02, xgb_tuned_00)
- condition_aware/ 目录（45个student模型txt文件）
- Teacher训练脚本（生成.npz的那一步）

---

## 6. 我们v3.0复现失败的原因

v3.0抄参考方案时犯的错误：

1. **没用预训练teacher** — 我们在train.py里当场训teacher，而参考是加载提前训好的.npz
2. **teacher权重抄固定的** — 参考的4个权重(0.14/0.12/0.35/0.39)是针对调参后的teacher优化的，我们的teacher不同
3. **bias抄固定的** — [1.0, 1.0791, 1.0007] 是针对参考预测分布{0:13185,1:3272,2:3543}的
4. **bias搜索范围太小** — [0.9, 1.15]搜到边界0.9，说明最优值可能更低
5. **特征维度少了1维** — 131 vs 132（参考多一个重复的log1p）

---

## 7. 跨机器部署

### 远端
```
主机: LAPTOP-K4UA7J1Q
SSH: 20665@LAPTOP-K4UA7J1Q (密码登录)
GPU: RTX 4060 Laptop 8GB, CUDA 13.1
Conda: D:\anaconda3\envs\iscc-gpu\python.exe (3.10)
工作目录: E:\bisai\
```

### 传输
老机 → 新机:
```bash
# 老机(10.51.165.112)
cd E:\赛题数据 && python -m http.server 8899
```
```bat
:: 新机
curl http://10.51.165.112:8899/xxx.zip -o E:\bisai\xxx.zip
tar -xf E:\bisai\xxx.zip
```

### 新机坑点
- 系统Python 3.8在PATH前面 → pip必须用 `D:\anaconda3\envs\iscc-gpu\python.exe -m pip`
- SSH无GUI → notepad不可用
- 中文路径在SSH传参时被截断 → 优先英文路径
- .cmd必须GBK编码，.ps1用UTF-8
- tar默认解压到C:\Users\20665而非当前目录

### GPU加速关键
```python
CatBoost:  task_type="GPU", devices="0"
XGBoost 3.x: device="cuda"  # 不是tree_method="gpu_hist"!
LightGBM: device="gpu"
# 加上n_jobs=4替代原来的n_jobs=1
```

---

## 8. 当前代码目录

```
E:\赛题数据\powershell恶意脚本检测\
├── 源码/                    ← v1.x~v2.0 (old code)
├── 源码_v3/                 ← v3.0 KD复现
│   ├── common.py            ← 特征工程 + KD工具
│   ├── train.py             ← teacher训练 + KD student
│   └── predict.py           ← 加载模型预测
├── 版本记录/v1.1~v3.0/      ← 完整归档
├── 模型/                    ← 当前模型
├── 提交结果/                 ← 当前提交
├── data_train.csv (48K)
├── data_test.csv  (20K)
├── 经验总结.md              ← 本文档
├── 远端.md                  ← 远端机器信息
├── README.md
└── ACTION_LOG.md
```

## 9. 参考代码位置

```
G:\BaiduNetdiskDownload\powershell\powershell\powershell\
├── 源码（必交）\
│   ├── train.py    (190行)
│   ├── test.py     (65行)
│   └── common.py   (190行)
├── 模型（必交）\
│   ├── metadata.json
│   └── key_stats.json  (大文件, ~240K tokens)
└── 提交结果（必交）\
    └── submission.csv

外加2个.npz:
G:\BaiduNetdiskDownload\powershell\
├── tuned_smoke_group_lgbm_tuned_00.npz (384KB)
└── tuned_smoke_group_lgbm_tuned_01.npz (384KB)
```

---

## 10. 下一步建议

1. 拿到完整的4个teacher .npz + condition_aware/目录 → 直接跑predict复现0.713
2. 或者从零复现：按参考架构写完整的teacher训练+KD student pipeline
3. 核心不动：StratifiedGroupKFold + pairwise特征 + KD + per-class回归
4. bias搜索范围放宽到[0.75, 1.30]，teacher权重也搜
5. 不要加NN、不要搜63万融合权重、不要低阈值pseudo-label
