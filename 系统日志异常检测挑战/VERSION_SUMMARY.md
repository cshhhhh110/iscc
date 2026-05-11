# 系统日志异常检测 — 版本总结

## 评分公式

```
Score = 0.15 × F1_detect + 0.50 × IoU_loc + 0.35 × F1_type
```

| 指标 | 含义 | 权重 |
|------|------|------|
| F1_detect | 文档级二分类（有/无异常） | 15% |
| IoU_loc | 异常区间定位重叠度 | **50%** |
| F1_type | 异常类型分类（10类） | 35% |

## 版本演进

| 版本 | 模型 | 核心思路 | 本地 OOF | 平台分数 | 变化 |
|------|------|----------|----------|----------|------|
| v1.0 | SGD | 初始基线 | 0.686 | — | — |
| v1.1 | SGD | 完善特征+阈值搜索 | 0.9225 | 0.826 | 基线 |
| v1.2 | 多尝试 | LightGBM/XGBoost/校准 | — | — | 全部失败 |
| v1.3 | SGD | 扩特征+350阈值搜索 | 0.9243 | 0.829 | +0.003 |
| **v1.4** | **BiLSTM v2** | **SVD降维+序列建模** | **0.9565** | **0.885** | **+0.056** |

## v1.1–v1.3 架构（SGD 时代）

```
原始日志文本
  → HashingVectorizer(char_wb + word), 524K 维稀疏特征
  → SGDClassifier(log_loss) 逐行预测分数
  → 启发式解码器: 平滑 → 阈值 → 间隙填充 → 连通分量 → 最佳子区间
  → 350 次随机搜索调 10 类独立阈值
```

**优势：** SGD 在稀疏特征上 O(nnz) 极快，524K 维下 CPU 可训练
**劣势：**
- 每行独立预测，无序列上下文（第 5 行和第 50 行对模型一样）
- SGD 概率极端饱和（接近 0 或 1），阈值调不动
- 解码器是启发式的，不做端到端学习
- 524K 维中大量噪声维度，过拟合风险高

## v1.4 架构（BiLSTM v2）

```
原始日志文本
  → HashingVectorizer(char_wb, 262K) + HashingVectorizer(word, 262K)
  → TruncatedSVD(128) 各自降维 → 128 + 128 = 256 维
  → + 44 维数值特征（位置、时间戳、关键词、字符统计）
  → 300 维稠密行向量
  → BiLSTM(2层, 256 hidden, 双向) → 512 维上下文编码
  → 行分类头: Linear(512→11) → argmax → 行标签 (O + 10 类异常)
  → 文档分类头: 注意力池化 → Linear(512→1) → has_anomaly
  → 解码: 连续同标签行 → 区间
```

### 与 v1.3 的关键区别

| | v1.3 SGD | v1.4 BiLSTM v2 |
|---|---|---|
| 输入维度 | 524K 稀疏 | 300 稠密（SVD） |
| 序列感知 | 无 | 双向 LSTM |
| 行级分类 | 10 个独立二分类器 | 单头 11 类 softmax |
| 文档级分类 | 独立 SGD | 注意力池化共享编码器 |
| 解码器 | 启发式 350 次随机搜 | argmax 直接输出 |
| 参数量 | ~5M（稀疏） | ~2M（稠密） |
| 训练时间 | ~3.5h CPU | ~2h GPU |

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR |
| 梯度裁剪 | max_norm=1.0 |
| Early Stopping | patience=8 |
| CV | 5-fold StratifiedKFold |
| 全量重训 | 25 epochs（CV 中位数） |
| Dropout | 0.35 |
| Batch Size | 32 |

### 5 折验证结果

| Fold | Best Epoch | Val Score |
|------|-----------|-----------|
| 1 | 23 | 0.9498 |
| 2 | 25 | 0.9597 |
| 3 | 19 | 0.9600 |
| 4 | 30 | 0.9527 |
| 5 | 30 | 0.9604 |
| **OOF** | **25 (median)** | **0.9565** |

### OOF 明细

| 指标 | 值 |
|------|-----|
| F1_detect | 0.9999 |
| IoU_loc | 0.9150 |
| F1_type | 0.9972 |

## 失败记录

| 尝试 | 问题 | 教训 |
|------|------|------|
| LightGBM | pip 版不支持 CUDA，OpenCL 跑在集显 | LightGBM 不适合此数据规模 |
| XGBoost GPU | 524K 特征 × 1M 行 > 6GB 显存 OOM | 树模型不适合超高维稀疏特征 |
| XGBoost CPU | 太慢（4h+ 仅完成 1 折） | 同上，O(n_features) vs SGD O(nnz) |
| SGD+Isotonic 校准 | 小验证集校准映射不可靠，IoU 崩溃 | 不要校准极端稀疏模型 |
| BiLSTM+CRF (v1) | 特征未归一化+CRF 实现 bug，损失爆炸 | 先做特征工程再做序列模型 |
| decoder 改全局搜索 | 去掉阈值后假阳性爆炸，OOF 0.713 | SGD 极值概率需要阈值过滤 |

## 文件结构

```
系统日志异常检测挑战/
├── run_nn_v2.ps1          # 一键运行脚本
├── 源码/
│   ├── common.py           # 公共函数（特征、解码、评估、IO）
│   ├── cache.py            # 特征缓存（解析+上下文预计算）
│   ├── build_dense_features.py  # SVD 稠密特征构建
│   ├── model_nn_v2.py      # BiLSTM 模型定义
│   ├── train_nn_v2.py      # BiLSTM 训练脚本
│   ├── predict_nn_v2.py    # BiLSTM 预测脚本
│   ├── train.py            # SGD 训练（v1.3，已废弃）
│   ├── predict.py          # SGD 预测（v1.3，已废弃）
│   ├── predict_ensemble.py # 集成预测（未使用）
│   └── pseudo_label.py     # 伪标签生成（未使用）
├── 模型/                   # 模型文件（.joblib 不进 git）
├── 提交结果/               # 提交文件（.csv 进 git）
├── 缓存/                   # 特征缓存（进 git）
└── 版本记录/               # 历史版本完整快照（不进 git）
```

## 运行命令

```powershell
# 一键运行
.\run_nn_v2.ps1

# 或分步运行
python 源码/build_dense_features.py   # 构建稠密特征（~5min CPU）
python 源码/train_nn_v2.py            # 训练（~2h GPU）
python 源码/predict_nn_v2.py          # 预测（~2min GPU）
```

## 下一步方向

1. **伪标签** — 用 v1.4 标注 test.csv，高置信度样本加入训练，预期 +0.01~0.02
2. **多折 Bagging** — 训 3-5 个不同 seed 的模型取平均概率，降低方差
3. **更大的 SVD 维度** — 从 128 扩到 256 或 512，保留更多信号
4. **位置编码** — 加入行位置 embedding，帮助 BiLSTM 感知绝对位置
5. **CRF 解码层** — 在稳定训练基础上加 CRF，约束标签转移一致性
6. **数据增强** — 对日志行做同义词替换/插入/删除，增加训练样本
