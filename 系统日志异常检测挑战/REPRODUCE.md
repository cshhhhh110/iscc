# 复现说明 (v1.7)

## 环境

Python 3.10 + CUDA GPU

```bash
pip install torch numpy scikit-learn scipy joblib tqdm pandas
```

## 复现步骤

### 1. 构建稠密特征（~10 分钟）

```bash
python 源码/build_dense_features.py --train-file train.csv
```

### 2. 训练 3 个种子模型（~4.5 小时，可并行）

```bash
python 源码/train_nn_v2.py --seed 20260504
python 源码/train_nn_v2.py --seed 42
python 源码/train_nn_v2.py --seed 123
```

每次训完改名，或通过 `--model-path` 指定不同路径。

### 3. 集成预测 + 阈值调优（~5 分钟）

```bash
python 源码/predict_ensemble_v2.py --tune --models 模型/model_s0.joblib 模型/model_s1.joblib 模型/model_s2.joblib
```

输出 → `提交结果/submission_ensemble_v2.csv`

## 技术要点

| 组件 | 说明 |
|------|------|
| 特征 | HashingVectorizer(char_wb 3-5 + word 1-2) SVD 降维到 300 维 |
| 模型 | BiLSTM 2层 256 hidden, CE 损失, AMP 混合精度 |
| 集成 | 3 seed 概率平均 |
| 解码 | 每类独立阈值联合搜索 + 平滑 + 间隙填充 + 每类95分位跨度上限 + 边界微调 |
| 跨度 | 每类从训练集自动统计 95 分位 max_len，不再共享全局上限 |

## 预期成绩

平台分数 ~0.915。因 AMP 浮点精度和阈值随机搜索，复现结果在 0.912-0.917 间波动属正常。
