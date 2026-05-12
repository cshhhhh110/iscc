# 复现说明

## 环境

- Python 3.10，CUDA GPU（可选，CPU 也能跑但慢）
- 依赖：`pip install torch numpy scikit-learn scipy joblib tqdm pandas`

## 复现步骤

### 1. 构建稠密特征（~10 分钟）

```bash
python 源码/build_dense_features.py --train-file train.csv
```

### 2. 训练 3 个种子模型（~4.5 小时，需 GPU）

```bash
python 源码/train_nn_v2.py --seed 20260504 --model-path 模型/model_s0.joblib
python 源码/train_nn_v2.py --seed 42 --model-path 模型/model_s1.joblib
python 源码/train_nn_v2.py --seed 123 --model-path 模型/model_s2.joblib
```

### 3. 集成预测 + 阈值调优（~5 分钟）

```bash
python 源码/predict_ensemble_v2.py --models 模型/model_s0.joblib 模型/model_s1.joblib 模型/model_s2.joblib --tune
```

输出 → `提交结果/submission_ensemble_v2.csv`

## 技术要点

| 组件 | 说明 |
|------|------|
| 特征 | HashingVectorizer(char+word) SVD 降维到 300 维 |
| 模型 | BiLSTM 2层 256 hidden, argmax 解码 |
| 集成 | 3 seed 概率平均 |
| 解码 | 每类随机阈值 + 平滑 + 间隙填充 + 边界微调 |
| 调优 | 300 次联合随机搜索 |
