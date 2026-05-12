# 复现说明 (v1.6)

## 环境

- Python 3.10，CUDA GPU
- `pip install torch numpy scikit-learn scipy joblib tqdm pandas`

## 复现步骤

### 1. 构建稠密特征（~10 分钟）

```bash
python 源码/build_dense_features.py --train-file train.csv
```

### 2. 训练 5 个种子模型（~9 小时，可多机并行）

```bash
python 源码/train_nn_v2.py --seed 20260504 --model-path 模型/model_s0.joblib
python 源码/train_nn_v2.py --seed 42 --model-path 模型/model_s1.joblib
python 源码/train_nn_v2.py --seed 123 --model-path 模型/model_s2.joblib
python 源码/train_nn_v2.py --seed 999 --model-path 模型/model_s3.joblib
python 源码/train_nn_v2.py --seed 777 --model-path 模型/model_s4.joblib
```

### 3. 集成预测 + 阈值调优（~5 分钟）

```bash
python 源码/predict_ensemble_v2.py --tune --models 模型/model_s0.joblib 模型/model_s1.joblib 模型/model_s2.joblib 模型/model_s3.joblib 模型/model_s4.joblib
```

输出 → `提交结果/submission_ensemble_v2.csv`

## 技术要点

| 组件 | 说明 |
|------|------|
| 特征 | HashingVectorizer(char+word) SVD 降维到 300 维 |
| 模型 | BiLSTM 2层 256 hidden, 长度加权 CE 损失, AMP |
| 集成 | 5 seed 概率平均 |
| 解码 | 每类随机阈值联合搜索 + 平滑 + 间隙填充 + 长文档放宽 + 边界微调 |
