# 系统日志异常检测挑战 — ISCC 数据安全赛道

## 任务

从系统日志中检测异常区间：给定日志文档，输出是否有异常、异常起始行、结束行、异常类型（10 类）。评分 = 0.15×F1_detect + 0.50×IoU_loc + 0.35×F1_type。

## 版本记录

| 版本 | 平台分 | 方法 |
|------|--------|------|
| v1.3 | 0.829 | SGD 稀疏基线 |
| v1.4 | 0.885 | BiLSTM v2 单 seed |
| v1.5 | 0.908 | 3-seed 集成 + 阈值联合搜索 |
| v1.6 | 0.909 | max_span_len 扩容 |
| **v1.7** | **0.915** | 每类独立 95 分位跨度上限 |

## 快速开始

```bash
pip install torch numpy scikit-learn scipy joblib tqdm pandas
python 源码/build_dense_features.py --train-file train.csv
python 源码/train_nn_v2.py --seed 20260504
python 源码/train_nn_v2.py --seed 42
python 源码/train_nn_v2.py --seed 123
python 源码/predict_ensemble_v2.py --tune --models 模型/model_s0.joblib 模型/model_s1.joblib 模型/model_s2.joblib
```

详见 `REPRODUCE.md`。

## 文件结构

```
├── 源码/              # 活跃版本代码
├── 模型/              # 模型文件（.joblib 不进 git）
├── 提交结果/           # 提交文件
├── 版本记录/           # 历史快照（不进 git）
├── 缓存/              # 特征缓存
├── REPRODUCE.md       # 详细复现说明
├── requirements.txt   # Python 依赖
└── README.md          # 本文件
```
