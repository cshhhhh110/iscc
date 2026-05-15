# ISCC PowerShell 恶意脚本检测 v3.7

本项目用于 PowerShell 恶意脚本三分类任务，当前活跃版本为 `v3.7_align`。

## 数据

- 训练集：`data_train.csv`，共 48065 条样本
- 测试集：`data_test.csv`，共 20000 条样本
- 特征：15 个离散维度特征
- 评估指标：Macro-F1

## 平台成绩

| 版本 | 平台分 | 关键改动 |
|------|--------|----------|
| v1.5 | 0.69104 | 7候选fusion基线 |
| v1.8 | 0.69779 | tree + pseudo-label |
| v3.3 | 0.70441 | Condition-Aware KD (V1_zero_base teacher) |
| **v3.7** | **0.70831** | **exact_key5 特征对齐 (class1 +46)** |
| _参考解_ | _0.713_ | |

## 运行

```powershell
cd 源码（必交）
python train_v3.4.py --train data_train.csv --test data_test.csv
```

## 版本记录

- `版本记录/original/` ~ `版本记录/v3.7_align/`

项目级日志见 `ACTION_LOG.md`。