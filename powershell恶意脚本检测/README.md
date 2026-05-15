# ISCC PowerShell 恶意脚本检测 v4.1

本项目用于 PowerShell 恶意脚本三分类任务，当前活跃版本为 `v4.1`。

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
| v2.0 | - | Mega 100模型集成 |
| **v4.1** | **TBD** | **Condition-Aware KD + weighted Hamming router** |

## 运行

```powershell
python 源码_v4\experiment_router.py    # 生成 submission
```

## 版本记录

- `版本记录/original/` ~ `版本记录/v3.0/`

项目级日志见 `ACTION_LOG.md`。