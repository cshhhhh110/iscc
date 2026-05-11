# ISCC PowerShell 恶意脚本检测 v1.7

本项目用于 PowerShell 恶意脚本三分类任务，当前活跃版本为 `v1.7`。

## 数据

- 训练集：`data_train.csv`，共 48065 条样本
- 测试集：`data_test.csv`，共 20000 条样本
- 特征：15 个离散维度特征
- 标签：`0` 正常脚本，`1` 一般恶意脚本，`2` 混淆恶意脚本
- 评估指标：Macro-F1

## 运行

```powershell
pip install -r requirements.txt
python .\源码\train.py --device auto
python .\源码\predict.py --device auto
python .\源码\validate_submission.py
```

或双击 `run.cmd`。

## 平台成绩

| 版本 | OOF | 平台 | 关键改动 |
|------|-----|------|----------|
| v1.5 | 0.755537 | 0.69104 | 7候选fusion基线 |
| v1.6 | 0.755472 | 0.68844 | 交互特征+KMeans+SMOTE+Focal Loss（无效） |
| v1.7 | 0.755350 | 0.68965 | 对抗验证（无效） |

## 当前版本产物

- `模型/model_bundle_v1.7.joblib`
- `模型/validation_report_v1.7.json`
- `提交结果/submission_v1.7.csv`

## 版本记录

- `版本记录/original/` ~ `版本记录/v1.4/`

项目级日志见 `ACTION_LOG.md`。
