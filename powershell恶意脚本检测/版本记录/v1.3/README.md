# 先读总规则：请先查看总目录的 [总要求.md](../../总要求.md) 和 [ACTION_LOG.md](../../ACTION_LOG.md)，再开始本项目。
# ISCC PowerShell 恶意脚本检测 v1.3

本项目用于 PowerShell 恶意脚本三分类任务，当前活跃版本为 `v1.3`。

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
python .\源码\predict.py
python .\源码\validate_submission.py
```

## 当前版本产物

- `模型/model_bundle_v1.3.joblib`
- `模型/validation_report_v1.3.json`
- `提交结果/submission_v1.3.csv`

## 版本记录

- `版本记录/original/`
- `版本记录/v1.1/`
- `版本记录/v1.2/`
- `版本记录/v1.3/`

项目级日志见 `ACTION_LOG.md`，总目录日志见 `../../ACTION_LOG.md`。
