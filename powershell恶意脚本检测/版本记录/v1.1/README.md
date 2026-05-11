# 先读总规则：请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`，再开始本项目。

# ISCC PowerShell 恶意脚本检测

本目录是 PowerShell 恶意脚本检测三分类任务的可复现 baseline。

## 数据

- `data_train.csv`: 48065 条训练样本，包含 `name`、`label` 和 15 个离散数值特征。
- `data_test.csv`: 20000 条测试样本，包含 `name` 和同样的 15 个特征。
- 标签含义：`0` 正常脚本，`1` 一般恶意脚本，`2` 混淆恶意脚本。
- 评估指标：Macro-F1。

## 目录

- `源码/`: 训练、预测和提交校验脚本。
- `模型/`: 训练后的模型包和验证报告。
- `提交结果/`: 默认输出 `submission_v1.1.csv`，官方另存名仍可用 `submission.csv`。
- `docker容器/`: 容器构建文件。
- `ACTION_LOG.md`: 操作记录。

## 运行

推荐训练使用 `C:\Users\30816\.conda\envs\iscc-gpu`，推理可在 CPU 或 GPU 环境执行。

```powershell
pip install -r requirements.txt
python .\源码\train.py --device auto
python .\源码\predict.py --device auto
python .\源码\validate_submission.py
```

训练脚本会自动比较树模型、PyTorch 深度模型和融合方案，并按 OOF Macro-F1 选最优结果。脚本会保存：

- `模型/model_bundle_v1.1.joblib`
- `模型/validation_report_v1.1.json`
- `提交结果/submission_v1.1.csv`
- 如需比赛官方文件名，再另存为 `提交结果/submission.csv`

提交文件严格为 UTF-8 CSV，两列：`name,label`。

## 版本

- 当前工作目录保存最新可运行版本。
- 历史快照放在 `版本记录/` 下，按 `original`、`v1.1` 这样的版本号保存。
- 本次迭代会同步保留 `源码/`、`模型/`、`提交结果/` 的对应快照。
