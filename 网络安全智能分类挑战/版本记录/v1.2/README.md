# 先读总规则

开始本项目之前，请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`。

# 网络安全智能分类挑战 v1.2

当前工作版是纯 GPU 主线 v1.2，保留 `v1.1` 作为可回滚版本，历史快照在 `版本记录/`。

## 任务

按真实 CSV 做 12 类数值表格分类，训练集 `train_data.csv`，测试集 `test_data.csv`，提交格式为 `id,label`。

## 目录

- `源码/`：训练、预测、融合、校验脚本
- `模型/`：bundle、OOF 概率、test 概率、验证报告
- `提交结果/`：正式提交和 smoke 提交
- `版本记录/`：历史版本快照
- `ACTION_LOG.md`：项目执行记录

## 环境

推荐在 `iscc-gpu` 环境中运行：

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_2.py
```

## 训练与预测

`train_gpu_v1_2.py` 只负责训练和保存 bundle，不直接写正式提交。

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_2.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_v1_2.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\blend_predictions_v1_2.py
```

## Smoke

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_2.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_v1_2.py --output-path 提交结果\smoke_submission_gpu_v1.2.csv
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\blend_predictions_v1_2.py --output-path 提交结果\smoke_submission_blend_v1.2.csv
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\smoke_submission_gpu_v1.2.csv
```

## 输出

- `模型/gpu_model_bundle_v1.2.pt`
- `模型/gpu_oof_probs_v1.2.npy`
- `模型/gpu_test_probs_v1.2.npy`
- `模型/gpu_validation_report_v1.2.json`
- `提交结果/submission_gpu_v1.2.csv`
- `提交结果/submission_blend_v1.2.csv`

## 打包

最终压缩包按竞赛要求包含：

- `源码/`
- `模型/`
- `提交结果/`
- `docker容器/`
- `requirements.txt`
- `README.md`
