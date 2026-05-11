# 先读总规则

开始本项目之前，请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`。

根目录的 `源码/` 只放当前活跃版本；历史代码、模型、结果和 `smoke_*` 统一进 `版本记录/<version>/`，文件名也要带版本号。

# 网络安全智能分类挑战 v1.3

当前工作版是纯 GPU 稳健提分版 v1.3。默认主线改为 `compact_resmlp`，`train` 只负责训练并保存 bundle / 概率 / 验证报告，`predict` 负责从 bundle 生成正式提交和少量候选提交。

## 任务

按真实 CSV 做 12 类数值表格分类，训练集 `train_data.csv`，测试集 `test_data.csv`，提交格式为 `id,label`。

## 目录

- `源码/`：当前活跃版本训练、预测、校验脚本
- `模型/`：bundle、OOF 概率、test 概率、验证报告和预测元数据
- `提交结果/`：正式提交和候选提交
- `版本记录/`：历史版本快照
- `ACTION_LOG.md`：项目执行记录

## 环境

推荐在 `iscc-gpu` 环境中运行：

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_3.py --smoke
```

## 训练与预测

`train_gpu_v1_3.py` 不直接写提交文件，完整训练后再运行 `predict_gpu_v1_3.py`。

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_3.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_v1_3.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\submission_gpu_v1.3.csv
```

`predict_gpu_v1_3.py` 默认写 `submission_gpu_v1.3.csv`，并额外写 OOF 上可对比的 seed / bias 候选提交；具体映射见 `模型/gpu_prediction_metadata_v1.3.json`。

## Smoke

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_v1_3.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_v1_3.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\smoke_submission_gpu_v1.3.csv
```

## 输出

- `模型/gpu_model_bundle_v1.3.pt`
- `模型/gpu_resmlp_oof_probs_v1.3.npy`
- `模型/gpu_resmlp_test_probs_v1.3.npy`
- `模型/gpu_validation_report_v1.3.json`
- `模型/gpu_prediction_metadata_v1.3.json`
- `提交结果/submission_gpu_v1.3.csv`
- `提交结果/submission_gpu_v1.3_*.csv` 候选提交

## 打包

最终压缩包按竞赛要求包含：

- `源码/`
- `模型/`
- `提交结果/`
- `docker容器/`
- `requirements.txt`
- `README.md`
