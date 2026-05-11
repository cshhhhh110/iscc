# 先读总规则

开始本项目之前，请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`。

根目录的 `源码/` 只放当前活跃版本；历史代码、模型、结果和 `smoke_*` 统一进 `版本记录/<version>/`，文件名也要带版本号。

# 网络安全智能分类挑战 v1.4

当前工作版是 FT-Transformer 冲高版 v1.4。默认主线回到 `submission_gpu_fttransformer_v1.1.csv` 那条路线：单模 FT-Transformer + robust median/IQR 预处理 + 多 seed，默认不再生成融合或对照提交。

## 任务

按真实 CSV 做 12 类数值表格分类，训练集 `train_data.csv`，测试集 `test_data.csv`，提交格式为 `id,label`。

## 目录

- `源码/`：当前活跃版本训练、预测、校验脚本
- `模型/`：bundle、OOF 概率、test 概率、验证报告
- `提交结果/`：正式提交
- `版本记录/`：历史版本快照
- `ACTION_LOG.md`：项目执行记录

## 环境

推荐在 `iscc-gpu` 环境中运行：

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_fttransformer_v1_4.py
```

## 训练与预测

`train_gpu_fttransformer_v1_4.py` 只负责训练和保存 bundle，不直接写正式提交。

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_fttransformer_v1_4.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_fttransformer_v1_4.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\submission_gpu_fttransformer_v1.4.csv
```

## Smoke

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_fttransformer_v1_4.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_fttransformer_v1_4.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\smoke_submission_gpu_fttransformer_v1.4.csv
```

## 输出

- `模型/gpu_fttransformer_model_bundle_v1.4.pt`
- `模型/gpu_fttransformer_oof_probs_v1.4.npy`
- `模型/gpu_fttransformer_test_probs_v1.4.npy`
- `模型/gpu_fttransformer_validation_report_v1.4.json`
- `提交结果/submission_gpu_fttransformer_v1.4.csv`

## 打包

最终压缩包按竞赛要求包含：

- `源码/`
- `模型/`
- `提交结果/`
- `docker容器/`
- `requirements.txt`
- `README.md`
70.7%