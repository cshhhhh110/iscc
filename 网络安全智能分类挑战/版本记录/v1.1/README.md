# 先读总规则

开始本项目之前，请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`。

# 网络安全智能分类挑战 v1.1

v1.1 是独立于旧版脚本的新架构版本，不覆盖 `train.py`、`train_gpu_tabular.py` 或已有提交文件。

## 思路

- 数据仍按真实 CSV 处理：50 个数值特征，12 类标签，提交格式为 `id,label`。
- 新架构使用 GPU 版 FT-Transformer：每个数值特征先变成 token，再经过 Transformer encoder。
- 输出文件全部带 `v1.1` 后缀；小规模测试输出带 `smoke_` 前缀，避免误提交。
- 默认只用 1 seed x 5 folds，避免训练时间过长；如要冲分，可手动增加 `--seeds`。

## 小规模测试

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_fttransformer_v1_1.py --smoke
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\smoke_submission_gpu_fttransformer_v1.1.csv
```

## 正式训练

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\train_gpu_fttransformer_v1_1.py
```

正式训练会生成：

- `模型/gpu_fttransformer_model_bundle_v1.1.pt`
- `模型/gpu_fttransformer_oof_probs_v1.1.npy`
- `模型/gpu_fttransformer_test_probs_v1.1.npy`
- `模型/gpu_fttransformer_validation_report_v1.1.json`
- `提交结果/submission_gpu_fttransformer_v1.1.csv`

## 复现预测

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\predict_gpu_fttransformer_v1_1.py
```

## 融合提交

如果旧版 `模型/sklearn_test_probs.npy` 存在，可以融合旧树模型和 v1.1 GPU 模型：

```powershell
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\blend_predictions_v1_1.py
& C:\Users\30816\.conda\envs\iscc-gpu\python.exe 源码\validate_submission.py --submission-path 提交结果\submission_blend_v1.1.csv
```
