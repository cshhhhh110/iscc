# 先读总规则

开始本项目之前，请先查看总目录的 `总要求.md` 和 `ACTION_LOG.md`。

# 网络安全智能分类挑战 v1.2

v1.2 是纯 GPU 主线版本，保留 `v1.1` 作为回滚快照，不覆盖旧版脚本和结果。

## 思路

- 仍然按真实 CSV 做 12 类数值表格分类。
- 纯 GPU 双模型：`FT-Transformer` + `Residual MLP`。
- 两个模型都走 AMP、早停、梯度裁剪和 fold 内标准化。
- 用 class-balanced focal loss，优先照顾 `class_10`、`class_11` 等弱类。
- 默认用 OOF 搜索到的纯 GPU 权重，不再把 CPU 基线混进默认提交。

## 运行

先确认在 `iscc-gpu` 环境中：

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
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

## 主要输出

- `模型/gpu_model_bundle_v1.2.pt`
- `模型/gpu_oof_probs_v1.2.npy`
- `模型/gpu_test_probs_v1.2.npy`
- `模型/gpu_validation_report_v1.2.json`
- `提交结果/submission_gpu_v1.2.csv`
- `提交结果/submission_blend_v1.2.csv`

## 备注

- `train` 只训练和保存 bundle / 概率，不直接写正式提交。
- `predict` 只读 bundle，生成正式提交。
- `blend` 用纯 GPU 两模型结果做可选融合。
70%