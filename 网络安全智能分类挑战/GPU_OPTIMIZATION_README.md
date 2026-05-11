# GPU optimization iteration

This is a separate GPU iteration. Existing source files such as `源码/train.py`, `源码/predict.py`, and `源码/common.py` are kept as backups and are not overwritten by this iteration.

## 1. Install CUDA PyTorch

Your current environment previously reported CPU-only PyTorch. Install a CUDA build before running GPU training:

```powershell
& C:\Users\30816\.conda\envs\iscc-ml\python.exe -m pip uninstall -y torch torchvision torchaudio
& C:\Users\30816\.conda\envs\iscc-ml\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA:

```powershell
& C:\Users\30816\.conda\envs\iscc-ml\python.exe -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`torch.cuda.is_available()` must be `True`.

## 2. Train GPU model

```powershell
cd "D:\桌面\赛题数据\网络安全智能分类挑战"
& C:\Users\30816\.conda\envs\iscc-ml\python.exe 源码\train_gpu_tabular.py
```

Default training:

- 2 seeds
- 5 folds
- 120 epochs
- batch size 4096
- early stopping patience 18
- mixed precision on CUDA

The training script has progress bars:

- outer bar: total seed/fold runs
- inner bar: epochs for the current fold
- postfix metrics: `train_loss`, `val_macro_f1`, `best_f1`, `patience`

Outputs:

- `模型/gpu_model_bundle.pt`
- `模型/gpu_oof_probs.npy`
- `模型/gpu_test_probs.npy`
- `模型/gpu_validation_report.json`
- `提交结果/submission_gpu.csv`

## 3. Blend submissions

After GPU training:

```powershell
& C:\Users\30816\.conda\envs\iscc-ml\python.exe 源码\blend_predictions.py
```

The blend script reads `模型/gpu_test_probs.npy`. If `模型/sklearn_test_probs.npy` is absent, it loads `模型/model_bundle.joblib` and computes sklearn probabilities automatically.

Default blend:

- GPU weight: 0.60
- sklearn weight: 0.40

Output:

- `提交结果/submission_blend.csv`
- `模型/blend_metadata.json`

## 4. Validate output

```powershell
& C:\Users\30816\.conda\envs\iscc-ml\python.exe 源码\validate_submission.py --submission-path 提交结果\submission_gpu.csv
& C:\Users\30816\.conda\envs\iscc-ml\python.exe 源码\validate_submission.py --submission-path 提交结果\submission_blend.csv
```

Submit `submission_blend.csv` first, and compare it with `submission_gpu.csv` if the platform allows multiple attempts.
