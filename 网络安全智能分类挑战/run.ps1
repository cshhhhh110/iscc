$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = "C:\Users\30816\.conda\envs\iscc-gpu\python.exe"
$SRC = Join-Path $ROOT "源码"
$OUT = Join-Path $ROOT "提交结果"

Write-Host ""
Write-Host "========================================"
Write-Host "  网络安全智能分类挑战 v1.9 一键运行"
Write-Host "  FT-Transformer 4-seed 集成"
Write-Host "========================================"
Write-Host ""

Write-Host "[1/3] 训练 (4 seeds x 5 folds = 20 模型集成) ..."
Write-Host ""
& $PYTHON (Join-Path $SRC "train_gpu_v1_9.py")
if ($LASTEXITCODE -ne 0) { Write-Host "训练失败!"; Read-Host "按回车退出"; exit 1 }

Write-Host ""
Write-Host "[2/3] 预测 ..."
Write-Host ""
& $PYTHON (Join-Path $SRC "predict_gpu_v1_9.py")
if ($LASTEXITCODE -ne 0) { Write-Host "预测失败!"; Read-Host "按回车退出"; exit 1 }

Write-Host ""
Write-Host "[3/3] 验证 ..."
Write-Host ""
& $PYTHON (Join-Path $SRC "validate_submission.py") --submission-path (Join-Path $OUT "submission_gpu_v1.9.csv")
if ($LASTEXITCODE -ne 0) { Write-Host "验证失败!"; Read-Host "按回车退出"; exit 1 }

Write-Host ""
Write-Host "========================================"
Write-Host "  完成! 提交文件: 提交结果\submission_gpu_v1.9.csv"
Write-Host "========================================"
Read-Host "按回车退出"