$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = "C:\Users\30816\.conda\envs\iscc-gpu\python.exe"

Write-Host ""
Write-Host "========================================"
Write-Host "  v2.0 可复现 FT-Transformer"
Write-Host "  python 源码/train_v2_0.py all"
Write-Host "========================================"
Write-Host ""

& $PYTHON (Join-Path $ROOT "源码\train_v2_0.py") all
if ($LASTEXITCODE -ne 0) { Read-Host "按回车退出"; exit 1 }

Write-Host ""
Write-Host "验证提交..."
& $PYTHON (Join-Path $ROOT "源码\validate_submission.py") --submission-path (Join-Path $ROOT "提交结果\submission_v2_0.csv")

Write-Host ""
Write-Host "完成! 提交: submission_v2_0.csv"
Read-Host "按回车退出"