$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$PYTHON = "C:\Users\30816\.conda\envs\iscc-gpu\python.exe"
$SRC = Join-Path $ROOT "源码"

Write-Host ""
Write-Host "========================================"
Write-Host "  复现 blend_top2_ft (平台 0.709)"
Write-Host "  v1.1 + v1.4 两组FT模型 50:50 融合"
Write-Host "========================================"
Write-Host ""

Write-Host "训练 + 预测 + 输出提交文件..."
Write-Host "预计 60-80 分钟 (GPU)"
Write-Host ""
& $PYTHON (Join-Path $SRC "reproduce_best.py") all
if ($LASTEXITCODE -ne 0) { Write-Host "失败!"; Read-Host "按回车退出"; exit 1 }

Write-Host ""
Write-Host "验证提交文件..."
& $PYTHON (Join-Path $SRC "validate_submission.py") --submission-path (Join-Path $ROOT "提交结果\blend_top2_ft.csv")

Write-Host ""
Write-Host "========================================"
Write-Host "  完成! 提交文件: 提交结果\blend_top2_ft.csv"
Write-Host "========================================"
Read-Host "按回车退出"