$PY = "$env:USERPROFILE\.conda\envs\iscc-gpu\python.exe"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
$SRC = Join-Path $ROOT "源码"

Write-Host "===== Build Dense Features $(Get-Date -Format 'HH:mm:ss') ====="
& $PY (Join-Path $SRC "build_dense_features.py")
if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED (exit code: $LASTEXITCODE)"; Read-Host "Press Enter"; exit 1 }

Write-Host "===== Train NN v3 (boundary) $(Get-Date -Format 'HH:mm:ss') ====="
& $PY (Join-Path $SRC "train_nn_v3.py")
if ($LASTEXITCODE -ne 0) { Write-Host "TRAIN FAILED (exit code: $LASTEXITCODE)"; Read-Host "Press Enter"; exit 1 }

Write-Host "===== Predict NN v3 $(Get-Date -Format 'HH:mm:ss') ====="
& $PY (Join-Path $SRC "predict_nn_v3.py")
if ($LASTEXITCODE -ne 0) { Write-Host "PREDICT FAILED (exit code: $LASTEXITCODE)"; Read-Host "Press Enter"; exit 1 }

Write-Host "===== DONE $(Get-Date -Format 'HH:mm:ss') ====="
Read-Host "Press Enter"