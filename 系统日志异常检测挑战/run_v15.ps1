$PY = "$env:USERPROFILE\.conda\envs\iscc-gpu\python.exe"
$SRC = "E:\赛题数据\系统日志异常检测挑战\源码"
$MODEL = "E:\赛题数据\系统日志异常检测挑战\模型"

Write-Host "===== Step 1/2: Train seed=999 $(Get-Date -Format 'HH:mm:ss') ====="
& $PY "$SRC\train_nn_v2.py" --seed 999 --model-path "$MODEL\model_bundle_nn_v2_seed999.joblib" --submission-path "E:\赛题数据\系统日志异常检测挑战\提交结果\submission_nn_v2_seed999.csv"
if ($LASTEXITCODE -ne 0) { Write-Host "SEED 999 FAILED"; Read-Host "Press Enter"; exit 1 }

Write-Host "===== Step 2/2: Ensemble $(Get-Date -Format 'HH:mm:ss') ====="
& $PY "$SRC\predict_ensemble_v2.py" --models "$MODEL\model_bundle_nn_v2.joblib" "$MODEL\model_bundle_nn_v2_seed42.joblib" "$MODEL\model_bundle_nn_v2_seed123.joblib" "$MODEL\model_bundle_nn_v2_seed999.joblib" --tune
if ($LASTEXITCODE -ne 0) { Write-Host "ENSEMBLE FAILED"; Read-Host "Press Enter"; exit 1 }

Write-Host "===== DONE $(Get-Date -Format 'HH:mm:ss') ====="
Read-Host "Press Enter"