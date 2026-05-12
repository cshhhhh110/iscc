$py = "D:\anaconda3\envs\iscc-gpu\python.exe"
$src = "E:\bisai\源码"
$models = "E:\bisai\模型"
$out = "E:\bisai\提交结果"

New-Item -ItemType Directory -Force $models | Out-Null
New-Item -ItemType Directory -Force $out | Out-Null

Write-Host "===== [1/4] FUSION TRAIN $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$src\train.py" --device auto --model-output "$models\model_fusion.joblib" --report-output "$models\report_fusion.json"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED"; pause; exit 1 }

Write-Host "===== [2/4] FUSION PREDICT $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$src\predict.py" --device auto --model "$models\model_fusion.joblib" --output "$out\submission_fusion.csv"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED"; pause; exit 1 }

Write-Host "===== [3/4] TREE TRAIN $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$src\train.py" --device auto --arch tree --model-output "$models\model_tree.joblib" --report-output "$models\report_tree.json"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED"; pause; exit 1 }

Write-Host "===== [4/4] TREE PREDICT $(Get-Date -Format 'HH:mm:ss') ====="
& $py "$src\predict.py" --device auto --model "$models\model_tree.joblib" --output "$out\submission_tree.csv"
if ($LASTEXITCODE -ne 0) { Write-Host "FAILED"; pause; exit 1 }

Write-Host "===== ALL DONE $(Get-Date -Format 'HH:mm:ss') ====="
Write-Host "submission_fusion.csv + submission_tree.csv"
