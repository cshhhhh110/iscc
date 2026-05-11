$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir
$Python = "$env:USERPROFILE\.conda\envs\iscc-gpu\python.exe"
$Log = Join-Path $ScriptDir "auto_log.txt"

"=== AUTO RUN v1.8 $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File $Log

# train
"TRAIN START $(Get-Date -Format 'HH:mm:ss')" | Out-File $Log -Append
$srcDir = Get-ChildItem -Directory | Where-Object { $_.Name -match '源' } | Select-Object -First 1
if (-not $srcDir) { $srcDir = Get-ChildItem -Directory | Where-Object { Test-Path (Join-Path $_.FullName 'train.py') } | Select-Object -First 1 }
if (-not $srcDir) { "CANNOT FIND SRC DIR" | Out-File $Log -Append; exit 1 }
& $Python (Join-Path $srcDir.FullName 'train.py') --device auto --pseudo-label *>&1 | Out-File $Log -Append
if ($LASTEXITCODE -ne 0) { "TRAIN FAILED" | Out-File $Log -Append; exit 1 }
"TRAIN END $(Get-Date -Format 'HH:mm:ss')" | Out-File $Log -Append

# predict
"PREDICT START $(Get-Date -Format 'HH:mm:ss')" | Out-File $Log -Append
& $Python (Join-Path $srcDir.FullName 'predict.py') --device auto *>&1 | Out-File $Log -Append
if ($LASTEXITCODE -ne 0) { "PREDICT FAILED" | Out-File $Log -Append; exit 1 }

# validate
$subFile = Join-Path $ScriptDir "提交结果\submission_v1.8.csv"
& $Python (Join-Path $srcDir.FullName 'validate_submission.py') --submission $subFile *>&1 | Out-File $Log -Append

"DONE $(Get-Date -Format 'HH:mm:ss')" | Out-File $Log -Append