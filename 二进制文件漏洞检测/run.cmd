@echo off
setlocal
set "PY=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "SRC=%~dp0ิดย๋"

echo ===== v2.4 TRAIN %time% =====
"%PY%" "%SRC%\train.py" --skip-pseudo
if errorlevel 1 (echo FAILED & pause & exit /b 1)

echo ===== v2.4 TEST %time% =====
"%PY%" "%SRC%\test.py"
if errorlevel 1 (echo FAILED & pause & exit /b 1)

echo ===== DONE %time% =====
pause