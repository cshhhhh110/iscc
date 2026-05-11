@echo off
setlocal
set "PY=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "SRC=%~dp0ิดย๋"

echo ===== TEACHER TRAIN %time% =====
"%PY%" "%SRC%\train.py" --device auto --pseudo-label
if errorlevel 1 (echo FAILED & pause & exit /b 1)

echo ===== PREDICT %time% =====
"%PY%" "%SRC%\predict.py" --device auto
if errorlevel 1 (echo FAILED & pause & exit /b 1)

echo ===== DONE %time% =====
pause