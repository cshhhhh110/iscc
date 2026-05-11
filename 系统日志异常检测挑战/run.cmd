@echo off
setlocal
set "PY=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "LOG=%~dp0"
set "SRC=%~dp0ิดย๋"

echo ===== Log Anomaly Train %time% =====
"%PY%" "%SRC%\train.py" 2> "%LOG%train_err.log"
if errorlevel 1 (echo TRAIN FAILED & pause & exit /b 1)

echo ===== Log Anomaly Predict %time% =====
"%PY%" "%SRC%\predict.py" 2> "%LOG%predict_err.log"
if errorlevel 1 (echo PREDICT FAILED & pause & exit /b 1)

echo ===== DONE %time% =====
pause