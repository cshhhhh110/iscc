@echo off
chcp 65001 >nul
setlocal
set "PY=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "LOG=%~dp0"
set "SRC=%~dp0源码\"

echo ===== Log Anomaly Build Dense Features %time% =====
"%PY%" "%SRC%build_dense_features.py" 2> "%LOG%build_err.log"
if errorlevel 1 (echo BUILD FAILED & type "%LOG%build_err.log" & pause & exit /b 1)

echo ===== Log Anomaly Train NN v2 %time% =====
"%PY%" "%SRC%train_nn_v2.py" 2> "%LOG%train_err.log"
if errorlevel 1 (echo TRAIN FAILED & type "%LOG%train_err.log" & pause & exit /b 1)

echo ===== Log Anomaly Predict NN v2 %time% =====
"%PY%" "%SRC%predict_nn_v2.py" 2> "%LOG%predict_err.log"
if errorlevel 1 (echo PREDICT FAILED & type "%LOG%predict_err.log" & pause & exit /b 1)

echo ===== DONE %time% =====
pause