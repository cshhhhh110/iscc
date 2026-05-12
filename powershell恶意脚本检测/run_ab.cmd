@echo off
setlocal
set "PY=D:\anaconda3\envs\iscc-gpu\python.exe"
set "SRC=%~dp0源码"
set "MODEL_DIR=%~dp0模型"
set "OUT_DIR=%~dp0提交结果"

echo ===== [1/4] FUSION TRAIN %time% =====
"%PY%" "%SRC%\train.py" --device auto --model-output "%MODEL_DIR%\model_fusion.joblib" --report-output "%MODEL_DIR%\report_fusion.json"
if errorlevel 1 (echo FUSION TRAIN FAILED & pause & exit /b 1)

echo ===== [2/4] FUSION PREDICT %time% =====
"%PY%" "%SRC%\predict.py" --device auto --model "%MODEL_DIR%\model_fusion.joblib" --output "%OUT_DIR%\submission_fusion.csv"
if errorlevel 1 (echo FUSION PREDICT FAILED & pause & exit /b 1)

echo ===== [3/4] TREE TRAIN %time% =====
"%PY%" "%SRC%\train.py" --device auto --arch tree --model-output "%MODEL_DIR%\model_tree.joblib" --report-output "%MODEL_DIR%\report_tree.json"
if errorlevel 1 (echo TREE TRAIN FAILED & pause & exit /b 1)

echo ===== [4/4] TREE PREDICT %time% =====
"%PY%" "%SRC%\predict.py" --device auto --model "%MODEL_DIR%\model_tree.joblib" --output "%OUT_DIR%\submission_tree.csv"
if errorlevel 1 (echo TREE PREDICT FAILED & pause & exit /b 1)

echo ===== ALL DONE %time% =====
echo.
echo Results:
echo   %OUT_DIR%\submission_fusion.csv
echo   %OUT_DIR%\submission_tree.csv
pause
