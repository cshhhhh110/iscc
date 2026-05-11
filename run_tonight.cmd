@echo off
setlocal
set "PYTHON=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "ROOT=%~dp0"

echo ===== [1/4] Binary v1.9 TRAIN %date% %time% =====
"%PYTHON%" "%ROOT%二进制文件漏洞检测\源码\train.py" --skip-pseudo > "%ROOT%二进制文件漏洞检测\train_v1.9.log" 2> "%ROOT%二进制文件漏洞检测\train_v1.9_err.log"
if errorlevel 1 (
    echo BINARY TRAIN FAILED
    pause
    exit /b 1
)
echo ===== [1/4] DONE %time% =====

echo ===== [2/4] Binary v1.9 TEST %time% =====
"%PYTHON%" "%ROOT%二进制文件漏洞检测\源码\test.py" > "%ROOT%二进制文件漏洞检测\test_v1.9.log" 2> "%ROOT%二进制文件漏洞检测\test_v1.9_err.log"
if errorlevel 1 (
    echo BINARY TEST FAILED
    pause
    exit /b 1
)
echo ===== [2/4] DONE %time% =====

echo ===== [3/4] Log Anomaly TRAIN %time% =====
"%PYTHON%" "%ROOT%系统日志异常检测挑战\源码\train.py" > "%ROOT%系统日志异常检测挑战\train.log" 2> "%ROOT%系统日志异常检测挑战\train_err.log"
if errorlevel 1 (
    echo LOG TRAIN FAILED
    pause
    exit /b 1
)
echo ===== [3/4] DONE %time% =====

echo ===== [4/4] Log Anomaly PREDICT %time% =====
"%PYTHON%" "%ROOT%系统日志异常检测挑战\源码\predict.py" > "%ROOT%系统日志异常检测挑战\predict.log" 2> "%ROOT%系统日志异常检测挑战\predict_err.log"
if errorlevel 1 (
    echo LOG PREDICT FAILED
    pause
    exit /b 1
)
echo ===== [4/4] DONE %time% =====

echo.
echo ===== ALL DONE %date% %time% =====
pause