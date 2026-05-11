@echo off
setlocal
set "PY=%USERPROFILE%\.conda\envs\iscc-gpu\python.exe"
set "ROOT=%~dp0"

echo =========================================
echo   NIGHT RUN - 3 projects serial
echo   %date% %time%
echo =========================================

:: ---- 1. Binary v2.3 ----
echo.
echo ##### [1/3] Binary v2.3 TRAIN  %time% #####
"%PY%" "%ROOT%二进制文件漏洞检测\源码\train.py" --skip-pseudo
if errorlevel 1 (echo BINARY TRAIN FAILED & exit /b 1)
echo ##### [1/3] Binary v2.3 TEST  %time% #####
"%PY%" "%ROOT%二进制文件漏洞检测\源码\test.py"
if errorlevel 1 (echo BINARY TEST FAILED & exit /b 1)
echo ##### Binary DONE %time% #####

:: ---- 2. PowerShell ----
echo.
echo ##### [2/3] PowerShell TRAIN  %time% #####
"%PY%" "%ROOT%powershell恶意脚本检测\源码\train.py"
if errorlevel 1 (echo PS TRAIN FAILED & exit /b 1)
echo ##### [2/3] PowerShell PREDICT  %time% #####
"%PY%" "%ROOT%powershell恶意脚本检测\源码\predict.py"
if errorlevel 1 (echo PS PREDICT FAILED & exit /b 1)
echo ##### PowerShell DONE %time% #####

:: ---- 3. Network Security ----
echo.
echo ##### [3/3] NetSec TRAIN  %time% #####
"%PY%" "%ROOT%网络安全智能分类挑战\源码\train_gpu_v1_6.py"
if errorlevel 1 (echo NET TRAIN FAILED & exit /b 1)
echo ##### [3/3] NetSec PREDICT  %time% #####
"%PY%" "%ROOT%网络安全智能分类挑战\源码\predict_gpu_v1_6.py"
if errorlevel 1 (echo NET PREDICT FAILED & exit /b 1)
echo ##### NetSec DONE %time% #####

echo.
echo =========================================
echo   ALL DONE %date% %time%
echo =========================================
pause