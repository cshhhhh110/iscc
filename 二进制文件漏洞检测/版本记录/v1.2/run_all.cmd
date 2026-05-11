@echo off
setlocal
set "ROOT=%~dp0"
"%USERPROFILE%\.conda\envs\iscc-ml\python.exe" "%ROOT%源码\train.py"
if errorlevel 1 exit /b 1
"%USERPROFILE%\.conda\envs\iscc-ml\python.exe" "%ROOT%源码\test.py"
endlocal
