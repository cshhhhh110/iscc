@echo off
setlocal
set "ROOT=%~dp0"
"%USERPROFILE%\.conda\envs\iscc-ml\python.exe" "%ROOT%源码\train.py"
endlocal
