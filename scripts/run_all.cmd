@echo off
setlocal

set "HEAPO_CONFIG=%~1"
if "%HEAPO_CONFIG%"=="" set "HEAPO_CONFIG=.\configs\local.yaml"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_all.ps1" -Config "%HEAPO_CONFIG%"
exit /b %ERRORLEVEL%
