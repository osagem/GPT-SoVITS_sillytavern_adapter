@echo off
chcp 65001 >nul
title GPT-SoVITS Adapter

cd /d "H:\ai\GPT-Sovits\GPT-SoVITS_sillytavern_adapter"

echo [Adapter] Working dir: %CD%
echo [Adapter] Python: H:\ai\GPT-Sovits\GPT-SoVITS-v5-20261006nvidia50x0\runtime\python.exe
echo [Adapter] Starting adapter.py ...
echo.

"H:\ai\GPT-Sovits\GPT-SoVITS-v5-20261006nvidia50x0\runtime\python.exe" -s adapter.py

echo.
echo [Adapter] exited with code %ERRORLEVEL%
pause