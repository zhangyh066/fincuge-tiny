@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

rem 优先用仓库里的 venv，没有就用系统 python
set "PY=%~dp0venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"

set "HF_HOME=%~dp0hf"

echo ============================================================
echo   FinCUGE-Tiny web UI
echo   Model load takes about 15 seconds.
echo   Browser will open at http://127.0.0.1:7860
echo   Close this window to stop the service.
echo ============================================================
echo.

start "" /min cmd /c "timeout /t 12 >nul & start "" http://127.0.0.1:7860"

"%PY%" -u "%~dp0webui.py"

echo.
echo ==== Service stopped. Press any key to exit. ====
pause >nul
