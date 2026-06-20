@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Creating local Python environment...
  python -m venv .venv
)

echo Checking dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Failed to install dependencies.
  pause
  exit /b 1
)

echo Starting 3D room editor at http://127.0.0.1:8787
".venv\Scripts\python.exe" spatial_editor_server.py --host 127.0.0.1 --port 8787 --open
pause
