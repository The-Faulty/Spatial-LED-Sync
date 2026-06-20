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

echo Starting Pi 5 profile live engine. WLED output follows config.json.
".venv\Scripts\python.exe" main.py --no-debug --profile pi_5 --overload-policy adaptive_quality
pause
