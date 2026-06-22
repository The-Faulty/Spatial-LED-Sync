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

echo Running subsystem benchmarks against a 60fps frame budget...
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_zero --seconds 0.75
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 0.75

echo Running stress subsystem benchmark with 2000 LEDs and 100 active waves...
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 0.05 --mode stress

echo For the absolute ceiling test, run:
echo ".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 0.05 --mode ceiling
pause
