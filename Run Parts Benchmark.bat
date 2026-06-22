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

echo Running stress subsystem benchmark with 2000 LEDs and the Pi 5 profile wave cap...
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 1.0 --warmup 10 --min-iterations 20 --mode stress
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 1.0 --warmup 10 --min-iterations 20 --mode stress --render-mode edge_effects
".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 1.0 --warmup 10 --min-iterations 20 --mode stress --render-mode hybrid_edge_full


echo For a 100-wave overload test, run:
echo ".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --mode stress --wave-count 100 --seconds 2 --warmup 20 --min-iterations 20
echo For the absolute ceiling test, run:
echo ".venv\Scripts\python.exe" benchmark_parts.py --profile pi_5 --seconds 0.05 --mode ceiling
pause
