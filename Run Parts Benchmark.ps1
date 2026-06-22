$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

Write-Host "Checking dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Running subsystem benchmarks against a 60fps frame budget..."
.\.venv\Scripts\python.exe benchmark_parts.py --profile pi_zero --seconds 0.75
.\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --seconds 0.75

Write-Host "Running stress subsystem benchmark with 2000 LEDs and the Pi 5 profile wave cap..."
.\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --seconds 1.0 --warmup 10 --min-iterations 20 --mode stress

Write-Host "For a 100-wave overload test, run:"
Write-Host ".\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --mode stress --wave-count 100 --seconds 2 --warmup 20 --min-iterations 20"
Write-Host "For the absolute ceiling test, run:"
Write-Host ".\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --seconds 0.05 --mode ceiling"
