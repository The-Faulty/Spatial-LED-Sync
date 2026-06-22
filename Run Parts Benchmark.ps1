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

Write-Host "Running stress subsystem benchmark with 2000 LEDs and 100 active waves..."
.\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --seconds 0.05 --mode stress

Write-Host "For the absolute ceiling test, run:"
Write-Host ".\.venv\Scripts\python.exe benchmark_parts.py --profile pi_5 --seconds 0.05 --mode ceiling"
