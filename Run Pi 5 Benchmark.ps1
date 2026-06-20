$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

Write-Host "Checking dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Running Pi 5 profile benchmark..."
.\.venv\Scripts\python.exe benchmark.py --profile pi_5 --seconds 15
