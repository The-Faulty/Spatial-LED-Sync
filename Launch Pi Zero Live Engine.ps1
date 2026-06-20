$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

Write-Host "Checking dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Starting Pi Zero profile live engine. WLED output follows config.json."
.\.venv\Scripts\python.exe main.py --no-debug --profile pi_zero --overload-policy adaptive_quality
