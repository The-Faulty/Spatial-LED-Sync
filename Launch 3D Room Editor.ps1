$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "Creating local Python environment..."
    python -m venv .venv
}

Write-Host "Checking dependencies..."
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "Starting 3D room editor at http://127.0.0.1:8787"
.\.venv\Scripts\python.exe spatial_editor_server.py --host 127.0.0.1 --port 8787 --open
