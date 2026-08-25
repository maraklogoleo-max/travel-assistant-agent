$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

if (-not (Test-Path "$Root\.venv\Scripts\python.exe")) {
    python -m venv "$Root\.venv"
}

& "$Root\.venv\Scripts\python.exe" -m pip install -e "$Root\backend[dev]"
Push-Location "$Root\frontend"
try {
    npm install
} finally {
    Pop-Location
}

if (-not (Test-Path "$Root\backend\.env")) {
    Copy-Item "$Root\backend\.env.example" "$Root\backend\.env"
}

Write-Host "Setup complete. Add DEEPSEEK_API_KEY and AMAP_API_KEY to backend\.env."
