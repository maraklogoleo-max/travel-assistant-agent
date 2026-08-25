$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$RunDir = Join-Path $Root ".run"
$ProcessFile = Join-Path $RunDir "processes.json"

function Test-PortInUse([int]$Port) {
    $Client = [System.Net.Sockets.TcpClient]::new()
    try {
        $Connect = $Client.ConnectAsync("127.0.0.1", $Port)
        return $Connect.Wait(400) -and $Client.Connected
    } catch {
        return $false
    } finally {
        $Client.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Dependencies are missing. Run .\setup.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $Root "backend\.env"))) {
    Copy-Item -LiteralPath (Join-Path $Root "backend\.env.example") -Destination (Join-Path $Root "backend\.env")
}
$BusyPorts = @(3000, 8000) | Where-Object { Test-PortInUse $_ }
if ($BusyPorts.Count -gt 0) {
    throw "Port(s) $($BusyPorts -join ', ') are already in use. If this project is already running, use .\restart.ps1 so updated environment variables are loaded."
}

New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$Backend = Start-Process -FilePath $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000" -WorkingDirectory (Join-Path $Root "backend") -WindowStyle Hidden -PassThru
$Frontend = Start-Process -FilePath "npm.cmd" -ArgumentList "run", "dev" -WorkingDirectory (Join-Path $Root "frontend") -WindowStyle Hidden -PassThru
@{
    Root = [System.IO.Path]::GetFullPath($Root)
    Backend = @{ Id = $Backend.Id; StartedAt = $Backend.StartTime.ToUniversalTime().ToString("O") }
    Frontend = @{ Id = $Frontend.Id; StartedAt = $Frontend.StartTime.ToUniversalTime().ToString("O") }
} | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $ProcessFile -Encoding UTF8

try {
    Write-Host "Travel Assistant Agent is running at http://localhost:3000"
    Write-Host "Backend API is running at http://localhost:8000"
    Write-Host "Press Ctrl+C to stop both services."
    Wait-Process -Id $Backend.Id, $Frontend.Id
} finally {
    foreach ($Process in @($Backend, $Frontend)) {
        if (-not $Process.HasExited) { Stop-Process -Id $Process.Id -ErrorAction SilentlyContinue }
    }
    Remove-Item -LiteralPath $ProcessFile -Force -ErrorAction SilentlyContinue
}
