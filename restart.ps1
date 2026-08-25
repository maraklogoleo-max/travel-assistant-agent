$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$RunDir = Join-Path $Root ".run"
$ProcessFile = Join-Path $RunDir "processes.json"

function Stop-ChildProcessTree([int]$Id) {
    $Children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $Id"
    foreach ($Child in $Children) { Stop-ChildProcessTree -Id ([int]$Child.ProcessId) }
    Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
}

function Stop-ProjectProcessTree([int]$Id, [datetime]$ExpectedStart) {
    $Process = Get-Process -Id $Id -ErrorAction SilentlyContinue
    if (-not $Process) { return }
    $ActualStart = $Process.StartTime.ToUniversalTime()
    if ([Math]::Abs(($ActualStart - $ExpectedStart.ToUniversalTime()).TotalSeconds) -gt 2) {
        throw "PID $Id has been reused by another process; refusing to stop it."
    }
    $Children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $Id"
    foreach ($Child in $Children) { Stop-ChildProcessTree -Id ([int]$Child.ProcessId) }
    Stop-Process -Id $Id -Force -ErrorAction SilentlyContinue
}

if (Test-Path -LiteralPath $ProcessFile) {
    $Record = Get-Content -LiteralPath $ProcessFile -Raw | ConvertFrom-Json
    if ([System.IO.Path]::GetFullPath([string]$Record.Root) -ne [System.IO.Path]::GetFullPath($Root)) {
        throw "The recorded processes belong to another workspace; refusing to stop them."
    }
    foreach ($Entry in @($Record.Backend, $Record.Frontend)) {
        Stop-ProjectProcessTree -Id ([int]$Entry.Id) -ExpectedStart ([datetime]$Entry.StartedAt)
    }
    Remove-Item -LiteralPath $ProcessFile -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "No project PID record was found. Existing processes will not be stopped automatically."
}

& (Join-Path $Root "start.ps1")
