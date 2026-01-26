Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AppRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$Venv = Join-Path $AppRoot ".venv"
$Python = Join-Path $Venv "Scripts\\python.exe"

if (-not (Test-Path $Python)) {
  Write-Host "Python venv missing. Run install.ps1 first."
  exit 1
}

& $Python (Join-Path $AppRoot "run.py")
