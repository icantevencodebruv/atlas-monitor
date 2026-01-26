Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$AppRoot = Resolve-Path (Join-Path $PSScriptRoot "..\\..")
$Venv = Join-Path $AppRoot ".venv"
$Wheelhouse = Join-Path $AppRoot "wheelhouse"

if (-not (Test-Path $Venv)) {
  python -m venv $Venv
}

$Pip = Join-Path $Venv "Scripts\\pip.exe"

if (Test-Path $Wheelhouse) {
  & $Pip install --no-index --find-links $Wheelhouse -r (Join-Path $AppRoot "requirements.txt")
} else {
  & $Pip install -r (Join-Path $AppRoot "requirements.txt")
}

$TaskName = "HugoLeonRecorder"
$RunScript = Join-Path $PSScriptRoot "run_app.ps1"

schtasks /Create /F /SC ONLOGON /TN $TaskName /TR ("powershell -ExecutionPolicy Bypass -File `"$RunScript`"") | Out-Null

Write-Host "Installed. Launching now..."
powershell -ExecutionPolicy Bypass -File "$RunScript"
