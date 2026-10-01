param(
    [string]$Config = ".\configs\local.yaml",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
)

$ErrorActionPreference = "Stop"

Write-Host "Running cached-result statistical finalization..."
& $PythonLauncher $PythonVersion -m heapo_forecasting.cli finalize --config $Config
if ($LASTEXITCODE -ne 0) {
    throw "Statistical finalization failed with exit code $LASTEXITCODE"
}

Write-Host "Statistical finalization completed."
