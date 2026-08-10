param(
    [string]$Config = ".\configs\local.yaml",
    [string]$RunDir = ".\outputs\heapo_main",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11",
    [switch]$Tiff
)

$ErrorActionPreference = "Stop"

function Invoke-Python {
    param([string[]]$Arguments)
    & $PythonLauncher $PythonVersion @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Running full HEAPO workflow..."
Invoke-Python @("-m", "heapo_forecasting.cli", "full", "--config", $Config)

Write-Host "Running cached statistical finalization..."
Invoke-Python @("-m", "heapo_forecasting.cli", "finalize", "--config", $Config)

Write-Host "Creating manuscript tables and figures..."
$FigureArguments = @(".\scripts\make_manuscript_outputs.py", "--run-dir", $RunDir, "--dpi", "600")
if ($Tiff) {
    $FigureArguments += "--tiff"
}
Invoke-Python $FigureArguments

Write-Host "HEAPO workflow completed successfully."
