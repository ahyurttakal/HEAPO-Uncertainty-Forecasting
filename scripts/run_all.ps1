param(
    [string]$Config = ".\configs\local.yaml",
    [string]$RunDir = "",
    [string]$PythonLauncher = "py",
    [string]$PythonVersion = "-3.11"
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

if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $ResolvedRunDir = & $PythonLauncher $PythonVersion "-c" `
        "from heapo_forecasting.config import load_config, run_dir; import sys; print(run_dir(load_config(sys.argv[1])))" `
        $Config
    if ($LASTEXITCODE -ne 0) {
        throw "Could not resolve the run directory from the configuration."
    }
    $RunDir = ($ResolvedRunDir | Select-Object -Last 1).Trim()
}

Write-Host "Creating manuscript tables and figures..."
$FigureArguments = @(".\scripts\make_manuscript_outputs.py", "--run-dir", $RunDir, "--dpi", "600", "--tiff")
Invoke-Python $FigureArguments

Write-Host "HEAPO workflow completed successfully."
