param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$cmdArgs = @(
    "scripts\eval_stage2_downstream_shared.py",
    "--batch-size", "$BatchSize",
    "--testing-times", "$TestingTimes"
)
if ($DryRun) { $cmdArgs += "--dry-run" }
if ($Force) { $cmdArgs += "--force" }

& $PythonExe @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Shared downstream evaluation failed. Re-run the same command to resume."
}
