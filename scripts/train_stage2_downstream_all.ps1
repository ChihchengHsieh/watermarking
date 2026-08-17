param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [int]$Epochs = 120,
    [int]$BatchSize = 8,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$cmdArgs = @(
    "scripts\train_stage2_downstream_shared.py",
    "--manifest-csv", $ManifestCsv,
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize"
)
if ($DryRun) {
    $cmdArgs += "--dry-run"
}

& $PythonExe @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Stage 2 shared downstream training failed. Re-run the same command to resume."
}
