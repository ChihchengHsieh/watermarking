param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string]$Checkpoint = "",
    [string]$OutputCsv = "",
    [switch]$DryRun,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "scripts\eval_stage2_scheduler_run.py",
    "--manifest-csv", $ManifestCsv,
    "--run-id", $RunId,
    "--batch-size", "$BatchSize",
    "--testing-times", "$TestingTimes"
)

if ($Checkpoint -ne "") {
    $argsList += @("--checkpoint", $Checkpoint)
}
if ($OutputCsv -ne "") {
    $argsList += @("--output-csv", $OutputCsv)
}

if ($DryRun) {
    $argsList += "--dry-run"
}

& $PythonExe @argsList
