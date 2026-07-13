param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [switch]$DryRun,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 3
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv"
}

$row = Import-Csv $ManifestCsv | Where-Object { $_.run_id -eq $RunId } | Select-Object -First 1
if (!$row) {
    throw "Run ID not found in manifest: $RunId"
}

$runDir = $row.run_dir
$latest = Join-Path $runDir "checkpoints/latest.pth"
$outputCsv = Join-Path $runDir "attack_eval_summary_latest.csv"

if (!(Test-Path $latest)) {
    throw "Latest checkpoint not found: $latest"
}

Write-Host "Evaluating latest checkpoint for $RunId"
Write-Host "  checkpoint=$latest"
Write-Host "  output_csv=$outputCsv"

$argsList = @(
    "scripts\eval_stage2_scheduler_run.py",
    "--manifest-csv", $ManifestCsv,
    "--run-id", $RunId,
    "--checkpoint", $latest,
    "--output-csv", $outputCsv,
    "--batch-size", "$BatchSize",
    "--testing-times", "$TestingTimes"
)

if ($DryRun) {
    $argsList += "--dry-run"
}

& $PythonExe @argsList
if ($LASTEXITCODE -ne 0) {
    throw "Latest checkpoint evaluation failed: $RunId"
}
