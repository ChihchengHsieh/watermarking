param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$RunId = "",
    [switch]$All,
    [switch]$DryRun,
    [int]$SaveInterval = 0,
    [int]$LogInterval = 0
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}
if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$rows = Import-Csv $ManifestCsv
if ($RunId -ne "") {
    $rows = $rows | Where-Object { $_.run_id -eq $RunId }
    if ($rows.Count -eq 0) {
        throw "Run ID not found in manifest: $RunId"
    }
} elseif (!$All) {
    throw "Specify either -RunId <id> or -All."
}

foreach ($row in $rows) {
    Write-Host ("=" * 80)
    Write-Host "Running Stage 2 scheduler job: $($row.run_id)"
    $algorithm = "fomaml"
    if ($row.PSObject.Properties.Name -contains "meta_algorithm" -and $row.meta_algorithm) {
        $algorithm = $row.meta_algorithm
    }
    Write-Host "  meta_algorithm=$algorithm scheduler=$($row.scheduler) seed=$($row.seed) steps=$($row.steps)"
    $cmdArgs = @(
        "scripts\run_stage2_scheduler_training.py",
        "--manifest-csv", $ManifestCsv,
        "--run-id", $row.run_id
    )
    if ($DryRun) {
        $cmdArgs += "--dry-run"
    }
    if ($SaveInterval -gt 0) {
        $cmdArgs += @("--save-interval", "$SaveInterval")
    }
    if ($LogInterval -gt 0) {
        $cmdArgs += @("--log-interval", "$LogInterval")
    }
    & $PythonExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Stage 2 scheduler job failed: $($row.run_id)"
    }
}
