param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [switch]$PreferAnchorEval
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$priority = @{
    "uniform" = 0
    "ats" = 1
    "bass" = 2
    "bandit_ucb" = 3
    "residual" = 4
}

$rows = Import-Csv $ManifestCsv | Sort-Object `
    @{ Expression = { if ($priority.ContainsKey($_.scheduler)) { $priority[$_.scheduler] } else { 999 } } }, `
    @{ Expression = { [int]$_.seed } }

$status = foreach ($row in $rows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    $state = if ($hasEval) {
        "evaluated"
    } elseif ($hasCheckpoint) {
        "needs_eval"
    } else {
        "needs_train"
    }

    [pscustomobject]@{
        run_id = $row.run_id
        scheduler = $row.scheduler
        seed = [int]$row.seed
        state = $state
        checkpoint = if ($hasCheckpoint) { "yes" } else { "no" }
        eval_csv = if ($hasEval) { "yes" } else { "no" }
    }
}

$status | Format-Table -AutoSize

$nextEval = $status | Where-Object { $_.state -eq "needs_eval" } | Select-Object -First 1
$nextAdaptiveTrain = $status |
    Where-Object { $_.state -eq "needs_train" -and $_.scheduler -ne "uniform" -and $_.scheduler -ne "cycle" } |
    Select-Object -First 1
$nextTrain = $status | Where-Object { $_.state -eq "needs_train" } | Select-Object -First 1
$manifestArg = "-ManifestCsv $ManifestCsv"

Write-Host ""
if ($PreferAnchorEval -and $nextEval) {
    Write-Host "Next evaluation command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 $manifestArg -RunId $($nextEval.run_id)"
} elseif ($nextAdaptiveTrain) {
    if ($nextEval) {
        Write-Host "Pending anchor evaluation:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 $manifestArg -RunId $($nextEval.run_id)"
        Write-Host ""
    }
    Write-Host "Next SOTA/adaptive training command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 $manifestArg -RunId $($nextAdaptiveTrain.run_id)"
} elseif ($nextTrain) {
    Write-Host "Next training command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 $manifestArg -RunId $($nextTrain.run_id)"
} elseif ($nextEval) {
    Write-Host "Next evaluation command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 $manifestArg -RunId $($nextEval.run_id)"
} else {
    Write-Host "All scheduler manifest rows have evaluation CSVs."
    Write-Host "Aggregate command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_aggregate_scheduler_benchmark.ps1 $manifestArg"
}
