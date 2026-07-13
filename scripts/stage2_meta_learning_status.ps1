param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
}

$priority = @{
    "fomaml" = 0
    "maml" = 1
    "anil" = 2
    "reptile" = 3
    "matching_net" = 4
    "proto_net" = 5
    "r2d2_ridge" = 6
}

$rows = Import-Csv $ManifestCsv | Sort-Object `
    @{ Expression = { if ($priority.ContainsKey($_.meta_algorithm)) { $priority[$_.meta_algorithm] } else { 999 } } }, `
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
        meta_algorithm = $row.meta_algorithm
        scheduler = $row.scheduler
        seed = [int]$row.seed
        state = $state
        checkpoint = if ($hasCheckpoint) { "yes" } else { "no" }
        eval_csv = if ($hasEval) { "yes" } else { "no" }
    }
}

$status | Format-Table -AutoSize

$nextEval = $status | Where-Object { $_.state -eq "needs_eval" } | Select-Object -First 1
$nextTrain = $status | Where-Object { $_.state -eq "needs_train" } | Select-Object -First 1

Write-Host ""
if ($nextEval) {
    Write-Host "Next evaluation command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $($nextEval.run_id)"
} elseif ($nextTrain) {
    Write-Host "Next training command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $($nextTrain.run_id)"
} else {
    Write-Host "All meta-learning manifest rows have evaluation CSVs."
    Write-Host "Aggregate command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_aggregate_meta_learning_benchmark.ps1"
}
