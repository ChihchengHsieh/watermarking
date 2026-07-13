param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [switch]$Execute,
    [switch]$DryRun,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$priority = @{
    "ats" = 0
    "bass" = 1
    "bandit_ucb" = 2
    "residual" = 3
}

$rows = Import-Csv $ManifestCsv |
    Where-Object { $priority.ContainsKey($_.scheduler) } |
    Sort-Object `
        @{ Expression = { [int]$_.seed } }, `
        @{ Expression = { $priority[$_.scheduler] } }

$trainRow = $null
foreach ($row in $rows) {
    if (!(Test-Path $row.checkpoint_path)) {
        $trainRow = $row
        break
    }
}

if (!$trainRow) {
    Write-Host "All priority scheduler checkpoints already exist."
    Write-Host "Next evaluation command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1 -Execute"
    return
}

$trainCmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $($trainRow.run_id)"
$evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $($trainRow.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
$finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1"

if ($DryRun) {
    $trainCmd += " -DryRun"
    $evalCmd += " -DryRun"
}

if (!$Execute) {
    Write-Host "Next full scheduler cycle:"
    Write-Host $trainCmd
    Write-Host $evalCmd
    Write-Host $finalizeCmd
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run this train/eval/finalize cycle."
    return
}

Write-Host ("=" * 80)
Write-Host "Training: $($trainRow.run_id)"
if ($DryRun) {
    powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $trainRow.run_id -DryRun
} else {
    powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $trainRow.run_id
}
if ($LASTEXITCODE -ne 0) {
    throw "Training failed: $($trainRow.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Evaluating: $($trainRow.run_id)"
if ($DryRun -and !(Test-Path $trainRow.checkpoint_path)) {
    Write-Host "[DRY-RUN] skipping evaluation because training dry-run does not create checkpoint:"
    Write-Host $trainRow.checkpoint_path
} elseif ($DryRun) {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $trainRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
} else {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $trainRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
}
if (!$DryRun -and $LASTEXITCODE -ne 0) {
    throw "Evaluation failed: $($trainRow.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Finalizing scheduler outputs"
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
if ($LASTEXITCODE -ne 0) {
    throw "Finalize failed after run: $($trainRow.run_id)"
}
