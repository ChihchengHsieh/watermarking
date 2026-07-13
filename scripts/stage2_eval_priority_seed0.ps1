param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string[]]$RunIds = @(
        "scheduler_ats_seed0_steps2000",
        "scheduler_bass_seed0_steps2000",
        "scheduler_bandit_ucb_seed0_steps2000",
        "scheduler_residual_seed0_steps2000"
    ),
    [switch]$IncludeAnchor,
    [switch]$Execute,
    [switch]$DryRun,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

if ($IncludeAnchor -and !($RunIds -contains "scheduler_uniform_seed0_steps2000")) {
    $RunIds = @("scheduler_uniform_seed0_steps2000") + $RunIds
}

$manifest = Import-Csv $ManifestCsv
$selected = @()
$missing = @()
foreach ($runId in $RunIds) {
    $row = $manifest | Where-Object { $_.run_id -eq $runId } | Select-Object -First 1
    if (!$row) {
        $missing += $runId
        continue
    }
    $selected += $row
}
if ($missing.Count -gt 0) {
    throw "Run ID(s) not found in manifest: $($missing -join ', ')"
}

$ready = @()
$status = @()
foreach ($row in $selected) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    $state = if ($hasEval) {
        "already_evaluated"
    } elseif ($hasCheckpoint) {
        "ready"
    } else {
        "missing_checkpoint"
    }

    $status += [pscustomobject]@{
        run_id = $row.run_id
        scheduler = $row.scheduler
        seed = [int]$row.seed
        state = $state
        checkpoint = if ($hasCheckpoint) { "yes" } else { "no" }
        eval_csv = if ($hasEval) { "yes" } else { "no" }
    }

    if ($state -eq "ready") {
        $ready += $row
    }
}

$status | Format-Table -AutoSize

Write-Host ""
if ($ready.Count -eq 0) {
    Write-Host "No selected priority seed-0 runs are ready for evaluation."
    Write-Host "Train first, for example:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1 -Execute"
    return
}

foreach ($row in $ready) {
    $cmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $($row.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
    if ($DryRun) {
        $cmd += " -DryRun"
    }

    if (!$Execute) {
        Write-Host $cmd
        continue
    }

    Write-Host ("=" * 80)
    Write-Host "Executing: $cmd"
    if ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $row.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $row.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Priority seed-0 evaluation failed: $($row.run_id)"
    }
}

if (!$Execute) {
    Write-Host ""
    Write-Host "Preview only. Add -Execute to evaluate ready runs."
}
