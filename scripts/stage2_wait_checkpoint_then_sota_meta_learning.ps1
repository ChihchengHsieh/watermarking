param(
    [string]$RunId = "scheduler_bandit_ucb_seed0_steps2000",
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 360,
    [switch]$Execute,
    [switch]$EvalLatest,
    [switch]$StopAfterCheckpoint,
    [switch]$StartPilot,
    [switch]$StartFullContext,
    [switch]$AllowConcurrent
)

$ErrorActionPreference = "Stop"

if ($StartPilot -and $StartFullContext) {
    throw "Choose only one: -StartPilot or -StartFullContext."
}

Write-Host "Wait for checkpoint, then switch to SOTA/canonical meta-learning"
Write-Host ("=" * 80)
Write-Host "RunId=$RunId"
Write-Host "ManifestCsv=$ManifestCsv"
Write-Host "Execute=$Execute EvalLatest=$EvalLatest StopAfterCheckpoint=$StopAfterCheckpoint StartPilot=$StartPilot StartFullContext=$StartFullContext"
Write-Host ""

if (!$Execute) {
    Write-Host "Preview command sequence:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_for_checkpoint.ps1 -RunId $RunId -PollSeconds $PollSeconds -TimeoutMinutes $TimeoutMinutes"
    if ($EvalLatest) {
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -ManifestCsv $ManifestCsv -RunId $RunId"
    }
    if ($StopAfterCheckpoint) {
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId $RunId -Execute"
    }
    if ($StartPilot) {
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1 -Execute"
    }
    if ($StartFullContext) {
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1 -Execute"
    }
    Write-Host ""
    Write-Host "Preview only. Add -Execute to wait and run selected follow-up steps."
    return
}

powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_for_checkpoint.ps1 `
    -RunId $RunId `
    -PollSeconds $PollSeconds `
    -TimeoutMinutes $TimeoutMinutes
if ($LASTEXITCODE -ne 0) {
    throw "Checkpoint wait failed for $RunId."
}

if ($EvalLatest) {
    Write-Host ("=" * 80)
    Write-Host "Running latest-checkpoint quick-look evaluation"
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 `
        -ManifestCsv $ManifestCsv `
        -RunId $RunId
    if ($LASTEXITCODE -ne 0) {
        throw "Latest-checkpoint quick-look evaluation failed for $RunId."
    }
}

if ($StopAfterCheckpoint) {
    Write-Host ("=" * 80)
    Write-Host "Stopping active run after checkpoint"
    powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 `
        -RunId $RunId `
        -Execute
    if ($LASTEXITCODE -ne 0) {
        throw "Stopping active run failed for $RunId."
    }
}

if ($StartPilot) {
    Write-Host ("=" * 80)
    Write-Host "Starting all-row SOTA/canonical meta-learning pilot"
    $argsList = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_all_sota_meta_learning_pilot.ps1",
        "-Execute"
    )
    if ($AllowConcurrent) { $argsList += "-AllowConcurrent" }
    powershell @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "All-row SOTA/canonical meta-learning pilot failed."
    }
}

if ($StartFullContext) {
    Write-Host ("=" * 80)
    Write-Host "Starting all-row SOTA/canonical meta-learning full context"
    $argsList = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_all_sota_meta_learning_context.ps1",
        "-Execute"
    )
    if ($AllowConcurrent) { $argsList += "-AllowConcurrent" }
    powershell @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "All-row SOTA/canonical meta-learning full context failed."
    }
}
