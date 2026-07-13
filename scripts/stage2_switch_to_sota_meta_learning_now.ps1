param(
    [string]$RunId = "scheduler_bandit_ucb_seed0_steps2000",
    [switch]$Execute,
    [switch]$ForceNoCheckpoint,
    [switch]$StartPilot,
    [switch]$StartFullContext,
    [switch]$AllowConcurrent
)

$ErrorActionPreference = "Stop"

if ($StartPilot -and $StartFullContext) {
    throw "Choose only one: -StartPilot or -StartFullContext."
}

Write-Host "Switch to SOTA/canonical meta-learning path"
Write-Host ("=" * 80)
Write-Host "Target active run to stop: $RunId"
Write-Host "Execute=$Execute ForceNoCheckpoint=$ForceNoCheckpoint StartPilot=$StartPilot StartFullContext=$StartFullContext"
Write-Host ""

$stopArgs = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\stage2_stop_active_run.ps1",
    "-RunId", $RunId
)
if ($Execute) { $stopArgs += "-Execute" }
if ($ForceNoCheckpoint) { $stopArgs += "-ForceNoCheckpoint" }

Write-Host "Stop step:"
powershell @stopArgs
if ($LASTEXITCODE -ne 0) {
    throw "Stop step failed or was refused."
}

Write-Host ""
Write-Host "SOTA/canonical meta-learning readiness:"
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_readiness.ps1
if ($LASTEXITCODE -ne 0) {
    throw "SOTA/canonical meta-learning readiness check failed."
}

if (!$Execute) {
    Write-Host ""
    Write-Host "Preview only. To discard a no-checkpoint active run and start the pilot:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1 -Execute -ForceNoCheckpoint -StartPilot"
    Write-Host ""
    Write-Host "To discard a no-checkpoint active run and start the full context row:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1 -Execute -ForceNoCheckpoint -StartFullContext"
    return
}

if ($StartPilot) {
    $pilotArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_next_meta_learning_pilot.ps1"
    )
    if ($AllowConcurrent) { $pilotArgs += "-AllowConcurrent" }
    Write-Host ""
    Write-Host "Starting SOTA/canonical meta-learning pilot:"
    powershell @pilotArgs -Execute
    if ($LASTEXITCODE -ne 0) {
        throw "SOTA/canonical meta-learning pilot failed."
    }
    return
}

if ($StartFullContext) {
    $fullArgs = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_next_sota_meta_learning_context.ps1"
    )
    if ($AllowConcurrent) { $fullArgs += "-AllowConcurrent" }
    Write-Host ""
    Write-Host "Starting full SOTA/canonical meta-learning context row:"
    powershell @fullArgs -Execute
    if ($LASTEXITCODE -ne 0) {
        throw "SOTA/canonical meta-learning context row failed."
    }
    return
}

Write-Host ""
Write-Host "Active run stop phase completed. No new job started."
Write-Host "Start the pilot:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute"
Write-Host "Or start the full context row:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute"
