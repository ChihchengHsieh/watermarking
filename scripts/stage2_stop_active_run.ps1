param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [switch]$Execute,
    [switch]$ForceNoCheckpoint
)

$ErrorActionPreference = "Stop"

$matches = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py" -and
            $_.CommandLine -match [regex]::Escape($RunId)
        }
)

if ($matches.Count -eq 0) {
    Write-Host "No active Stage 2 Python process found for RunId: $RunId"
    return
}

Write-Host "Matched active Stage 2 process(es):"
$matches | Select-Object ProcessId, CreationDate, CommandLine | Format-List

$candidateRunDirs = @(
    "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/$RunId",
    "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/$RunId",
    "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/$RunId"
)
$checkpointCandidates = foreach ($runDir in $candidateRunDirs) {
    Join-Path $runDir "checkpoints/latest.pth"
    Join-Path $runDir "checkpoints/final.pth"
}
$existingCheckpoints = @($checkpointCandidates | Where-Object { Test-Path $_ })

if ($existingCheckpoints.Count -gt 0) {
    Write-Host "Existing checkpoint(s):"
    $existingCheckpoints | ForEach-Object { Write-Host "  $_" }
} else {
    Write-Host "No latest/final checkpoint found for $RunId."
    Write-Host "Stopping now will discard progress since the last checkpoint."
}

if (!$Execute) {
    Write-Host "Preview only. Add -Execute to stop the matched process(es)."
    return
}

if ($existingCheckpoints.Count -eq 0 -and !$ForceNoCheckpoint) {
    throw "Refusing to stop $RunId because no latest/final checkpoint exists. Re-run with -ForceNoCheckpoint only if discarding current progress is intentional."
}

foreach ($proc in $matches) {
    Write-Host "Stopping process $($proc.ProcessId) for RunId=$RunId"
    Stop-Process -Id $proc.ProcessId -Force
}

Write-Host "Stopped $($matches.Count) process(es)."
