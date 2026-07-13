param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$PilotManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [string]$MetaPilotManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv",
    [double]$LongRunEtaHoursThreshold = 24.0
)

$ErrorActionPreference = "Stop"

function Get-ActiveStage2Python {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
            }
    )
}

function Get-RunIdFromCommandLine {
    param([string]$CommandLine)
    if ($CommandLine -match "--run-id\s+([^\s]+)") {
        return $Matches[1]
    }
    return ""
}

function Get-ManifestRow {
    param([string]$RunId)
    foreach ($manifest in @($SchedulerManifest, $PilotManifest, $MetaManifest, $MetaPilotManifest)) {
        if (!(Test-Path $manifest)) {
            continue
        }
        $row = Import-Csv $manifest | Where-Object { $_.run_id -eq $RunId } | Select-Object -First 1
        if ($row) {
            $row | Add-Member -NotePropertyName manifest_csv -NotePropertyValue $manifest -Force
            return $row
        }
    }
    return $null
}

function Get-RunTimingSummary {
    param($Row, [string]$CommandLine = "")
    if (!$Row) {
        return $null
    }
    $timingPath = Join-Path $Row.run_dir "timing.csv"
    if (!(Test-Path $timingPath)) {
        return $null
    }
    $timingRows = @(Import-Csv $timingPath)
    if ($timingRows.Count -eq 0) {
        return $null
    }
    $recent = @($timingRows | Select-Object -Last ([Math]::Min(10, $timingRows.Count)))
    $avgStepSec = (($recent | ForEach-Object { [double]$_.step_time_sec } | Measure-Object -Average).Average)
    $currentStep = [int]$timingRows[-1].global_step
    $targetSteps = [int]$Row.steps
    $etaHours = (($targetSteps - $currentStep) * $avgStepSec) / 3600.0
    $saveInterval = 100
    if ($CommandLine -match "--save-interval\s+([0-9]+)") {
        $saveInterval = [int]$Matches[1]
    }
    $nextCheckpointStep = $null
    $etaToCheckpointHours = $null
    if ($saveInterval -gt 0) {
        $nextCheckpointStep = [math]::Ceiling(($currentStep + 1) / $saveInterval) * $saveInterval
        $nextCheckpointStep = [math]::Min($nextCheckpointStep, $targetSteps)
        if ($nextCheckpointStep -gt $currentStep) {
            $etaToCheckpointHours = (($nextCheckpointStep - $currentStep) * $avgStepSec) / 3600.0
        }
    }
    return [pscustomobject]@{
        current_step = $currentStep
        target_steps = $targetSteps
        avg_recent_step_sec = [math]::Round($avgStepSec, 1)
        eta_hours = [math]::Round($etaHours, 1)
        save_interval = $saveInterval
        next_checkpoint_step = $nextCheckpointStep
        eta_to_checkpoint_hours = if ($etaToCheckpointHours -ne $null) { [math]::Round($etaToCheckpointHours, 1) } else { $null }
        timing_last_write = (Get-Item $timingPath).LastWriteTime
    }
}

function Get-LatestCheckpointPath {
    param($Row)
    if (!$Row) {
        return ""
    }
    $path = Join-Path $Row.run_dir "checkpoints/latest.pth"
    if (Test-Path $path) {
        return $path
    }
    return ""
}

$active = Get-ActiveStage2Python

Write-Host "Stage 2 Benchmark Decision Report"
Write-Host ("=" * 80)

if ($active.Count -eq 0) {
    Write-Host "No active Stage 2 Python process detected."
    Write-Host ""
    Write-Host "Recommended next SOTA/canonical meta-learning pilot command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute"
    Write-Host ""
    Write-Host "Recommended total pilot queue command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 -Execute"
    Write-Host ""
    Write-Host "Recommended status command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_pilot_goal_status.ps1"
    return
}

foreach ($proc in $active) {
    $runId = Get-RunIdFromCommandLine $proc.CommandLine
    $row = Get-ManifestRow $runId
    $timing = Get-RunTimingSummary $row $proc.CommandLine
    $latestCheckpoint = Get-LatestCheckpointPath $row

    Write-Host "Active process:"
    [pscustomobject]@{
        process_id = $proc.ProcessId
        run_id = $runId
        created = $proc.CreationDate
        manifest_steps = if ($row) { $row.steps } else { "" }
        current_step = if ($timing) { $timing.current_step } else { "" }
        eta_hours = if ($timing) { $timing.eta_hours } else { "" }
        save_interval = if ($timing) { $timing.save_interval } else { "" }
        next_checkpoint_step = if ($timing) { $timing.next_checkpoint_step } else { "" }
        eta_to_checkpoint_hours = if ($timing) { $timing.eta_to_checkpoint_hours } else { "" }
        latest_checkpoint = $latestCheckpoint
    } | Format-List

    if ($latestCheckpoint) {
        Write-Host "Latest checkpoint exists. Quick-look eval command:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -ManifestCsv $($row.manifest_csv) -RunId $runId"
    }

    if ($timing -and $timing.eta_hours -gt $LongRunEtaHoursThreshold) {
        Write-Host "Recommendation: this run is too long for first-pass benchmarking."
        if ($timing.eta_to_checkpoint_hours -ne $null) {
            Write-Host "Next checkpoint estimate:"
            Write-Host "  step $($timing.next_checkpoint_step), about $($timing.eta_to_checkpoint_hours) hours away"
        }
        Write-Host "Preview stop:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId $runId"
        Write-Host "Stop if confirmed:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId $runId -Execute"
        Write-Host "If no latest/final checkpoint exists and discarding progress is intentional:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId $runId -Execute -ForceNoCheckpoint"
        Write-Host "Then start the SOTA/canonical meta-learning pilot:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute"
        Write-Host "Or use the total pilot queue:"
        Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 -Execute"
    } else {
        Write-Host "Recommendation: let the active process continue, or inspect status again later."
    }
    Write-Host ""
}
