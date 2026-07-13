param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $SchedulerManifest)) {
    throw "Scheduler manifest not found: $SchedulerManifest"
}
if (!(Test-Path $MetaManifest)) {
    throw "Meta-learning manifest not found: $MetaManifest"
}

function Get-RunState {
    param($Row)
    $hasCheckpoint = Test-Path $Row.checkpoint_path
    $hasEval = Test-Path $Row.eval_csv
    if ($hasEval) {
        return "evaluated"
    }
    if ($hasCheckpoint) {
        return "needs_eval"
    }
    return "needs_train"
}

function Get-PartialAttackCount {
    param($Row)
    $evalDir = Split-Path -Parent $Row.eval_csv
    $artifactDir = Join-Path $evalDir "eval_artifacts"
    if (!(Test-Path $artifactDir)) {
        return 0
    }
    return @(
        Get-ChildItem $artifactDir -Recurse -Filter "eval_results.pt" -File -ErrorAction SilentlyContinue
    ).Count
}

$schedulerRows = Import-Csv $SchedulerManifest
$metaRows = Import-Csv $MetaManifest

$uniformAnchor = $schedulerRows |
    Where-Object { $_.scheduler -eq "uniform" -and [int]$_.seed -eq 0 } |
    Select-Object -First 1

$schedulerPriority = @{
    "ats" = 0
    "bass" = 1
    "bandit_ucb" = 2
    "residual" = 3
}

$prioritySchedulers = $schedulerRows |
    Where-Object { $schedulerPriority.ContainsKey($_.scheduler) -and [int]$_.seed -eq 0 } |
    Sort-Object @{ Expression = { $schedulerPriority[$_.scheduler] } }

$metaPriority = @{
    "fomaml" = 0
    "maml" = 1
    "anil" = 2
    "reptile" = 3
    "matching_net" = 4
    "proto_net" = 5
    "r2d2_ridge" = 6
}

$metaContext = $metaRows |
    Sort-Object @{ Expression = { if ($metaPriority.ContainsKey($_.meta_algorithm)) { $metaPriority[$_.meta_algorithm] } else { 999 } } }

$rows = @()
if ($uniformAnchor) {
    $rows += [pscustomobject]@{
        group = "anchor"
        run_id = $uniformAnchor.run_id
        method = $uniformAnchor.scheduler
        state = Get-RunState $uniformAnchor
        partial_attacks = Get-PartialAttackCount $uniformAnchor
    }
}

foreach ($row in $metaContext) {
    $rows += [pscustomobject]@{
        group = "meta_context"
        run_id = $row.run_id
        method = $row.meta_algorithm
        state = Get-RunState $row
        partial_attacks = Get-PartialAttackCount $row
    }
}

foreach ($row in $prioritySchedulers) {
    $rows += [pscustomobject]@{
        group = "scheduler_seed0"
        run_id = $row.run_id
        method = $row.scheduler
        state = Get-RunState $row
        partial_attacks = Get-PartialAttackCount $row
    }
}

$rows | Format-Table -AutoSize

$activeStage2Python = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
        }
)

if ($activeStage2Python.Count -gt 0) {
    Write-Host ""
    Write-Host "Active Stage 2 Python process(es):"
    $processRows = foreach ($proc in $activeStage2Python) {
        $runId = ""
        if ($proc.CommandLine -match "--run-id\s+([^\s]+)") {
            $runId = $Matches[1]
        }
        $timingPath = ""
        $lastWrite = $null
        if ($runId) {
            $schedulerRunDir = Join-Path (Split-Path -Parent $SchedulerManifest) $runId
            $metaRunDir = Join-Path (Split-Path -Parent $MetaManifest) $runId
            foreach ($candidate in @($schedulerRunDir, $metaRunDir)) {
                $candidateTiming = Join-Path $candidate "timing.csv"
                if (Test-Path $candidateTiming) {
                    $timingPath = $candidateTiming
                    $lastWrite = (Get-Item $candidateTiming).LastWriteTime
                    break
                }
            }
        }
        $targetSteps = $null
        $currentStep = $null
        $avgStepSec = $null
        $etaHours = $null
        $runDir = ""
        $latestCheckpoint = ""
        $hasLatestCheckpoint = $false
        $sourceManifest = ""
        $saveInterval = 100
        if ($proc.CommandLine -match "--save-interval\s+([0-9]+)") {
            $saveInterval = [int]$Matches[1]
        }
        $nextCheckpointStep = $null
        $etaToCheckpointHours = $null
        $manifestRow = @($schedulerRows + $metaRows | Where-Object { $_.run_id -eq $runId } | Select-Object -First 1)
        if ($manifestRow.Count -gt 0 -and $manifestRow[0].steps) {
            $targetSteps = [int]$manifestRow[0].steps
            $runDir = $manifestRow[0].run_dir
            if (@($schedulerRows | Where-Object { $_.run_id -eq $runId }).Count -gt 0) {
                $sourceManifest = $SchedulerManifest
            } elseif (@($metaRows | Where-Object { $_.run_id -eq $runId }).Count -gt 0) {
                $sourceManifest = $MetaManifest
            }
            if ($runDir) {
                $latestCheckpoint = Join-Path $runDir "checkpoints/latest.pth"
                $hasLatestCheckpoint = Test-Path $latestCheckpoint
            }
        }
        if ($timingPath -and (Test-Path $timingPath)) {
            $timingRows = @(Import-Csv $timingPath)
            if ($timingRows.Count -gt 0) {
                $currentStep = [int]$timingRows[-1].global_step
                $recent = @($timingRows | Select-Object -Last ([Math]::Min(10, $timingRows.Count)))
                $avgStepSec = [math]::Round((($recent | ForEach-Object { [double]$_.step_time_sec } | Measure-Object -Average).Average), 1)
                if ($targetSteps -and $currentStep -lt $targetSteps -and $avgStepSec) {
                    $etaHours = [math]::Round((($targetSteps - $currentStep) * $avgStepSec) / 3600.0, 1)
                }
                if ($saveInterval -gt 0 -and $avgStepSec) {
                    $nextCheckpointStep = [math]::Ceiling(($currentStep + 1) / $saveInterval) * $saveInterval
                    if ($targetSteps) {
                        $nextCheckpointStep = [math]::Min($nextCheckpointStep, $targetSteps)
                    }
                    if ($nextCheckpointStep -gt $currentStep) {
                        $etaToCheckpointHours = [math]::Round((($nextCheckpointStep - $currentStep) * $avgStepSec) / 3600.0, 1)
                    }
                }
            }
        }
        [pscustomobject]@{
            ProcessId = $proc.ProcessId
            CreationDate = $proc.CreationDate
            RunId = $runId
            CurrentStep = $currentStep
            TargetSteps = $targetSteps
            AvgRecentStepSec = $avgStepSec
            EtaHours = $etaHours
            SaveInterval = $saveInterval
            NextCheckpointStep = $nextCheckpointStep
            EtaToCheckpointHours = $etaToCheckpointHours
            HasLatestCheckpoint = $hasLatestCheckpoint
            LatestCheckpoint = $latestCheckpoint
            ManifestCsv = $sourceManifest
            TimingLastWrite = $lastWrite
            MinutesSinceTimingWrite = if ($lastWrite) { [math]::Round(((Get-Date) - $lastWrite).TotalMinutes, 1) } else { $null }
            CommandLine = $proc.CommandLine
        }
    }
    $processRows | Format-List
    $quickLookRows = @($processRows | Where-Object { $_.RunId -and $_.HasLatestCheckpoint })
    if ($quickLookRows.Count -gt 0) {
        Write-Host ""
        Write-Host "Latest-checkpoint quick-look evaluation command(s):"
        foreach ($row in $quickLookRows) {
            Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -ManifestCsv $($row.ManifestCsv) -RunId $($row.RunId)"
        }
    }
}

Write-Host ""
Write-Host "Next benchmark-goal command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1 -Execute"
Write-Host ""
Write-Host "Preview next action:"
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1
