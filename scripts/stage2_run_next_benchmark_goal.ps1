param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$IncludeSchedulerSeedExpansion,
    [switch]$AllowConcurrent,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $SchedulerManifest)) {
    throw "Scheduler manifest not found: $SchedulerManifest"
}
if (!(Test-Path $MetaManifest)) {
    throw "Meta-learning manifest not found: $MetaManifest"
}

if (!$AllowConcurrent) {
    $activeStage2Python = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
            }
    )
    if ($activeStage2Python.Count -gt 0) {
        Write-Host "Active Stage 2 Python process detected; not starting another benchmark job."
        $activeStage2Python |
            Select-Object ProcessId, CreationDate, CommandLine |
            Format-List
        Write-Host "Re-run with -AllowConcurrent only if you intentionally want concurrent GPU jobs."
        return
    }
}

function Invoke-CommandLine {
    param(
        [string]$Label,
        [scriptblock]$Command
    )
    Write-Host ("=" * 80)
    Write-Host $Label
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed."
    }
}

$allSchedulerRows = Import-Csv $SchedulerManifest
$uniformAnchor = $allSchedulerRows |
    Where-Object { $_.scheduler -eq "uniform" -and [int]$_.seed -eq 0 } |
    Select-Object -First 1

if ($uniformAnchor -and (Test-Path $uniformAnchor.checkpoint_path) -and !(Test-Path $uniformAnchor.eval_csv)) {
    $evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $($uniformAnchor.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
    $finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1"
    if ($DryRun) {
        $evalCmd += " -DryRun"
    }

    if (!$Execute) {
        Write-Host "Next benchmark-goal step: evaluate uniform seed-0 anchor"
        Write-Host $evalCmd
        Write-Host $finalizeCmd
        Write-Host ""
        Write-Host "This does not train more uniform runs. It only creates the anchor needed for delta_vs_uniform.csv."
        Write-Host "Preview only. Add -Execute to run this step."
        return
    }

    Write-Host ("=" * 80)
    Write-Host "Evaluating uniform seed-0 anchor: $($uniformAnchor.run_id)"
    if ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $uniformAnchor.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $uniformAnchor.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($uniformAnchor.run_id)"
    }

    Invoke-CommandLine "Finalizing scheduler outputs" {
        powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
    }
    return
}

$metaPriority = @{
    "fomaml" = 0
    "maml" = 1
    "anil" = 2
    "reptile" = 3
    "matching_net" = 4
    "proto_net" = 5
    "r2d2_ridge" = 6
}

$metaRows = Import-Csv $MetaManifest | Sort-Object `
    @{ Expression = { if ($metaPriority.ContainsKey($_.meta_algorithm)) { $metaPriority[$_.meta_algorithm] } else { 999 } } }, `
    @{ Expression = { [int]$_.seed } }

$metaRow = $null
foreach ($row in $metaRows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    if (!$hasCheckpoint -or !$hasEval) {
        $metaRow = $row
        break
    }
}

if ($metaRow) {
    $trainCmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $MetaManifest -RunId $($metaRow.run_id)"
    $evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $($metaRow.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
    $finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1"
    if ($DryRun) {
        $trainCmd += " -DryRun"
        $evalCmd += " -DryRun"
    }

    if (!$Execute) {
        Write-Host "Next benchmark-goal step: SOTA/canonical meta-learning algorithm context"
        if (!(Test-Path $metaRow.checkpoint_path)) {
            Write-Host $trainCmd
        } else {
            Write-Host "Checkpoint already exists; training will be skipped."
        }
        Write-Host $evalCmd
        Write-Host $finalizeCmd
        Write-Host ""
        Write-Host "Preview only. Add -Execute to run this step."
        return
    }

    if (!(Test-Path $metaRow.checkpoint_path)) {
        Invoke-CommandLine "Training SOTA/canonical meta-learning row: $($metaRow.run_id)" {
            if ($DryRun) {
                powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $MetaManifest -RunId $metaRow.run_id -DryRun
            } else {
                powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $MetaManifest -RunId $metaRow.run_id
            }
        }
    } else {
        Write-Host "Checkpoint already exists; skipping training: $($metaRow.run_id)"
    }

    Write-Host ("=" * 80)
    Write-Host "Evaluating SOTA/canonical meta-learning row: $($metaRow.run_id)"
    if ($DryRun -and !(Test-Path $metaRow.checkpoint_path)) {
        Write-Host "[DRY-RUN] skipping evaluation because training dry-run does not create checkpoint:"
        Write-Host $metaRow.checkpoint_path
    } elseif ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $metaRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $metaRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
    }
    if (!$DryRun -and $LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($metaRow.run_id)"
    }

    Invoke-CommandLine "Finalizing SOTA/canonical meta-learning context outputs" {
        powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
    }
    return
}

$schedulerPriority = @{
    "ats" = 0
    "bass" = 1
    "asr" = 2
    "gcp_proxy" = 3
    "bandit_ucb" = 4
}

$schedulerRows = $allSchedulerRows |
    Where-Object {
        $schedulerPriority.ContainsKey($_.scheduler) -and
        ($IncludeSchedulerSeedExpansion -or [int]$_.seed -eq 0)
    } |
    Sort-Object `
        @{ Expression = { [int]$_.seed } }, `
        @{ Expression = { $schedulerPriority[$_.scheduler] } }

$schedulerRow = $null
$schedulerMode = ""
foreach ($row in $schedulerRows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    if ($hasCheckpoint -and !$hasEval) {
        $schedulerRow = $row
        $schedulerMode = "eval"
        break
    }
}

if (!$schedulerRow) {
    foreach ($row in $schedulerRows) {
        $hasCheckpoint = Test-Path $row.checkpoint_path
        if (!$hasCheckpoint) {
            $schedulerRow = $row
            $schedulerMode = "train_eval"
            break
        }
    }
}

if ($schedulerRow) {
    $trainCmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $($schedulerRow.run_id)"
    $evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $($schedulerRow.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
    $finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1"
    if ($DryRun) {
        $trainCmd += " -DryRun"
        $evalCmd += " -DryRun"
    }

    if (!$Execute) {
        Write-Host "Next benchmark-goal step: scheduler baseline"
        if ($schedulerMode -eq "train_eval") {
            Write-Host $trainCmd
        } else {
            Write-Host "Checkpoint already exists; training will be skipped."
        }
        Write-Host $evalCmd
        Write-Host $finalizeCmd
        Write-Host ""
        Write-Host "Preview only. Add -Execute to run this step."
        return
    }

    if ($schedulerMode -eq "train_eval") {
        Invoke-CommandLine "Training scheduler row: $($schedulerRow.run_id)" {
            if ($DryRun) {
                powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $schedulerRow.run_id -DryRun
            } else {
                powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $schedulerRow.run_id
            }
        }
    } else {
        Write-Host "Checkpoint already exists; skipping training: $($schedulerRow.run_id)"
    }

    Write-Host ("=" * 80)
    Write-Host "Evaluating scheduler row: $($schedulerRow.run_id)"
    if ($DryRun -and !(Test-Path $schedulerRow.checkpoint_path)) {
        Write-Host "[DRY-RUN] skipping evaluation because training dry-run does not create checkpoint:"
        Write-Host $schedulerRow.checkpoint_path
    } elseif ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $schedulerRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $schedulerRow.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
    }
    if (!$DryRun -and $LASTEXITCODE -ne 0) {
        throw "Evaluation failed: $($schedulerRow.run_id)"
    }

    Invoke-CommandLine "Finalizing scheduler outputs" {
        powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
    }
    return
}

Write-Host "Benchmark goal queue is complete for SOTA/canonical meta-learning context rows and scheduler seed0 rows."
Write-Host "Run the audit:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_audit.ps1"
