param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowConcurrent,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
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
        Write-Host "Active Stage 2 Python process detected; not starting SOTA meta-learning context job."
        $activeStage2Python |
            Select-Object ProcessId, CreationDate, CommandLine |
            Format-List
        Write-Host "Re-run with -AllowConcurrent only if you intentionally want concurrent GPU jobs."
        return
    }
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

$selected = $null
foreach ($row in $rows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    if (!$hasCheckpoint -or !$hasEval) {
        $selected = $row
        break
    }
}

if (!$selected) {
    Write-Host "All meta-learning context rows already have checkpoints and evaluation CSVs."
    Write-Host "Aggregate command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1"
    return
}

$trainCmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $($selected.run_id)"
$evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $($selected.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
$finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1"

if ($DryRun) {
    $trainCmd += " -DryRun"
    $evalCmd += " -DryRun"
}

if (!$Execute) {
    Write-Host "Next full meta-learning context cycle:"
    if (!(Test-Path $selected.checkpoint_path)) {
        Write-Host $trainCmd
    } else {
        Write-Host "Checkpoint already exists; training will be skipped."
    }
    Write-Host $evalCmd
    Write-Host $finalizeCmd
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run this train/eval/finalize cycle."
    return
}

if (!(Test-Path $selected.checkpoint_path)) {
    Write-Host ("=" * 80)
    Write-Host "Training: $($selected.run_id)"
    if ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed: $($selected.run_id)"
    }
} else {
    Write-Host "Checkpoint already exists; skipping training: $($selected.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Evaluating: $($selected.run_id)"
if ($DryRun -and !(Test-Path $selected.checkpoint_path)) {
    Write-Host "[DRY-RUN] skipping evaluation because training dry-run does not create checkpoint:"
    Write-Host $selected.checkpoint_path
} elseif ($DryRun) {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
} else {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
}
if (!$DryRun -and $LASTEXITCODE -ne 0) {
    throw "Evaluation failed: $($selected.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Finalizing meta-learning context outputs"
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
if ($LASTEXITCODE -ne 0) {
    throw "Finalize failed after run: $($selected.run_id)"
}
