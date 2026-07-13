param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowConcurrent,
    [int]$SaveInterval = 10,
    [int]$LogInterval = 10,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 3
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Pilot manifest not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_pilot_manifest.ps1"
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
        Write-Host "Active Stage 2 Python process detected; not starting meta-learning pilot job."
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

$rows = Import-Csv $ManifestCsv |
    Sort-Object `
        @{ Expression = { if ($priority.ContainsKey($_.meta_algorithm)) { $priority[$_.meta_algorithm] } else { 999 } } }, `
        @{ Expression = { [int]$_.seed } }

$selected = $null
$mode = ""
foreach ($row in $rows) {
    if ((Test-Path $row.checkpoint_path) -and !(Test-Path $row.eval_csv)) {
        $selected = $row
        $mode = "eval"
        break
    }
}
if (!$selected) {
    foreach ($row in $rows) {
        if (!(Test-Path $row.checkpoint_path)) {
            $selected = $row
            $mode = "train_eval"
            break
        }
    }
}

if (!$selected) {
    Write-Host "Meta-learning pilot queue is complete."
    Write-Host "Finalize:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1 -ManifestCsv $ManifestCsv -OutputDir $OutputDir"
    return
}

$trainCmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $($selected.run_id) -SaveInterval $SaveInterval -LogInterval $LogInterval"
$evalCmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -ManifestCsv $ManifestCsv -RunId $($selected.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
$finalizeCmd = "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1 -ManifestCsv $ManifestCsv -OutputDir $OutputDir"
if ($DryRun) {
    $trainCmd += " -DryRun"
    $evalCmd += " -DryRun"
}

if (!$Execute) {
    Write-Host "Next meta-learning pilot step: $($selected.run_id)"
    if ($mode -eq "train_eval") {
        Write-Host $trainCmd
    } else {
        Write-Host "Checkpoint already exists; training will be skipped."
    }
    Write-Host $evalCmd
    Write-Host $finalizeCmd
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run this pilot step."
    return
}

if ($mode -eq "train_eval") {
    Write-Host ("=" * 80)
    Write-Host "Training meta-learning pilot row: $($selected.run_id)"
    if ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id -SaveInterval $SaveInterval -LogInterval $LogInterval -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id -SaveInterval $SaveInterval -LogInterval $LogInterval
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Meta-learning pilot training failed: $($selected.run_id)"
    }
} else {
    Write-Host "Checkpoint already exists; skipping training: $($selected.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Evaluating meta-learning pilot row: $($selected.run_id)"
if ($DryRun -and !(Test-Path $selected.checkpoint_path)) {
    Write-Host "[DRY-RUN] skipping evaluation because training dry-run does not create checkpoint:"
    Write-Host $selected.checkpoint_path
} elseif ($DryRun) {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
} else {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -ManifestCsv $ManifestCsv -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
}
if (!$DryRun -and $LASTEXITCODE -ne 0) {
    throw "Meta-learning pilot evaluation failed: $($selected.run_id)"
}

Write-Host ("=" * 80)
Write-Host "Finalizing meta-learning pilot outputs"
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1 -ManifestCsv $ManifestCsv -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Meta-learning pilot finalize failed after run: $($selected.run_id)"
}
