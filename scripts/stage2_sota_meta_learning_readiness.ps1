param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
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

$rows = @(Import-Csv $ManifestCsv | Sort-Object `
    @{ Expression = { if ($priority.ContainsKey($_.meta_algorithm)) { $priority[$_.meta_algorithm] } else { 999 } } }, `
    @{ Expression = { [int]$_.seed } })

Write-Host "SOTA/canonical meta-learning context readiness"
Write-Host "Manifest: $ManifestCsv"
Write-Host "Formal rows: $($rows.Count)"
Write-Host ""

$formalStatus = foreach ($row in $rows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    $state = if ($hasEval) {
        "evaluated"
    } elseif ($hasCheckpoint) {
        "needs_eval"
    } else {
        "needs_train"
    }

    [pscustomobject]@{
        run_id = $row.run_id
        meta_algorithm = $row.meta_algorithm
        scheduler = $row.scheduler
        seed = [int]$row.seed
        steps = [int]$row.steps
        state = $state
    }
}

$formalStatus | Format-Table -AutoSize

$formalRunIds = @{}
foreach ($row in $rows) {
    $formalRunIds[$row.run_id] = $true
}

Write-Host ""
Write-Host "Ignored legacy output directories not present in the formal manifest:"
$legacyDirs = @()
if (Test-Path $OutputDir) {
    $legacyDirs = @(
        Get-ChildItem -Path $OutputDir -Directory |
            Where-Object { $_.Name -like "meta_*_uniform_*" -and !$formalRunIds.ContainsKey($_.Name) } |
            Sort-Object Name
    )
}

if ($legacyDirs.Count -eq 0) {
    Write-Host "none"
} else {
    $legacyDirs | Select-Object Name, LastWriteTime | Format-Table -AutoSize
}

Write-Host ""
$activeStage2Python = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
        }
)

if ($activeStage2Python.Count -gt 0) {
    Write-Host "Active Stage 2 Python process detected; do not start a SOTA/canonical meta-learning context row unless concurrent GPU use is intentional."
    $activeStage2Python | Select-Object ProcessId, CreationDate, CommandLine | Format-List
} else {
    Write-Host "No active Stage 2 Python process detected."
}

$nextEval = $formalStatus | Where-Object { $_.state -eq "needs_eval" } | Select-Object -First 1
$nextTrain = $formalStatus | Where-Object { $_.state -eq "needs_train" } | Select-Object -First 1

Write-Host ""
if ($nextEval) {
    Write-Host "Next required SOTA/canonical meta-learning command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId $($nextEval.run_id)"
} elseif ($nextTrain) {
    Write-Host "Next required SOTA/canonical meta-learning command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv $ManifestCsv -RunId $($nextTrain.run_id)"
} else {
    Write-Host "All formal SOTA/canonical meta-learning context rows have evaluation CSVs."
    Write-Host "Finalize command:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1"
}
