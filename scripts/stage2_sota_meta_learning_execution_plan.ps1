param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
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

Write-Host "SOTA/canonical meta-learning execution plan"
Write-Host "Manifest: $ManifestCsv"
Write-Host ""

$activeStage2Python = @(
    Get-CimInstance Win32_Process |
        Where-Object {
            $_.Name -eq "python.exe" -and
            $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
        }
)

if ($activeStage2Python.Count -gt 0) {
    Write-Host "Active Stage 2 job detected. Wait before running this plan unless concurrent GPU use is intentional:"
    $activeStage2Python | Select-Object ProcessId, CreationDate, CommandLine | Format-List
    Write-Host ""
}

Write-Host "Preflight:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_readiness.ps1"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_algorithm_units.ps1"
Write-Host ""

Write-Host "Run these formal context rows in order. Prefer the queue wrapper so checkpoints/evaluation/finalization stay manifest-aligned:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute -BatchSize $BatchSize -TestingTimes $TestingTimes"
Write-Host ""

Write-Host "Formal row order:"
foreach ($row in $rows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    $state = if ($hasEval) {
        "evaluated"
    } elseif ($hasCheckpoint) {
        "needs_eval"
    } else {
        "needs_train"
    }
    Write-Host ("- {0}: {1}" -f $row.run_id, $state)
}

Write-Host ""
Write-Host "Finalize tables after rows are evaluated:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1"
Write-Host ""
Write-Host "Completion gate after finalization:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_completion_gate.ps1"
