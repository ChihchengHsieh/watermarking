param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv"
)

$ErrorActionPreference = "Stop"

function Get-RunState {
    param($Row)
    if (Test-Path $Row.eval_csv) {
        return "evaluated"
    }
    if (Test-Path $Row.checkpoint_path) {
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

if (!(Test-Path $SchedulerManifest)) {
    throw "Missing scheduler pilot manifest: $SchedulerManifest. Generate it with scripts\stage2_scheduler_pilot_manifest.ps1"
}
if (!(Test-Path $MetaManifest)) {
    throw "Missing meta-learning pilot manifest: $MetaManifest. Generate it with scripts\stage2_meta_learning_pilot_manifest.ps1"
}

$schedulerPriority = @{
    "uniform" = -1
    "ats" = 0
    "bass" = 1
    "bandit_ucb" = 2
    "residual" = 3
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

$schedulerRows = Import-Csv $SchedulerManifest |
    Sort-Object @{ Expression = { if ($schedulerPriority.ContainsKey($_.scheduler)) { $schedulerPriority[$_.scheduler] } else { 999 } } }
$metaRows = Import-Csv $MetaManifest |
    Sort-Object @{ Expression = { if ($metaPriority.ContainsKey($_.meta_algorithm)) { $metaPriority[$_.meta_algorithm] } else { 999 } } }

$status = @()
$status += foreach ($row in $metaRows) {
    [pscustomobject]@{
        group = "meta_pilot"
        run_id = $row.run_id
        method = $row.meta_algorithm
        seed = [int]$row.seed
        steps = [int]$row.steps
        state = Get-RunState $row
        partial_attacks = Get-PartialAttackCount $row
    }
}
$status += foreach ($row in $schedulerRows) {
    [pscustomobject]@{
        group = "scheduler_pilot"
        run_id = $row.run_id
        method = $row.scheduler
        seed = [int]$row.seed
        steps = [int]$row.steps
        state = Get-RunState $row
        partial_attacks = Get-PartialAttackCount $row
    }
}

$status | Format-Table -AutoSize

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
    $activeStage2Python |
        Select-Object ProcessId, CreationDate, CommandLine |
        Format-List
}

Write-Host ""
Write-Host "Next pilot-goal command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 -Execute"
Write-Host ""
Write-Host "Preview next pilot-goal action:"
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 `
    -SchedulerManifest $SchedulerManifest `
    -MetaManifest $MetaManifest
