param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string[]]$SchedulerPriority = @("ats", "bass", "bandit_ucb", "residual"),
    [switch]$IncludeAnchor,
    [switch]$Execute,
    [switch]$DryRun,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$priority = @{}
for ($i = 0; $i -lt $SchedulerPriority.Count; $i++) {
    $priority[$SchedulerPriority[$i]] = $i
}
if ($IncludeAnchor) {
    $priority["uniform"] = -1
}

$rows = Import-Csv $ManifestCsv |
    Where-Object { $priority.ContainsKey($_.scheduler) } |
    Sort-Object `
        @{ Expression = { [int]$_.seed } }, `
        @{ Expression = { $priority[$_.scheduler] } }

$selected = $null
foreach ($row in $rows) {
    $hasCheckpoint = Test-Path $row.checkpoint_path
    $hasEval = Test-Path $row.eval_csv
    if ($hasCheckpoint -and !$hasEval) {
        $selected = $row
        break
    }
}

if (!$selected) {
    Write-Host "No selected scheduler run is ready for evaluation."
    Write-Host "Train the next priority scheduler first:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1 -Execute"
    return
}

$cmd = "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $($selected.run_id) -BatchSize $BatchSize -TestingTimes $TestingTimes"
if ($DryRun) {
    $cmd += " -DryRun"
}

if (!$Execute) {
    Write-Host $cmd
    Write-Host ""
    Write-Host "Preview only. Add -Execute to evaluate this run."
    return
}

Write-Host ("=" * 80)
Write-Host "Executing: $cmd"
if ($DryRun) {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes -DryRun
} else {
    powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId $selected.run_id -BatchSize $BatchSize -TestingTimes $TestingTimes
}
if ($LASTEXITCODE -ne 0) {
    throw "Scheduler evaluation failed: $($selected.run_id)"
}
