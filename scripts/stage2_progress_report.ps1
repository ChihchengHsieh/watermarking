param(
    [string]$RunId = "scheduler_uniform_seed0_steps2000",
    [string]$BenchmarkDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark"
)

$ErrorActionPreference = "Stop"

$runDir = Join-Path $BenchmarkDir $RunId
$timingPath = Join-Path $runDir "timing.csv"
$schedulerPath = Join-Path $runDir "scheduler.jsonl"
$latestCkpt = Join-Path $runDir "checkpoints/latest.pth"
$finalCkpt = Join-Path $runDir "checkpoints/final.pth"
$reportPath = Join-Path $runDir "progress_report.md"

if (!(Test-Path $runDir)) {
    throw "Run directory not found: $runDir"
}
if (!(Test-Path $timingPath)) {
    throw "Timing log not found yet: $timingPath"
}

$rows = Import-Csv $timingPath
if ($rows.Count -eq 0) {
    throw "Timing log has no training rows yet: $timingPath"
}

$last = $rows[-1]
$recent = $rows | Select-Object -Last ([Math]::Min(20, $rows.Count))
$avgRecentLoss = ($recent | Measure-Object meta_loss -Average).Average
$avgRecentStepSec = ($recent | Measure-Object step_time_sec -Average).Average
$taskCounts = @{}
foreach ($row in $rows) {
    foreach ($task in ($row.tasks -split "\|")) {
        if ($task -eq "") { continue }
        if (!$taskCounts.ContainsKey($task)) { $taskCounts[$task] = 0 }
        $taskCounts[$task] += 1
    }
}

$taskLines = $taskCounts.GetEnumerator() |
    Sort-Object Name |
    ForEach-Object { "- ``$($_.Name)``: $($_.Value)" }

$schedulerTail = @()
if (Test-Path $schedulerPath) {
    $schedulerTail = Get-Content $schedulerPath -Tail 3
}

$checkpointStatus = @()
$checkpointStatus += "- latest checkpoint: $(if (Test-Path $latestCkpt) { 'exists' } else { 'not yet' })"
$checkpointStatus += "- final checkpoint: $(if (Test-Path $finalCkpt) { 'exists' } else { 'not yet' })"

$lines = @(
    "# Stage 2 Training Progress",
    "",
    "- run id: ``$RunId``",
    "- current step: ``$($last.global_step)``",
    "- latest meta loss: ``$($last.meta_loss)``",
    "- recent mean meta loss: ``$('{0:F6}' -f [double]$avgRecentLoss)``",
    "- latest grad norm: ``$($last.grad_norm)``",
    "- recent mean step time: ``$('{0:F2}' -f [double]$avgRecentStepSec)s``",
    "",
    "## Checkpoints",
    ""
)
$lines += $checkpointStatus
$lines += @(
    "",
    "## Sampled Tasks So Far",
    ""
)
$lines += $taskLines
$lines += @(
    "",
    "## Recent Scheduler Records",
    ""
)

if ($schedulerTail.Count -gt 0) {
    $lines += '```json'
    $lines += $schedulerTail
    $lines += '```'
} else {
    $lines += "No scheduler records written yet."
}

$lines | Set-Content -Path $reportPath -Encoding UTF8

Write-Host "Wrote progress report to $reportPath"
Write-Host "Current step: $($last.global_step)"
Write-Host "Latest meta loss: $($last.meta_loss)"
$recentLossText = "{0:F6}" -f [double]$avgRecentLoss
Write-Host "Recent mean meta loss: $recentLossText"
