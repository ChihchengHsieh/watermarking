param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string[]]$SchedulerPriority = @("ats", "bass", "bandit_ucb", "residual"),
    [int]$Count = 1,
    [switch]$Execute,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($Count -lt 1) {
    throw "-Count must be at least 1."
}
if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$priority = @{}
for ($i = 0; $i -lt $SchedulerPriority.Count; $i++) {
    $priority[$SchedulerPriority[$i]] = $i
}

$rows = Import-Csv $ManifestCsv |
    Where-Object { $priority.ContainsKey($_.scheduler) } |
    Sort-Object `
        @{ Expression = { [int]$_.seed } }, `
        @{ Expression = { $priority[$_.scheduler] } }

$todo = @()
foreach ($row in $rows) {
    if (!(Test-Path $row.checkpoint_path)) {
        $todo += $row
    }
}

$selected = $todo | Select-Object -First $Count
if ($selected.Count -eq 0) {
    Write-Host "No priority scheduler training runs need checkpoints."
    Write-Host "Next step is evaluation:"
    Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1 -Execute"
    return
}

foreach ($row in $selected) {
    $cmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $($row.run_id)"
    if ($DryRun) {
        $cmd += " -DryRun"
    }

    if (!$Execute) {
        Write-Host $cmd
        continue
    }

    Write-Host ("=" * 80)
    Write-Host "Executing: $cmd"
    if ($DryRun) {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $row.run_id -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $row.run_id
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Scheduler training failed: $($row.run_id)"
    }
}

if (!$Execute) {
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run the selected job(s). Use -Count N to select more than one."
}
