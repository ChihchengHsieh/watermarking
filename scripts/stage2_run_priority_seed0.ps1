param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string[]]$RunIds = @(
        "scheduler_ats_seed0_steps2000",
        "scheduler_bass_seed0_steps2000",
        "scheduler_bandit_ucb_seed0_steps2000",
        "scheduler_residual_seed0_steps2000"
    ),
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

$manifest = Import-Csv $ManifestCsv
$missing = @()
foreach ($runId in $RunIds) {
    if (!(($manifest | Where-Object { $_.run_id -eq $runId } | Select-Object -First 1))) {
        $missing += $runId
    }
}
if ($missing.Count -gt 0) {
    throw "Run ID(s) not found in manifest: $($missing -join ', ')"
}

foreach ($runId in $RunIds) {
    $row = $manifest | Where-Object { $_.run_id -eq $runId } | Select-Object -First 1
    $hasFinalCheckpoint = Test-Path $row.checkpoint_path
    if ($hasFinalCheckpoint -and !$Force) {
        Write-Host "Skipping $runId because final checkpoint already exists: $($row.checkpoint_path)"
        continue
    }

    $cmd = "powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $runId"
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
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $runId -DryRun
    } else {
        powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId $runId
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Priority seed-0 run failed: $runId"
    }
}

if (!$Execute) {
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run these jobs. Add -Force to include runs with existing final checkpoints."
}
