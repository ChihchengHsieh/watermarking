param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Pilot manifest not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_pilot_manifest.ps1"
}

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

$priority = @{
    "uniform" = -1
    "ats" = 0
    "bass" = 1
    "bandit_ucb" = 2
    "residual" = 3
}

$rows = Import-Csv $ManifestCsv |
    Sort-Object @{ Expression = { if ($priority.ContainsKey($_.scheduler)) { $priority[$_.scheduler] } else { 999 } } }

$status = foreach ($row in $rows) {
    [pscustomobject]@{
        run_id = $row.run_id
        scheduler = $row.scheduler
        seed = [int]$row.seed
        steps = [int]$row.steps
        state = Get-RunState $row
        checkpoint = if (Test-Path $row.checkpoint_path) { "yes" } else { "no" }
        eval_csv = if (Test-Path $row.eval_csv) { "yes" } else { "no" }
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
Write-Host "Next pilot command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1 -Execute"
Write-Host ""
Write-Host "Preview next pilot action:"
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1 -ManifestCsv $ManifestCsv
