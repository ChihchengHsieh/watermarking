param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_smoke",
    [string[]]$MetaAlgorithms = @("fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge"),
    [string]$Scheduler = "ats",
    [switch]$RunTraining,
    [switch]$AllowConcurrent
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$active = Get-CimInstance Win32_Process |
    Where-Object {
        $_.CommandLine -like "*run_stage2_scheduler_training.py*" -or
        $_.CommandLine -like "*eval_stage2_scheduler_run.py*"
    }

if ($RunTraining -and !$AllowConcurrent -and @($active).Count -gt 0) {
    Write-Host "Active Stage 2 Python process detected; not starting training smoke."
    $active | Select-Object ProcessId, CreationDate, CommandLine | Format-List
    Write-Host "Re-run with -AllowConcurrent only if you intentionally want concurrent GPU jobs."
    exit 2
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

Write-Host "Stage 2 meta-learning algorithm smoke"
Write-Host "OutputDir=$OutputDir"
Write-Host "Scheduler=$Scheduler"
Write-Host "RunTraining=$RunTraining"

foreach ($algorithm in $MetaAlgorithms) {
    $runId = "smoke_${algorithm}_${Scheduler}_seed0_steps1"
    $runDir = Join-Path $OutputDir $runId

    $cmdArgs = @(
        "scripts\run_stage2_scheduler_training.py",
        "--meta-algorithm", $algorithm,
        "--scheduler", $Scheduler,
        "--seed", "0",
        "--steps", "1",
        "--n-support", "1",
        "--n-query", "1",
        "--meta-batch-size", "1",
        "--tasks-per-epoch", "2",
        "--log-interval", "1",
        "--save-interval", "1",
        "--attack-pool", "clean,jpeg",
        "--run-dir", $runDir
    )

    if (!$RunTraining) {
        $cmdArgs += "--dry-run"
    }

    Write-Host ("=" * 80)
    Write-Host "Smoke: meta_algorithm=$algorithm scheduler=$Scheduler"
    & $PythonExe @cmdArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Meta-learning smoke failed: $algorithm"
    }
}

Write-Host "Stage 2 meta-learning smoke passed."
