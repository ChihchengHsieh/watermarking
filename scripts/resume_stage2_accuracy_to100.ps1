param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [int]$BatchSize = 8
)

$ErrorActionPreference = "Stop"
if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repoRoot

$manifest = "papers\meta_learning\benchmark_outputs\stage2_controlled_six\controlled_six_runs.csv"
$runIds = @(
    "controlled_uniform_seed19980802_steps2000_s8_q8",
    "controlled_bandit_ucb_seed19980802_steps2000_s8_q8",
    "controlled_ats_seed19980802_steps2000_s8_q8",
    "controlled_bass_seed19980802_steps2000_s8_q8",
    "controlled_asr_seed19980802_steps2000_s8_q8",
    "controlled_metaspidermark_original_seed19980802_steps2000_s8_q8"
)

$trainArgs = @(
    "scripts\train_stage2_downstream_shared.py",
    "--manifest-csv", $manifest,
    "--run-ids"
) + $runIds + @(
    "--epochs", "100",
    "--batch-size", "$BatchSize",
    "--output-name", "downstream_shared87_best_acc",
    "--snapshot-epochs", "100"
)

Write-Host "[RESUME] Extending all six methods from epoch 87 to epoch 100" -ForegroundColor Cyan
& $PythonExe @trainArgs
if ($LASTEXITCODE -ne 0) {
    throw "Downstream training extension failed with exit code $LASTEXITCODE."
}

Write-Host "[DONE] All six methods reached epoch 100." -ForegroundColor Green
Write-Host "Ask Codex to compare the new best_acc.pth files with the saved epoch-87 snapshot before evaluation."
