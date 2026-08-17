param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [int]$Epochs = 120,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5,
    [switch]$DryRun,
    [switch]$ForceEvaluation
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }

$root = "papers\meta_learning\benchmark_outputs\stage2_controlled_six"
$manifest = Join-Path $root "controlled_six_runs.csv"
$runIds = @(
    "controlled_uniform_seed19980802_steps2000_s8_q8",
    "controlled_bandit_ucb_seed19980802_steps2000_s8_q8",
    "controlled_ats_seed19980802_steps2000_s8_q8",
    "controlled_bass_seed19980802_steps2000_s8_q8",
    "controlled_asr_seed19980802_steps2000_s8_q8",
    "controlled_metaspidermark_original_seed19980802_steps2000_s8_q8"
)

if ($DryRun) {
    & $PythonExe scripts\train_stage2_meta_shared_six.py --baseline-only --dry-run
    if ($LASTEXITCODE -ne 0) { throw "Controlled six-method dry run failed." }
    Write-Host "[DRY-RUN] Meta manifest validated. Downstream/evaluation require completed meta checkpoints."
    exit 0
}

& $PythonExe scripts\train_stage2_meta_shared_six.py --baseline-only
if ($LASTEXITCODE -ne 0) { throw "Shared meta-training failed. Re-run this command to resume." }

$downstreamArgs = @(
    "scripts\train_stage2_downstream_shared.py",
    "--manifest-csv", $manifest,
    "--run-ids"
) + $runIds + @(
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--output-name", "downstream_shared120"
)
& $PythonExe @downstreamArgs
if ($LASTEXITCODE -ne 0) { throw "Shared downstream training failed. Re-run this command to resume." }

$evalArgs = @(
    "scripts\eval_stage2_downstream_shared.py",
    "--manifest-csv", $manifest,
    "--run-ids"
) + $runIds + @(
    "--downstream-dir", "downstream_shared120",
    "--output-dir", "eval_results\stage2_controlled_six_paired",
    "--batch-size", "$BatchSize",
    "--testing-times", "$TestingTimes"
)
if ($ForceEvaluation) { $evalArgs += "--force" }
& $PythonExe @evalArgs
if ($LASTEXITCODE -ne 0) { throw "Paired evaluation failed. Re-run this command to resume." }

Write-Host "[DONE] controlled six-method meta-training, downstream training, and evaluation complete."
