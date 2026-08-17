param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [int]$Epochs = 87,
    [int]$BatchSize = 8,
    [int]$BootstrapSamples = 2000,
    [int]$EvaluationSeed = 20260815
)

$ErrorActionPreference = "Stop"
if (!(Test-Path $PythonExe)) { throw "Python executable not found: $PythonExe" }

$manifest = "papers\meta_learning\benchmark_outputs\stage2_controlled_six\controlled_six_runs.csv"
$downstreamName = "downstream_shared87_best_acc"
$evaluationDir = "eval_results\stage2_controlled_six_best_acc_coco_full_seed$EvaluationSeed"
$runIds = @(
    "controlled_uniform_seed19980802_steps2000_s8_q8",
    "controlled_bandit_ucb_seed19980802_steps2000_s8_q8",
    "controlled_ats_seed19980802_steps2000_s8_q8",
    "controlled_bass_seed19980802_steps2000_s8_q8",
    "controlled_asr_seed19980802_steps2000_s8_q8",
    "controlled_metaspidermark_original_seed19980802_steps2000_s8_q8"
)

Write-Host "[1/3] Replaying shared downstream training with Accuracy checkpointing" -ForegroundColor Cyan
$trainArgs = @(
    "scripts\train_stage2_downstream_shared.py",
    "--manifest-csv", $manifest,
    "--run-ids"
) + $runIds + @(
    "--epochs", "$Epochs",
    "--batch-size", "$BatchSize",
    "--output-name", $downstreamName
)
& $PythonExe @trainArgs
if ($LASTEXITCODE -ne 0) { throw "Accuracy-first downstream training failed." }

Write-Host "[2/3] Evaluating all six Accuracy-selected checkpoints on the full COCO set" -ForegroundColor Cyan
$evalArgs = @(
    "scripts\eval_stage2_downstream_shared.py",
    "--manifest-csv", $manifest,
    "--run-ids"
) + $runIds + @(
    "--downstream-dir", $downstreamName,
    "--checkpoint-name", "best_acc.pth",
    "--data-dir", "verifier_dataset_coco_octoweb",
    "--output-dir", $evaluationDir,
    "--batch-size", "$BatchSize",
    "--testing-times", "1",
    "--validation-split", "1.0",
    "--seed", "$EvaluationSeed"
)
& $PythonExe @evalArgs
if ($LASTEXITCODE -ne 0) { throw "Full-set paired evaluation failed." }

Write-Host "[3/3] Computing paired image-level bootstrap confidence intervals" -ForegroundColor Cyan
& $PythonExe "scripts\bootstrap_stage2_eval_ci.py" `
    --eval-dir $evaluationDir `
    --bootstrap-samples $BootstrapSamples `
    --seed $EvaluationSeed
if ($LASTEXITCODE -ne 0) { throw "Bootstrap CI calculation failed." }

Write-Host "[DONE] Accuracy-first results: $evaluationDir\bootstrap_ci_summary.csv" -ForegroundColor Green
