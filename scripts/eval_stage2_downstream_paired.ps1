param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$cmdArgs = @(
    "scripts\eval_stage2_downstream_shared.py",
    "--output-dir", "eval_results\stage2_downstream_paired_meta",
    "--extra-checkpoints",
    "metaspidermark_epoch116=local_checkpoints\verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_epoch116.pth",
    "metaspidermark_epoch110=local_checkpoints\verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_epoch110.pth",
    "--batch-size", "$BatchSize",
    "--testing-times", "$TestingTimes"
)
if ($DryRun) { $cmdArgs += "--dry-run" }
if ($Force) { $cmdArgs += "--force" }

& $PythonExe @cmdArgs
if ($LASTEXITCODE -ne 0) {
    throw "Paired downstream evaluation failed. Re-run the same command to resume."
}
