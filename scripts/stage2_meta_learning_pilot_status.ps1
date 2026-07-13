param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Pilot manifest not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_pilot_manifest.ps1"
}

powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1 -ManifestCsv $ManifestCsv

Write-Host ""
Write-Host "Next meta-learning pilot command:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute"
