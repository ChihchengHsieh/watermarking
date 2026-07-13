param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

Write-Host ("=" * 80)
Write-Host "Aggregating Stage 2 scheduler benchmark"
powershell -ExecutionPolicy Bypass -File scripts\stage2_aggregate_scheduler_benchmark.ps1 `
    -ManifestCsv $ManifestCsv `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Scheduler aggregation failed."
}

Write-Host ("=" * 80)
Write-Host "Generating scheduler paper tables"
powershell -ExecutionPolicy Bypass -File scripts\stage2_make_scheduler_tables.ps1 `
    -InputDir $OutputDir `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Scheduler table generation failed."
}

Write-Host ("=" * 80)
Write-Host "Current scheduler benchmark status"
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1 `
    -ManifestCsv $ManifestCsv
