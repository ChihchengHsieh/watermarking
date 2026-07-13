param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
}

Write-Host ("=" * 80)
Write-Host "Aggregating SOTA Stage 2 meta-learning context benchmark"
powershell -ExecutionPolicy Bypass -File scripts\stage2_aggregate_meta_learning_benchmark.ps1 `
    -ManifestCsv $ManifestCsv `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Meta-learning aggregation failed."
}

Write-Host ("=" * 80)
Write-Host "Generating SOTA/canonical meta-learning context paper tables"
powershell -ExecutionPolicy Bypass -File scripts\stage2_make_meta_learning_tables.ps1 `
    -InputDir $OutputDir `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Meta-learning table generation failed."
}

Write-Host ("=" * 80)
Write-Host "Current SOTA/canonical meta-learning context benchmark status"
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1 `
    -ManifestCsv $ManifestCsv
