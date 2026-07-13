param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
}

$requiredAlgorithms = @("fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge")
$rows = @(Import-Csv $ManifestCsv)
$failures = @()
$formalRunIds = @{}

foreach ($algorithm in $requiredAlgorithms) {
    $matches = @($rows | Where-Object { $_.meta_algorithm -eq $algorithm -and $_.scheduler -eq "ats" -and [int]$_.seed -eq 0 })
    if ($matches.Count -ne 1) {
        $failures += "Expected exactly one ats/seed0 row for ${algorithm}; found $($matches.Count)."
        continue
    }

    $row = $matches[0]
    $formalRunIds[$row.run_id] = $true
    if (!(Test-Path $row.checkpoint_path)) {
        $failures += "Missing checkpoint for $($row.run_id): $($row.checkpoint_path)"
    }
    if (!(Test-Path $row.eval_csv)) {
        $failures += "Missing evaluation CSV for $($row.run_id): $($row.eval_csv)"
    }
}

$normalizedCsv = Join-Path $OutputDir "normalized_meta_learning_results.csv"
$summaryBySeedCsv = Join-Path $OutputDir "summary_by_meta_algorithm_seed.csv"
$summaryCsv = Join-Path $OutputDir "summary_by_meta_algorithm.csv"
$summaryTex = Join-Path $OutputDir "paper_table_meta_algorithm_summary.tex"

if (!(Test-Path $normalizedCsv)) {
    $failures += "Missing normalized CSV: $normalizedCsv"
} else {
    $normalizedRows = @(Import-Csv $normalizedCsv)
    foreach ($row in $normalizedRows) {
        if (!$formalRunIds.ContainsKey($row.run_id)) {
            $failures += "Normalized CSV contains non-formal run_id $($row.run_id)."
        }
        if ($row.scheduler -ne "ats" -or [int]$row.seed -ne 0) {
            $failures += "Normalized CSV contains non-fixed scheduler/seed row: run_id=$($row.run_id), scheduler=$($row.scheduler), seed=$($row.seed)."
        }
    }
}

if (!(Test-Path $summaryBySeedCsv)) {
    $failures += "Missing per-seed summary CSV: $summaryBySeedCsv"
} else {
    $summaryBySeedRows = @(Import-Csv $summaryBySeedCsv)
    foreach ($row in $summaryBySeedRows) {
        if ($row.scheduler -ne "ats" -or [int]$row.seed -ne 0) {
            $failures += "Per-seed summary contains non-fixed scheduler/seed row: algorithm=$($row.meta_algorithm), scheduler=$($row.scheduler), seed=$($row.seed)."
        }
    }
}

if (!(Test-Path $summaryCsv)) {
    $failures += "Missing summary CSV: $summaryCsv"
} else {
    $summaryRows = @(Import-Csv $summaryCsv)
    foreach ($algorithm in $requiredAlgorithms) {
        if (@($summaryRows | Where-Object { $_.meta_algorithm -eq $algorithm }).Count -lt 1) {
            $failures += "Summary CSV has no row for $algorithm."
        }
    }
}

if (!(Test-Path $summaryTex)) {
    $failures += "Missing paper summary table: $summaryTex"
} else {
    $summaryText = Get-Content $summaryTex -Raw
    if ($summaryText -match "Optional") {
        $failures += "Paper summary table still uses Optional wording: $summaryTex"
    }
    if ($summaryText -notmatch "SOTA/canonical") {
        $failures += "Paper summary table does not identify the context as SOTA/canonical: $summaryTex"
    }
    if ($summaryText -match "Pending") {
        $failures += "Paper summary table is still a pending placeholder: $summaryTex"
    }
}

if ($failures.Count -gt 0) {
    Write-Host "SOTA/canonical meta-learning completion gate: FAILED"
    $failures | ForEach-Object { Write-Host "- $_" }
    throw "SOTA/canonical meta-learning context is incomplete."
}

Write-Host "SOTA/canonical meta-learning completion gate: PASSED"
Write-Host "All required fixed-scheduler SOTA/canonical meta-learning rows have checkpoints, evaluation CSVs, and table outputs."
