param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark",
    [string]$BaselineScheduler = "uniform"
)

$ErrorActionPreference = "Stop"

$AttackOrder = @(
    "clean",
    "jpeg_strong",
    "msg_app_combo",
    "down_up",
    "blur",
    "random_crop",
    "occlusion",
    "geom_warp",
    "train_aug_mix"
)

function Get-AttackIndex {
    param([string]$Attack)
    $idx = [array]::IndexOf($AttackOrder, $Attack)
    if ($idx -lt 0) { return $AttackOrder.Count }
    return $idx
}

function Get-FirstValue {
    param($Row, [string[]]$Names)
    foreach ($name in $Names) {
        if ($Row.PSObject.Properties.Name -contains $name) {
            $value = $Row.$name
            if ($null -ne $value -and "$value" -ne "") {
                return [double]$value
            }
        }
    }
    throw "Could not find any populated metric column: $($Names -join ', ')"
}

function Export-CsvWithSchema {
    param(
        $Rows,
        [string]$Path,
        [string[]]$Columns
    )
    if ($null -ne $Rows -and @($Rows).Count -gt 0) {
        $Rows | Export-Csv $Path -NoTypeInformation
    } else {
        ($Columns -join ",") | Set-Content -Path $Path -Encoding UTF8
    }
}

if (!(Test-Path $ManifestCsv)) {
    throw "Missing manifest CSV: $ManifestCsv. Generate it with scripts\stage2_scheduler_manifest.ps1"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$manifest = Import-Csv $ManifestCsv
$normalized = @()
$missing = @()

foreach ($run in $manifest) {
    if (!(Test-Path $run.eval_csv)) {
        $missing += $run
        continue
    }

    $evalRows = Import-Csv $run.eval_csv
    foreach ($row in $evalRows) {
        $normalized += [pscustomobject]@{
            run_id          = $run.run_id
            scheduler       = $run.scheduler
            seed            = [int]$run.seed
            steps           = [int]$run.steps
            n_support       = [int]$run.n_support
            n_query         = [int]$run.n_query
            checkpoint_path = $run.checkpoint_path
            attack          = $row.attack
            accuracy        = Get-FirstValue $row @("our_acc", "accuracy", "acc")
            auroc           = Get-FirstValue $row @("our_auc", "our_auroc", "auroc", "auc")
            source_csv      = $run.eval_csv
        }
    }
}

$normalized = $normalized | Sort-Object scheduler, seed, @{ Expression = { Get-AttackIndex $_.attack } }, attack
$normalizedPath = Join-Path $OutputDir "normalized_scheduler_results.csv"
Export-CsvWithSchema $normalized $normalizedPath @(
    "run_id", "scheduler", "seed", "steps", "n_support", "n_query",
    "checkpoint_path", "attack", "accuracy", "auroc", "source_csv"
)

$summary = $normalized |
    Group-Object scheduler, seed |
    ForEach-Object {
        $rows = $_.Group
        [pscustomobject]@{
            scheduler     = $rows[0].scheduler
            seed          = $rows[0].seed
            steps         = $rows[0].steps
            n_support     = $rows[0].n_support
            n_query       = $rows[0].n_query
            num_attacks   = ($rows | Select-Object -ExpandProperty attack -Unique).Count
            mean_accuracy = (($rows | Measure-Object accuracy -Average).Average)
            mean_auroc    = (($rows | Measure-Object auroc -Average).Average)
            worst_accuracy = (($rows | Measure-Object accuracy -Minimum).Minimum)
            worst_auroc    = (($rows | Measure-Object auroc -Minimum).Minimum)
        }
    } |
    Sort-Object scheduler, seed

$summaryPath = Join-Path $OutputDir "summary_by_scheduler_seed.csv"
Export-CsvWithSchema $summary $summaryPath @(
    "scheduler", "seed", "steps", "n_support", "n_query", "num_attacks",
    "mean_accuracy", "mean_auroc", "worst_accuracy", "worst_auroc"
)

$schedulerSummary = $summary |
    Group-Object scheduler |
    ForEach-Object {
        $rows = $_.Group
        [pscustomobject]@{
            scheduler           = $_.Name
            num_seeds           = $rows.Count
            mean_accuracy       = (($rows | Measure-Object mean_accuracy -Average).Average)
            mean_auroc          = (($rows | Measure-Object mean_auroc -Average).Average)
            mean_worst_accuracy = (($rows | Measure-Object worst_accuracy -Average).Average)
            mean_worst_auroc    = (($rows | Measure-Object worst_auroc -Average).Average)
        }
    } |
    Sort-Object mean_accuracy -Descending

$schedulerSummaryPath = Join-Path $OutputDir "summary_by_scheduler.csv"
Export-CsvWithSchema $schedulerSummary $schedulerSummaryPath @(
    "scheduler", "num_seeds", "mean_accuracy", "mean_auroc",
    "mean_worst_accuracy", "mean_worst_auroc"
)

$baselineBySeedAttack = @{}
$normalized |
    Where-Object { $_.scheduler -eq $BaselineScheduler } |
    ForEach-Object {
        $baselineBySeedAttack["$($_.seed)|$($_.attack)"] = $_
    }

$deltas = @()
if ($baselineBySeedAttack.Count -gt 0) {
    $deltas = $normalized |
        Where-Object { $_.scheduler -ne $BaselineScheduler } |
        ForEach-Object {
            $key = "$($_.seed)|$($_.attack)"
            if (!$baselineBySeedAttack.ContainsKey($key)) {
                return
            }
            $base = $baselineBySeedAttack[$key]
            [pscustomobject]@{
                scheduler         = $_.scheduler
                baseline_scheduler = $BaselineScheduler
                seed              = $_.seed
                attack            = $_.attack
                accuracy          = $_.accuracy
                baseline_accuracy = $base.accuracy
                delta_accuracy    = $_.accuracy - $base.accuracy
                auroc             = $_.auroc
                baseline_auroc    = $base.auroc
                delta_auroc       = $_.auroc - $base.auroc
            }
        } |
        Sort-Object scheduler, seed, @{ Expression = { Get-AttackIndex $_.attack } }, attack
}

$deltaPath = Join-Path $OutputDir "delta_vs_uniform.csv"
Export-CsvWithSchema $deltas $deltaPath @(
    "scheduler", "baseline_scheduler", "seed", "attack", "accuracy",
    "baseline_accuracy", "delta_accuracy", "auroc", "baseline_auroc",
    "delta_auroc"
)

$missingPath = Join-Path $OutputDir "missing_runs.csv"
$missing | Export-Csv $missingPath -NoTypeInformation

Write-Host "Wrote normalized scheduler results to $normalizedPath"
Write-Host "Wrote per-seed summary to $summaryPath"
Write-Host "Wrote scheduler summary to $schedulerSummaryPath"
Write-Host "Wrote deltas to $deltaPath"
Write-Host "Missing/unrun rows: $($missing.Count). Details in $missingPath"
