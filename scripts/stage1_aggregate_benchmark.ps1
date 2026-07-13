param(
    [string]$NonMetaCsv = "eval_results/our_improved_final/attack_eval_summary.csv",
    [string]$MetaSweepCsv = "eval_results/downstream_meta_checkpoint_sweep/combined_attack_eval_summary.csv",
    [string]$SelectedMetaCheckpoint = "epoch116",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage1_current",
    [string]$BaselineMethod = "SpiderMark-no-meta"
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

function Convert-NonMetaRows {
    param([array]$Rows)
    foreach ($row in $Rows) {
        [pscustomobject]@{
            method          = "SpiderMark-no-meta"
            family          = "SpiderMark verifier"
            meta_learning   = "no"
            scheduler       = "none"
            checkpoint_rule = "existing_final"
            checkpoint_label = "nonmeta_final"
            attack          = $row.attack
            accuracy        = [double]$row.our_acc
            auroc           = [double]$row.our_auc
            source_csv      = $NonMetaCsv
        }
    }
}

function Convert-MetaRows {
    param([array]$Rows)
    foreach ($row in $Rows) {
        $label = $row.checkpoint_label
        [pscustomobject]@{
            method          = "MetaSpiderMark-$label"
            family          = "MetaSpiderMark verifier"
            meta_learning   = "yes"
            scheduler       = "llm_residual"
            checkpoint_rule = $label
            checkpoint_label = $label
            attack          = $row.attack
            accuracy        = [double]$row.our_acc
            auroc           = [double]$row.our_auc
            source_csv      = $MetaSweepCsv
        }
    }
}

function Get-AttackIndex {
    param([string]$Attack)
    $idx = [array]::IndexOf($AttackOrder, $Attack)
    if ($idx -lt 0) { return $AttackOrder.Count }
    return $idx
}

if (!(Test-Path $NonMetaCsv)) {
    throw "Missing non-meta CSV: $NonMetaCsv"
}
if (!(Test-Path $MetaSweepCsv)) {
    throw "Missing meta sweep CSV: $MetaSweepCsv"
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$nonMeta = Import-Csv $NonMetaCsv
$meta = Import-Csv $MetaSweepCsv
$normalized = @()
$normalized += Convert-NonMetaRows $nonMeta
$normalized += Convert-MetaRows $meta
$normalized = $normalized | Sort-Object method, @{ Expression = { Get-AttackIndex $_.attack } }, attack

$normalizedPath = Join-Path $OutputDir "normalized_results.csv"
$normalized | Export-Csv $normalizedPath -NoTypeInformation

$summary = $normalized |
    Group-Object method |
    ForEach-Object {
        $rows = $_.Group
        [pscustomobject]@{
            method          = $_.Name
            family          = $rows[0].family
            meta_learning   = $rows[0].meta_learning
            scheduler       = $rows[0].scheduler
            checkpoint_rule = $rows[0].checkpoint_rule
            num_attacks     = ($rows | Select-Object -ExpandProperty attack -Unique).Count
            mean_accuracy   = (($rows | Measure-Object accuracy -Average).Average)
            mean_auroc      = (($rows | Measure-Object auroc -Average).Average)
        }
    } |
    Sort-Object method

$summaryPath = Join-Path $OutputDir "summary_by_method.csv"
$summary | Export-Csv $summaryPath -NoTypeInformation

$baselineRows = @{}
$normalized |
    Where-Object { $_.method -eq $BaselineMethod } |
    ForEach-Object { $baselineRows[$_.attack] = $_ }

if ($baselineRows.Count -eq 0) {
    throw "Baseline method not found: $BaselineMethod"
}

$deltas = $normalized |
    Where-Object { $_.method -ne $BaselineMethod -and $baselineRows.ContainsKey($_.attack) } |
    ForEach-Object {
        $base = $baselineRows[$_.attack]
        [pscustomobject]@{
            method            = $_.method
            baseline_method   = $BaselineMethod
            attack            = $_.attack
            accuracy          = $_.accuracy
            baseline_accuracy = $base.accuracy
            delta_accuracy    = $_.accuracy - $base.accuracy
            auroc             = $_.auroc
            baseline_auroc    = $base.auroc
            delta_auroc       = $_.auroc - $base.auroc
        }
    } |
    Sort-Object method, @{ Expression = { Get-AttackIndex $_.attack } }, attack

$deltaPath = Join-Path $OutputDir "delta_vs_baseline.csv"
$deltas | Export-Csv $deltaPath -NoTypeInformation

$selectedMethod = "MetaSpiderMark-$SelectedMetaCheckpoint"
$tableRows = @()
foreach ($attack in $AttackOrder) {
    $base = $normalized | Where-Object { $_.method -eq $BaselineMethod -and $_.attack -eq $attack } | Select-Object -First 1
    $metaRow = $normalized | Where-Object { $_.method -eq $selectedMethod -and $_.attack -eq $attack } | Select-Object -First 1
    if ($null -eq $base -or $null -eq $metaRow) {
        throw "Missing selected comparison row for attack=$attack"
    }
    $tableRows += [pscustomobject]@{
        attack = $attack
        base_acc = $base.accuracy
        base_auc = $base.auroc
        meta_acc = $metaRow.accuracy
        meta_auc = $metaRow.auroc
        delta_acc = $metaRow.accuracy - $base.accuracy
        delta_auc = $metaRow.auroc - $base.auroc
    }
}

$latex = New-Object System.Collections.Generic.List[string]
$latex.Add("% Auto-generated by scripts/stage1_aggregate_benchmark.ps1")
$latex.Add("\begin{table*}[t]")
$latex.Add("\centering")
$latex.Add("\caption{Stage 1 comparison between non-meta SpiderMark and the selected MetaSpiderMark verifier.}")
$latex.Add("\label{tab:stage1_generated_meta_vs_nometa}")
$latex.Add("\resizebox{\textwidth}{!}{")
$latex.Add("\begin{tabular}{lrrrrrr}")
$latex.Add("\toprule")
$latex.Add("\textbf{Attack} & \textbf{No-meta Acc} & \textbf{No-meta AUROC} & \textbf{Meta Acc} & \textbf{Meta AUROC} & \textbf{$\Delta$ Acc} & \textbf{$\Delta$ AUROC} \\")
$latex.Add("\midrule")
foreach ($row in $tableRows) {
    $attack = $row.attack.Replace("_", "\_")
    $latex.Add(("{0} & {1:F4} & {2:F4} & {3:F4} & {4:F4} & {5:+0.0000;-0.0000} & {6:+0.0000;-0.0000} \\" -f $attack, $row.base_acc, $row.base_auc, $row.meta_acc, $row.meta_auc, $row.delta_acc, $row.delta_auc))
}
$meanBaseAcc = ($tableRows | Measure-Object base_acc -Average).Average
$meanBaseAuc = ($tableRows | Measure-Object base_auc -Average).Average
$meanMetaAcc = ($tableRows | Measure-Object meta_acc -Average).Average
$meanMetaAuc = ($tableRows | Measure-Object meta_auc -Average).Average
$meanDeltaAcc = ($tableRows | Measure-Object delta_acc -Average).Average
$meanDeltaAuc = ($tableRows | Measure-Object delta_auc -Average).Average
$meanBaseAccText = "{0:F4}" -f $meanBaseAcc
$meanBaseAucText = "{0:F4}" -f $meanBaseAuc
$meanMetaAccText = "{0:F4}" -f $meanMetaAcc
$meanMetaAucText = "{0:F4}" -f $meanMetaAuc
$meanDeltaAccText = "{0:+0.0000;-0.0000}" -f $meanDeltaAcc
$meanDeltaAucText = "{0:+0.0000;-0.0000}" -f $meanDeltaAuc
$latex.Add("\midrule")
$latex.Add("\textbf{Mean} & \textbf{$meanBaseAccText} & \textbf{$meanBaseAucText} & \textbf{$meanMetaAccText} & \textbf{$meanMetaAucText} & \textbf{$meanDeltaAccText} & \textbf{$meanDeltaAucText} \\")
$latex.Add("\bottomrule")
$latex.Add("\end{tabular}")
$latex.Add("}")
$latex.Add("\end{table*}")

$latexPath = Join-Path $OutputDir "paper_table_meta_vs_nometa.tex"
$latex | Set-Content -Path $latexPath -Encoding UTF8

Write-Host "Wrote normalized results to $normalizedPath"
Write-Host "Wrote method summary to $summaryPath"
Write-Host "Wrote deltas to $deltaPath"
Write-Host "Wrote LaTeX table to $latexPath"
