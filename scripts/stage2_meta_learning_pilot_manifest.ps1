param(
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark",
    [string[]]$MetaAlgorithms = @("fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge"),
    [string]$Scheduler = "ats",
    [int]$Steps = 50,
    [int]$Support = 8,
    [int]$Query = 8,
    [string]$AttackPool = "clean,downup50,crop,jpeg,blur,msg_app,occlusion",
    [string]$EvalAttackSuite = "clean,jpeg_strong,msg_app_combo,occlusion"
)

$ErrorActionPreference = "Stop"

& scripts\stage2_meta_learning_manifest.ps1 `
    -OutputDir $OutputDir `
    -MetaAlgorithms $MetaAlgorithms `
    -Seeds @(0) `
    -Steps $Steps `
    -Support $Support `
    -Query $Query `
    -Scheduler $Scheduler `
    -AttackPool $AttackPool `
    -EvalAttackSuite $EvalAttackSuite

Write-Host "Wrote Stage 2 meta-learning pilot manifest."
