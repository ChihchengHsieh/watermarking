param(
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark",
    [int]$Steps = 50,
    [int]$Support = 8,
    [int]$Query = 8,
    [string]$EvalAttackSuite = "clean,jpeg_strong,msg_app_combo,occlusion"
)

$ErrorActionPreference = "Stop"

$manifestArgs = @{
    OutputDir = $OutputDir
    Schedulers = @("uniform", "ats", "bass", "bandit_ucb", "residual")
    Seeds = @(0)
    AnchorSchedulers = @("uniform")
    AnchorSeeds = @(0)
    Steps = $Steps
    Support = $Support
    Query = $Query
    AttackPool = "clean,downup50,crop,jpeg,blur,msg_app,occlusion"
    EvalAttackSuite = $EvalAttackSuite
}

& scripts\stage2_scheduler_manifest.ps1 @manifestArgs

Write-Host "Pilot scheduler benchmark:"
Write-Host "  output_dir=$OutputDir"
Write-Host "  steps=$Steps support=$Support query=$Query"
Write-Host "  eval_attack_suite=$EvalAttackSuite"
