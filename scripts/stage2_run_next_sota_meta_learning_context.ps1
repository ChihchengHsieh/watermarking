param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowConcurrent,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5
)

$ErrorActionPreference = "Stop"

$argsList = @(
    "-ExecutionPolicy", "Bypass",
    "-File", "scripts\stage2_run_next_meta_learning_cycle.ps1",
    "-ManifestCsv", $ManifestCsv,
    "-BatchSize", $BatchSize,
    "-TestingTimes", $TestingTimes
)

if ($Execute) { $argsList += "-Execute" }
if ($DryRun) { $argsList += "-DryRun" }
if ($AllowConcurrent) { $argsList += "-AllowConcurrent" }

powershell @argsList
if ($LASTEXITCODE -ne 0) {
    throw "SOTA meta-learning context runner failed."
}
