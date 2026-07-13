param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv",
    [string]$SchedulerOutputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv",
    [string]$MetaOutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowConcurrent,
    [int]$SaveInterval = 10,
    [int]$LogInterval = 10,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 3
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $SchedulerManifest)) {
    throw "Missing scheduler pilot manifest: $SchedulerManifest. Generate it with scripts\stage2_scheduler_pilot_manifest.ps1"
}
if (!(Test-Path $MetaManifest)) {
    throw "Missing meta-learning pilot manifest: $MetaManifest. Generate it with scripts\stage2_meta_learning_pilot_manifest.ps1"
}

if (!$AllowConcurrent) {
    $activeStage2Python = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
            }
    )
    if ($activeStage2Python.Count -gt 0) {
        Write-Host "Active Stage 2 Python process detected; not starting pilot goal job."
        $activeStage2Python |
            Select-Object ProcessId, CreationDate, CommandLine |
            Format-List
        Write-Host "Re-run with -AllowConcurrent only if you intentionally want concurrent GPU jobs."
        return
    }
}

function Test-Complete {
    param([string]$ManifestCsv)
    $rows = Import-Csv $ManifestCsv
    foreach ($row in $rows) {
        if (!(Test-Path $row.eval_csv)) {
            return $false
        }
    }
    return $true
}

$schedulerComplete = Test-Complete $SchedulerManifest
$metaComplete = Test-Complete $MetaManifest

if (!$metaComplete) {
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_next_meta_learning_pilot.ps1",
        "-ManifestCsv", $MetaManifest,
        "-OutputDir", $MetaOutputDir,
        "-SaveInterval", "$SaveInterval",
        "-LogInterval", "$LogInterval",
        "-BatchSize", "$BatchSize",
        "-TestingTimes", "$TestingTimes"
    )
    if ($Execute) { $args += "-Execute" }
    if ($DryRun) { $args += "-DryRun" }
    if ($AllowConcurrent) { $args += "-AllowConcurrent" }
    Write-Host "Pilot goal phase: SOTA/canonical meta-learning algorithm pilot"
    powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Meta-learning pilot goal step failed."
    }
    return
}

if (!$schedulerComplete) {
    $args = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_next_scheduler_pilot.ps1",
        "-ManifestCsv", $SchedulerManifest,
        "-OutputDir", $SchedulerOutputDir,
        "-SaveInterval", "$SaveInterval",
        "-LogInterval", "$LogInterval",
        "-BatchSize", "$BatchSize",
        "-TestingTimes", "$TestingTimes"
    )
    if ($Execute) { $args += "-Execute" }
    if ($DryRun) { $args += "-DryRun" }
    if ($AllowConcurrent) { $args += "-AllowConcurrent" }
    Write-Host "Pilot goal phase: scheduler ablation pilot"
    powershell @args
    if ($LASTEXITCODE -ne 0) {
        throw "Scheduler pilot goal step failed."
    }
    return
}

Write-Host "Pilot goal complete: scheduler and meta-learning pilot manifests have evaluation CSVs."
Write-Host "Finalize scheduler pilot:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1 -ManifestCsv $SchedulerManifest -OutputDir $SchedulerOutputDir"
Write-Host "Finalize meta-learning pilot:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1 -ManifestCsv $MetaManifest -OutputDir $MetaOutputDir"
