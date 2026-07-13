param(
    [string]$ManifestCsv = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark",
    [switch]$Execute,
    [switch]$DryRun,
    [switch]$AllowConcurrent,
    [int]$BatchSize = 8,
    [int]$TestingTimes = 5,
    [int]$MaxCycles = 7
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $ManifestCsv)) {
    throw "Manifest CSV not found: $ManifestCsv. Generate it with scripts\stage2_meta_learning_manifest.ps1"
}

function Get-ActiveStage2Python {
    return @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match "scripts\\(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py|scripts/(run_stage2_scheduler_training|eval_stage2_scheduler_run)\.py"
            }
    )
}

function Get-RemainingRows {
    param([string]$CsvPath)
    $rows = @(Import-Csv $CsvPath)
    return @(
        $rows | Where-Object {
            !(Test-Path $_.checkpoint_path) -or !(Test-Path $_.eval_csv)
        }
    )
}

Write-Host "Run all SOTA/canonical meta-learning context rows"
Write-Host "Manifest: $ManifestCsv"
Write-Host "OutputDir: $OutputDir"
Write-Host "Execute=$Execute DryRun=$DryRun AllowConcurrent=$AllowConcurrent MaxCycles=$MaxCycles"
Write-Host ""

if (!$Execute) {
    Write-Host "Preview next row only:"
    powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 `
        -ManifestCsv $ManifestCsv `
        -BatchSize $BatchSize `
        -TestingTimes $TestingTimes
    if ($LASTEXITCODE -ne 0) {
        throw "Preview failed."
    }
    Write-Host ""
    Write-Host "Preview only. Add -Execute to run rows sequentially."
    return
}

if (!$AllowConcurrent) {
    $activeStage2Python = @(Get-ActiveStage2Python)
    if ($activeStage2Python.Count -gt 0) {
        Write-Host "Active Stage 2 Python process detected; not starting all-row SOTA/canonical meta-learning run."
        $activeStage2Python |
            Select-Object ProcessId, CreationDate, CommandLine |
            Format-List
        Write-Host "Re-run with -AllowConcurrent only if concurrent GPU use is intentional."
        return
    }
}

for ($cycle = 1; $cycle -le $MaxCycles; $cycle++) {
    Write-Host ("=" * 80)
    Write-Host "SOTA/canonical meta-learning cycle $cycle / $MaxCycles"

    $remaining = Get-RemainingRows $ManifestCsv

    if ($remaining.Count -eq 0) {
        Write-Host "All SOTA/canonical meta-learning rows have checkpoints and evaluation CSVs."
        break
    }

    $argsList = @(
        "-ExecutionPolicy", "Bypass",
        "-File", "scripts\stage2_run_next_sota_meta_learning_context.ps1",
        "-ManifestCsv", $ManifestCsv,
        "-BatchSize", "$BatchSize",
        "-TestingTimes", "$TestingTimes"
    )
    if ($DryRun) { $argsList += "-DryRun" }
    if ($AllowConcurrent) { $argsList += "-AllowConcurrent" }
    $argsList += "-Execute"

    powershell @argsList
    if ($LASTEXITCODE -ne 0) {
        throw "SOTA/canonical meta-learning cycle failed at cycle $cycle."
    }

    $remainingAfter = Get-RemainingRows $ManifestCsv
    if ($remainingAfter.Count -ge $remaining.Count -and !$DryRun) {
        throw "SOTA/canonical meta-learning cycle made no progress at cycle $cycle. Remaining rows before=$($remaining.Count), after=$($remainingAfter.Count)."
    }
}

Write-Host ("=" * 80)
Write-Host "Finalizing SOTA/canonical meta-learning outputs"
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1 `
    -ManifestCsv $ManifestCsv `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "Finalization failed."
}

Write-Host ("=" * 80)
Write-Host "Checking SOTA/canonical meta-learning completion gate"
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_completion_gate.ps1 `
    -ManifestCsv $ManifestCsv `
    -OutputDir $OutputDir
if ($LASTEXITCODE -ne 0) {
    throw "SOTA/canonical meta-learning completion gate failed."
}
