param(
    [Parameter(Mandatory = $true)]
    [string]$RunId,
    [string[]]$SearchDirs = @(
        "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark",
        "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark",
        "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark",
        "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark"
    ),
    [int]$PollSeconds = 60,
    [int]$TimeoutMinutes = 360
)

$ErrorActionPreference = "Stop"

$checkpointPaths = foreach ($root in $SearchDirs) {
    Join-Path $root "$RunId/checkpoints/latest.pth"
    Join-Path $root "$RunId/checkpoints/final.pth"
}

function Get-ManifestForCheckpoint {
    param([string]$CheckpointPath)
    $normalized = $CheckpointPath -replace "/", "\"
    if ($normalized -match "stage2_scheduler_benchmark") {
        return "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv"
    }
    if ($normalized -match "stage2_scheduler_pilot_benchmark") {
        return "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv"
    }
    if ($normalized -match "stage2_meta_learning_benchmark") {
        return "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv"
    }
    if ($normalized -match "stage2_meta_learning_pilot_benchmark") {
        return "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv"
    }
    return ""
}

$deadline = (Get-Date).AddMinutes($TimeoutMinutes)
Write-Host "Waiting for checkpoint for RunId=$RunId"
Write-Host "TimeoutMinutes=$TimeoutMinutes PollSeconds=$PollSeconds"
$checkpointPaths | ForEach-Object { Write-Host "  $_" }

while ((Get-Date) -lt $deadline) {
    $existing = @($checkpointPaths | Where-Object { Test-Path $_ })
    if ($existing.Count -gt 0) {
        Write-Host ""
        Write-Host "Checkpoint found:"
        $existing | ForEach-Object {
            Get-Item $_ | Select-Object FullName, Length, LastWriteTime
        }
        $latest = @($existing | Where-Object { Split-Path -Leaf $_ -eq "latest.pth" } | Select-Object -First 1)
        if ($latest.Count -gt 0) {
            $manifestCsv = Get-ManifestForCheckpoint $latest[0]
            Write-Host ""
            Write-Host "Quick-look eval command:"
            if ($manifestCsv) {
                Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -ManifestCsv $manifestCsv -RunId $RunId"
            } else {
                Write-Host "powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -RunId $RunId"
            }
        }
        exit 0
    }

    $active = @(
        Get-CimInstance Win32_Process |
            Where-Object {
                $_.Name -eq "python.exe" -and
                $_.CommandLine -match [regex]::Escape($RunId)
            }
    )
    if ($active.Count -eq 0) {
        Write-Host ""
        Write-Host "No active Python process found for RunId=$RunId and no checkpoint exists."
        exit 2
    }

    Write-Host ("[{0}] still waiting..." -f (Get-Date).ToString("HH:mm:ss"))
    Start-Sleep -Seconds $PollSeconds
}

Write-Host "Timed out waiting for checkpoint for RunId=$RunId"
exit 124
