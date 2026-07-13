param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$InputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark"
)

$ErrorActionPreference = "Stop"

& $PythonExe scripts\stage2_make_scheduler_tables.py --input-dir $InputDir --output-dir $OutputDir
