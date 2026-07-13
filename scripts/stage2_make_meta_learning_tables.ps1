param(
    [string]$PythonExe = "C:\Users\chihc\miniconda3\envs\pytorch\python.exe",
    [string]$InputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark",
    [string]$OutputDir = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark"
)

$ErrorActionPreference = "Stop"

& $PythonExe scripts\stage2_make_meta_learning_tables.py --input-dir $InputDir --output-dir $OutputDir
