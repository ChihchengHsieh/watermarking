param(
    [string]$SchedulerManifest = "papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    [string]$MetaManifest = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv"
)

$ErrorActionPreference = "Stop"

function Assert-Condition {
    param(
        [bool]$Condition,
        [string]$Message
    )
    if ($Condition) {
        [pscustomobject]@{ status = "ok"; check = $Message }
    } else {
        [pscustomobject]@{ status = "fail"; check = $Message }
    }
}

$checks = @()

if (!(Test-Path $SchedulerManifest)) {
    throw "Missing scheduler manifest: $SchedulerManifest"
}
$schedulerRows = Import-Csv $SchedulerManifest
$schedulerNames = @($schedulerRows | Select-Object -ExpandProperty scheduler -Unique)
$runIds = @($schedulerRows | Select-Object -ExpandProperty run_id)

$checks += Assert-Condition ($schedulerRows.Count -eq 13) "scheduler manifest has 13 rows: uniform seed0 plus four adaptive schedulers over seeds 0/1/2"
$checks += Assert-Condition (@($schedulerRows | Where-Object { $_.scheduler -eq "uniform" }).Count -eq 1) "uniform appears only once"
$checks += Assert-Condition (@($schedulerRows | Where-Object { $_.scheduler -eq "uniform" -and [int]$_.seed -eq 0 }).Count -eq 1) "uniform is seed 0 only"
$checks += Assert-Condition (-not ($schedulerNames -contains "cycle")) "cycle is not in the main scheduler manifest"
$checks += Assert-Condition (-not ($schedulerNames -contains "hard_task")) "hard_task is not in the main scheduler manifest"
$checks += Assert-Condition (-not ($schedulerNames -contains "progress")) "progress is not in the main scheduler manifest"
foreach ($scheduler in @("ats", "bass", "bandit_ucb", "residual")) {
    $checks += Assert-Condition (@($schedulerRows | Where-Object { $_.scheduler -eq $scheduler }).Count -eq 3) "$scheduler has seeds 0, 1, and 2"
    $checks += Assert-Condition ($runIds -contains "scheduler_${scheduler}_seed0_steps2000") "$scheduler seed0 run exists"
}

if (!(Test-Path $MetaManifest)) {
    throw "Missing meta-learning manifest: $MetaManifest"
}
$metaRows = Import-Csv $MetaManifest
$checks += Assert-Condition ($metaRows.Count -eq 7) "meta-learning context manifest has seven seed-0 rows"
$checks += Assert-Condition (@($metaRows | Where-Object { $_.scheduler -ne "ats" }).Count -eq 0) "meta-learning context is fixed to ats scheduler"
$checks += Assert-Condition (@($metaRows | Where-Object { [int]$_.seed -ne 0 }).Count -eq 0) "meta-learning context is seed 0 only"
foreach ($algorithm in @("fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge")) {
    $checks += Assert-Condition (@($metaRows | Where-Object { $_.meta_algorithm -eq $algorithm }).Count -eq 1) "$algorithm context row exists"
}

$todoText = Get-Content "TODO.md" -Raw
$runbookText = Get-Content "papers/meta_learning/STAGE2_RUNBOOK.md" -Raw
$planText = Get-Content "papers/meta_learning/BENCHMARK_PLAN.md" -Raw
$readmeText = Get-Content "papers/meta_learning/README.md" -Raw
foreach ($runId in @("meta_fomaml_ats_seed0_steps2000", "meta_maml_ats_seed0_steps2000", "meta_anil_ats_seed0_steps2000", "meta_reptile_ats_seed0_steps2000", "meta_matching_net_ats_seed0_steps2000", "meta_proto_net_ats_seed0_steps2000", "meta_r2d2_ridge_ats_seed0_steps2000")) {
    $checks += Assert-Condition ($todoText -match [regex]::Escape($runId)) "TODO lists $runId"
    $checks += Assert-Condition ($runbookText -match [regex]::Escape($runId)) "runbook lists $runId"
    $checks += Assert-Condition ($planText -match [regex]::Escape($runId)) "benchmark plan lists $runId"
}
$checks += Assert-Condition (($runbookText -match "r2d2_ridge") -and ($runbookText -match "MetaOptNet")) "runbook documents R2D2-style ridge baseline and MetaOptNet boundary"
$checks += Assert-Condition (($todoText -match "r2d2_ridge") -and ($todoText -match "MetaOptNet")) "TODO documents R2D2-style ridge baseline and MetaOptNet boundary"
$checks += Assert-Condition (($planText -match "DERTS") -and ($planText -match "not currently implemented")) "benchmark plan lists DERTS as optional not currently implemented"
$checks += Assert-Condition (($todoText -match "DERTS") -and ($todoText -match "online scalar feedback")) "TODO records DERTS proxy approximation requirement"
$checks += Assert-Condition (($planText -match "derts_proxy") -and ($todoText -match "derts_proxy")) "docs list derts_proxy as exploratory runnable proxy"
$checks += Assert-Condition (($todoText -match "uniform") -and ($todoText -match "delta_vs_uniform")) "TODO states uniform seed0 eval anchors scheduler deltas"
$checks += Assert-Condition (($runbookText -match "uniform") -and ($runbookText -match "delta_vs_uniform")) "runbook states uniform seed0 eval anchors scheduler deltas"
$checks += Assert-Condition ($readmeText -match "stage2_run_next_benchmark_goal") "README recommends benchmark-goal queue wrapper"
$checks += Assert-Condition (($readmeText -match "cycle") -and ($readmeText -match "excluded\s+from\s+the\s+main\s+compute\s+path")) "README excludes cycle/hard_task/progress from main compute path"
$goalQueueText = Get-Content "scripts/stage2_run_next_benchmark_goal.ps1" -Raw
$checks += Assert-Condition (($goalQueueText -match "AllowConcurrent") -and ($goalQueueText -match "Active Stage 2 Python process")) "benchmark-goal queue guards against concurrent Stage 2 Python jobs"
$checks += Assert-Condition (($goalQueueText.IndexOf("`$metaPriority") -ge 0) -and ($goalQueueText.IndexOf("`$schedulerPriority") -ge 0) -and ($goalQueueText.IndexOf("`$metaPriority") -lt $goalQueueText.IndexOf("`$schedulerPriority"))) "benchmark-goal queue prioritizes SOTA/canonical meta-learning rows before scheduler ablations"
$checks += Assert-Condition (($todoText -notmatch "defer full SOTA/canonical") -and ($readmeText -notmatch "secondary\s+fixed-scheduler\s+context") -and ($runbookText -notmatch "Run only after the scheduler benchmark")) "docs do not defer SOTA/canonical meta-learning comparison behind scheduler sweeps"
$schedulerStatusText = Get-Content "scripts/stage2_scheduler_status.ps1" -Raw
$checks += Assert-Condition (($schedulerStatusText.Contains("manifestArg")) -and ($schedulerStatusText.Contains("eval_stage2_scheduler_run.ps1 `$manifestArg"))) "scheduler status preserves selected manifest in suggested commands"

$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_scheduler.ps1") "single-step scheduler training wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_eval_next_scheduler.ps1") "single-step scheduler evaluation wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_finalize_scheduler_outputs.ps1") "scheduler finalize wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_benchmark_goal.ps1") "recommended benchmark-goal queue wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_benchmark_goal_status.ps1") "benchmark-goal status wrapper exists"
$goalStatusText = Get-Content "scripts/stage2_benchmark_goal_status.ps1" -Raw
$trainingWrapperText = Get-Content "scripts/run_stage2_scheduler_training.ps1" -Raw
$trainingScriptText = Get-Content "scripts/run_stage2_scheduler_training.py" -Raw
$pilotQueueText = Get-Content "scripts/stage2_run_next_scheduler_pilot.ps1" -Raw
$schedulerAggregateText = Get-Content "scripts/stage2_aggregate_scheduler_benchmark.ps1" -Raw
$metaAggregateText = Get-Content "scripts/stage2_aggregate_meta_learning_benchmark.ps1" -Raw
$checks += Assert-Condition (($goalStatusText -match "MinutesSinceTimingWrite") -and ($goalStatusText -match "TimingLastWrite")) "benchmark-goal status shows active-process progress freshness"
$checks += Assert-Condition (($goalStatusText -match "EtaHours") -and ($goalStatusText -match "CurrentStep")) "benchmark-goal status estimates active-process ETA"
$checks += Assert-Condition (($goalStatusText -match "EtaToCheckpointHours") -and ($goalStatusText -match "NextCheckpointStep")) "benchmark-goal status estimates next-checkpoint ETA"
$checks += Assert-Condition (($goalStatusText -match "HasLatestCheckpoint") -and ($goalStatusText -match "eval_stage2_latest_checkpoint") -and ($goalStatusText -match "ManifestCsv")) "benchmark-goal status suggests manifest-aware latest-checkpoint quick-look eval"
$checks += Assert-Condition (($goalStatusText.IndexOf("group = `"meta_context`"") -ge 0) -and ($goalStatusText.IndexOf("group = `"scheduler_seed0`"") -ge 0) -and ($goalStatusText.IndexOf("group = `"meta_context`"") -lt $goalStatusText.IndexOf("group = `"scheduler_seed0`""))) "benchmark-goal status lists SOTA/canonical meta-learning rows before scheduler ablations"
$checks += Assert-Condition (($trainingWrapperText -match "SaveInterval") -and ($trainingWrapperText -match "LogInterval")) "training wrapper forwards save/log interval overrides"
$checks += Assert-Condition (($pilotQueueText -match "SaveInterval = 10") -and ($pilotQueueText -match "LogInterval = 10")) "pilot queue saves/logs every 10 steps by default"
$checks += Assert-Condition (($schedulerAggregateText -match "Export-CsvWithSchema") -and ($schedulerAggregateText -match "normalized_scheduler_results")) "scheduler aggregation writes schema for empty CSV outputs"
$checks += Assert-Condition (($metaAggregateText -match "Export-CsvWithSchema") -and ($metaAggregateText -match "normalized_meta_learning_results")) "meta-learning aggregation writes schema for empty CSV outputs"
$checks += Assert-Condition (Test-Path "scripts/stage2_scheduler_pilot_manifest.ps1") "scheduler pilot manifest wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_scheduler_pilot.ps1") "scheduler pilot queue wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_scheduler_pilot_status.ps1") "scheduler pilot status wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_pilot_goal_status.ps1") "unified pilot status wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_pilot_goal.ps1") "unified pilot queue wrapper exists"
$pilotStatusText = Get-Content "scripts/stage2_pilot_goal_status.ps1" -Raw
$checks += Assert-Condition (($pilotStatusText.IndexOf("group = `"meta_pilot`"") -ge 0) -and ($pilotStatusText.IndexOf("group = `"scheduler_pilot`"") -ge 0) -and ($pilotStatusText.IndexOf("group = `"meta_pilot`"") -lt $pilotStatusText.IndexOf("group = `"scheduler_pilot`""))) "unified pilot status lists SOTA/canonical meta-learning pilot before scheduler pilot"
$pilotGoalText = Get-Content "scripts/stage2_run_next_pilot_goal.ps1" -Raw
$checks += Assert-Condition (($pilotGoalText.IndexOf("stage2_run_next_meta_learning_pilot") -ge 0) -and ($pilotGoalText.IndexOf("stage2_run_next_scheduler_pilot") -ge 0) -and ($pilotGoalText.IndexOf("stage2_run_next_meta_learning_pilot") -lt $pilotGoalText.IndexOf("stage2_run_next_scheduler_pilot"))) "unified pilot queue prioritizes SOTA/canonical meta-learning pilot before scheduler pilot"
$checks += Assert-Condition (Test-Path "scripts/stage2_smoke_meta_learning_algorithms.ps1") "meta-learning algorithm smoke wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_smoke_meta_algorithm_units.ps1") "meta-learning algorithm unit smoke wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_smoke_derts_proxy_scheduler.ps1") "DERTS proxy scheduler smoke wrapper exists"
$checks += Assert-Condition (Test-Path "tests/test_stage2_meta_algorithms.py") "meta-learning algorithm unit tests exist"
$checks += Assert-Condition (Test-Path "scripts/stage2_meta_learning_pilot_manifest.ps1") "meta-learning pilot manifest wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_meta_learning_pilot_status.ps1") "meta-learning pilot status wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_meta_learning_pilot.ps1") "meta-learning pilot queue wrapper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_all_sota_meta_learning_pilot.ps1") "SOTA meta-learning all-row pilot runner exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_sota_meta_learning_pilot_completion_gate.ps1") "SOTA meta-learning pilot completion gate exists"
$sotaAllPilotText = Get-Content "scripts/stage2_run_all_sota_meta_learning_pilot.ps1" -Raw
$pilotGateText = Get-Content "scripts/stage2_sota_meta_learning_pilot_completion_gate.ps1" -Raw
$checks += Assert-Condition (($sotaAllPilotText -match "stage2_run_next_meta_learning_pilot") -and ($sotaAllPilotText -match "stage2_finalize_meta_learning_outputs") -and ($sotaAllPilotText -match "stage2_sota_meta_learning_pilot_completion_gate") -and ($sotaAllPilotText -match "Active Stage 2 Python process") -and ($sotaAllPilotText -match "made no progress") -and ($sotaAllPilotText -match "OutputDir") -and ($sotaAllPilotText -match "\@\(Get-ActiveStage2Python\)")) "SOTA all-row pilot runner chains pilot rows, output-dir-aware finalization, pilot gate, active-job guard, and no-progress guard"
$checks += Assert-Condition (($pilotGateText -match "steps -eq 50") -and ($pilotGateText -match "n_support -eq 8") -and ($pilotGateText -match "Missing evaluation CSV") -and ($pilotGateText -match "Pending") -and ($pilotGateText -match "SOTA/canonical")) "SOTA pilot gate enforces pilot shape, evaluated rows, and non-placeholder SOTA table"
$checks += Assert-Condition ($readmeText -match "stage2_smoke_meta_learning_algorithms") "README documents meta-learning algorithm smoke wrapper"
$checks += Assert-Condition ($readmeText -match "stage2_smoke_meta_algorithm_units") "README documents meta-learning algorithm unit smoke wrapper"
$checks += Assert-Condition (($todoText -match 'instead of assuming `pytest`') -and ($readmeText -match 'instead of assuming `pytest`') -and ($runbookText -match 'instead of assuming `pytest`')) "docs note pytest-free smoke wrappers for the conda environment"
$checks += Assert-Condition ($readmeText -match "stage2_smoke_derts_proxy_scheduler") "README documents DERTS proxy smoke wrapper"
$checks += Assert-Condition (($readmeText -match "stage2_meta_learning_pilot_manifest") -and ($readmeText -match "stage2_run_all_sota_meta_learning_pilot") -and ($readmeText -match "stage2_sota_meta_learning_pilot_completion_gate")) "README documents meta-learning pilot, all-row pilot runner, and pilot gate"
$checks += Assert-Condition ($readmeText -match "stage2_run_next_pilot_goal") "README documents unified pilot queue"
$checks += Assert-Condition ($runbookText -match "stage2_smoke_meta_learning_algorithms") "runbook documents meta-learning algorithm smoke wrapper"
$checks += Assert-Condition ($runbookText -match "stage2_smoke_meta_algorithm_units") "runbook documents meta-learning algorithm unit smoke wrapper"
$checks += Assert-Condition (($runbookText -match "stage2_meta_learning_pilot_manifest") -and ($runbookText -match "stage2_run_all_sota_meta_learning_pilot") -and ($runbookText -match "stage2_sota_meta_learning_pilot_completion_gate")) "runbook documents meta-learning pilot, all-row pilot runner, and pilot gate"
$checks += Assert-Condition ($runbookText -match "stage2_run_next_pilot_goal") "runbook documents unified pilot queue"
$checks += Assert-Condition (Test-Path "scripts/stage2_benchmark_decision_report.ps1") "benchmark decision report exists"
$decisionReportText = Get-Content "scripts/stage2_benchmark_decision_report.ps1" -Raw
$checks += Assert-Condition (($decisionReportText -match "latest_checkpoint") -and ($decisionReportText -match "eval_stage2_latest_checkpoint") -and ($decisionReportText -match "MetaManifest") -and ($decisionReportText -match "manifest_csv")) "decision report suggests manifest-aware latest-checkpoint quick-look eval"
$checks += Assert-Condition (($decisionReportText -match "stage2_run_next_meta_learning_pilot") -and ($decisionReportText -match "stage2_run_next_pilot_goal") -and ($decisionReportText -notmatch "first-pass scheduler ranking")) "decision report recommends SOTA/canonical meta-learning pilot before scheduler-first pilot"
$checks += Assert-Condition (Test-Path "papers/meta_learning/SOTA_META_LEARNING_READINESS.md") "SOTA meta-learning readiness note exists"
$readinessText = Get-Content "papers/meta_learning/SOTA_META_LEARNING_READINESS.md" -Raw
$checks += Assert-Condition (($readinessText -match "matching_net") -and ($readinessText -match "proto_net") -and ($readinessText -match "r2d2_ridge")) "readiness note lists SOTA meta-learning context baselines"
$checks += Assert-Condition (($readinessText -match "meta_.*_uniform_.*") -and ($readinessText -match "ignored")) "readiness note marks legacy uniform meta outputs as ignored"
$checks += Assert-Condition (Test-Path "scripts/stage2_sota_meta_learning_readiness.ps1") "SOTA meta-learning readiness wrapper exists"
$readinessScriptText = Get-Content "scripts/stage2_sota_meta_learning_readiness.ps1" -Raw
$checks += Assert-Condition (($readinessScriptText -match "Ignored legacy output directories") -and ($readinessScriptText -match "meta_.*_uniform_.*") -and ($readinessScriptText -match "Active Stage 2 Python process")) "SOTA readiness wrapper reports ignored legacy outputs and active jobs"
$checks += Assert-Condition (Test-Path "scripts/stage2_sota_meta_learning_execution_plan.ps1") "SOTA meta-learning execution plan wrapper exists"
$executionPlanText = Get-Content "scripts/stage2_sota_meta_learning_execution_plan.ps1" -Raw
$checks += Assert-Condition (($executionPlanText -match "stage2_run_next_sota_meta_learning_context") -and ($executionPlanText -match "Formal row order") -and ($executionPlanText -match "Active Stage 2 job detected") -and ($executionPlanText -match "stage2_sota_meta_learning_completion_gate")) "SOTA execution plan lists formal row order, active-job warning, and completion gate"
$checks += Assert-Condition (Test-Path "scripts/stage2_sota_meta_learning_completion_gate.ps1") "SOTA meta-learning completion gate exists"
$completionGateText = Get-Content "scripts/stage2_sota_meta_learning_completion_gate.ps1" -Raw
$checks += Assert-Condition (($completionGateText -match "requiredAlgorithms") -and ($completionGateText -match "Missing evaluation CSV") -and ($completionGateText -match "Pending") -and ($completionGateText -match "Optional") -and ($completionGateText -match "SOTA/canonical") -and ($completionGateText -match "non-formal run_id") -and ($completionGateText -match "non-fixed scheduler/seed")) "SOTA completion gate requires all formal rows, uncontaminated outputs, non-placeholder tables, and correct caption wording"
$mainMetaTableText = Get-Content "papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/paper_table_meta_algorithm_summary.tex" -Raw
$pilotMetaTableText = Get-Content "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/paper_table_meta_algorithm_summary.tex" -Raw
$checks += Assert-Condition (($mainMetaTableText -match "SOTA/canonical") -and ($mainMetaTableText -notmatch "Optional")) "main meta-learning paper table uses SOTA/canonical wording"
$checks += Assert-Condition (($pilotMetaTableText -match "SOTA/canonical") -and ($pilotMetaTableText -notmatch "Optional")) "pilot meta-learning paper table uses SOTA/canonical wording"
$checks += Assert-Condition (($todoText -match "stage2_sota_meta_learning_readiness") -and ($readmeText -match "stage2_sota_meta_learning_readiness") -and ($readinessText -match "stage2_sota_meta_learning_readiness")) "docs mention SOTA readiness wrapper"
$checks += Assert-Condition (Test-Path "scripts/stage2_stop_active_run.ps1") "explicit active-run stop helper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_switch_to_sota_meta_learning_now.ps1") "SOTA meta-learning switch helper exists"
$switchHelperText = Get-Content "scripts/stage2_switch_to_sota_meta_learning_now.ps1" -Raw
$checks += Assert-Condition (($switchHelperText -match "stage2_stop_active_run") -and ($switchHelperText -match "stage2_run_next_meta_learning_pilot") -and ($switchHelperText -match "stage2_run_next_sota_meta_learning_context") -and ($switchHelperText -match "ForceNoCheckpoint")) "SOTA switch helper previews/stops active run and starts pilot or full context only when requested"
$checks += Assert-Condition (Test-Path "scripts/stage2_wait_for_checkpoint.ps1") "checkpoint wait helper exists"
$checks += Assert-Condition (Test-Path "scripts/stage2_wait_checkpoint_then_sota_meta_learning.ps1") "checkpoint-to-SOTA meta-learning helper exists"
$waitThenSotaText = Get-Content "scripts/stage2_wait_checkpoint_then_sota_meta_learning.ps1" -Raw
$checks += Assert-Condition (($waitThenSotaText -match "stage2_wait_for_checkpoint") -and ($waitThenSotaText -match "eval_stage2_latest_checkpoint") -and ($waitThenSotaText -match "stage2_stop_active_run") -and ($waitThenSotaText -match "stage2_run_all_sota_meta_learning_pilot") -and ($waitThenSotaText -match "stage2_run_all_sota_meta_learning_context") -and ($waitThenSotaText -match "Preview command sequence")) "checkpoint-to-SOTA helper previews wait/eval/stop/start-pilot/full sequence"
$waitHelperText = Get-Content "scripts/stage2_wait_for_checkpoint.ps1" -Raw
$checks += Assert-Condition (($waitHelperText -match "eval_stage2_latest_checkpoint") -and ($waitHelperText -match "stage2_meta_learning_pilot_benchmark") -and ($waitHelperText -match "Get-ManifestForCheckpoint") -and ($waitHelperText -match "ManifestCsv")) "checkpoint wait helper suggests manifest-aware latest eval and searches pilot dirs"
$checks += Assert-Condition (Test-Path "scripts/eval_stage2_latest_checkpoint.ps1") "latest-checkpoint evaluation helper exists"
$latestEvalText = Get-Content "scripts/eval_stage2_latest_checkpoint.ps1" -Raw
$checks += Assert-Condition (($latestEvalText -match "latest.pth") -and ($latestEvalText -match "attack_eval_summary_latest.csv")) "latest-checkpoint evaluation writes separate quick-look CSV"
$stopHelperText = Get-Content "scripts/stage2_stop_active_run.ps1" -Raw
$checks += Assert-Condition (($stopHelperText -match "ForceNoCheckpoint") -and ($stopHelperText -match "Refusing to stop")) "active-run stop helper protects runs without checkpoints"
$checks += Assert-Condition (($todoText -match "stage2_stop_active_run") -and ($runbookText -match "stage2_stop_active_run")) "docs mention explicit active-run stop helper"
$checks += Assert-Condition (($todoText -match "stage2_switch_to_sota_meta_learning_now") -and ($runbookText -match "stage2_switch_to_sota_meta_learning_now") -and ($readmeText -match "stage2_switch_to_sota_meta_learning_now")) "docs mention SOTA meta-learning switch helper"
$checks += Assert-Condition (($todoText -match "stage2_wait_checkpoint_then_sota_meta_learning") -and ($runbookText -match "stage2_wait_checkpoint_then_sota_meta_learning") -and ($readmeText -match "stage2_wait_checkpoint_then_sota_meta_learning")) "docs mention checkpoint-to-SOTA meta-learning helper"
if (Test-Path "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv") {
    $pilotRows = @(Import-Csv "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/scheduler_runs.csv")
    $checks += Assert-Condition ($pilotRows.Count -eq 5) "scheduler pilot manifest has five seed-0 rows"
    $checks += Assert-Condition (@($pilotRows | Where-Object { [int]$_.seed -ne 0 }).Count -eq 0) "scheduler pilot uses seed 0 only"
    $checks += Assert-Condition (@($pilotRows | Where-Object { [int]$_.steps -ne 50 }).Count -eq 0) "scheduler pilot uses 50 training steps"
    foreach ($scheduler in @("uniform", "ats", "bass", "bandit_ucb", "residual")) {
        $checks += Assert-Condition (@($pilotRows | Where-Object { $_.scheduler -eq $scheduler }).Count -eq 1) "scheduler pilot includes $scheduler"
    }
    $schedulerPilotNormalized = "papers/meta_learning/benchmark_outputs/stage2_scheduler_pilot_benchmark/normalized_scheduler_results.csv"
    if (Test-Path $schedulerPilotNormalized) {
        $checks += Assert-Condition ((Get-Item $schedulerPilotNormalized).Length -gt 0) "scheduler pilot normalized CSV has header even before results"
    }
}
if (Test-Path "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv") {
    $metaPilotRows = @(Import-Csv "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/meta_learning_runs.csv")
    $checks += Assert-Condition ($metaPilotRows.Count -eq 7) "meta-learning pilot manifest has seven seed-0 rows"
    $checks += Assert-Condition (@($metaPilotRows | Where-Object { [int]$_.seed -ne 0 }).Count -eq 0) "meta-learning pilot uses seed 0 only"
    $checks += Assert-Condition (@($metaPilotRows | Where-Object { [int]$_.steps -ne 50 }).Count -eq 0) "meta-learning pilot uses 50 training steps"
    $checks += Assert-Condition (@($metaPilotRows | Where-Object { $_.scheduler -ne "ats" }).Count -eq 0) "meta-learning pilot fixes scheduler to ats"
    foreach ($algorithm in @("fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge")) {
        $checks += Assert-Condition (@($metaPilotRows | Where-Object { $_.meta_algorithm -eq $algorithm }).Count -eq 1) "meta-learning pilot includes $algorithm"
    }
    $metaPilotNormalized = "papers/meta_learning/benchmark_outputs/stage2_meta_learning_pilot_benchmark/normalized_meta_learning_results.csv"
    if (Test-Path $metaPilotNormalized) {
        $checks += Assert-Condition ((Get-Item $metaPilotNormalized).Length -gt 0) "meta-learning pilot normalized CSV has header even before results"
    }
}
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_meta_learning_cycle.ps1") "meta-learning context cycle wrapper exists"
$metaCycleText = Get-Content "scripts/stage2_run_next_meta_learning_cycle.ps1" -Raw
$checks += Assert-Condition (($metaCycleText -match "AllowConcurrent") -and ($metaCycleText -match "Active Stage 2 Python process") -and ($metaCycleText -match "SOTA meta-learning context job")) "meta-learning context cycle wrapper guards against concurrent Stage 2 jobs"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_next_sota_meta_learning_context.ps1") "SOTA meta-learning context runner alias exists"
$sotaContextRunnerText = Get-Content "scripts/stage2_run_next_sota_meta_learning_context.ps1" -Raw
$checks += Assert-Condition (($sotaContextRunnerText -match "stage2_run_next_meta_learning_cycle") -and ($sotaContextRunnerText -match "AllowConcurrent")) "SOTA meta-learning context runner alias forwards to guarded runner"
$checks += Assert-Condition (Test-Path "scripts/stage2_run_all_sota_meta_learning_context.ps1") "SOTA meta-learning all-row runner exists"
$sotaAllRunnerText = Get-Content "scripts/stage2_run_all_sota_meta_learning_context.ps1" -Raw
$checks += Assert-Condition (($sotaAllRunnerText -match "stage2_run_next_sota_meta_learning_context") -and ($sotaAllRunnerText -match "stage2_finalize_meta_learning_outputs") -and ($sotaAllRunnerText -match "stage2_sota_meta_learning_completion_gate") -and ($sotaAllRunnerText -match "Preview next row only") -and ($sotaAllRunnerText -match "OutputDir") -and ($sotaAllRunnerText -match "Active Stage 2 Python process") -and ($sotaAllRunnerText -match "made no progress") -and ($sotaAllRunnerText -match "\@\(Get-ActiveStage2Python\)")) "SOTA all-row runner chains rows, output-dir-aware finalization, completion gate, preview mode, active-job guard, and no-progress guard"
$checks += Assert-Condition (($todoText -match "stage2_run_next_sota_meta_learning_context") -and ($runbookText -match "stage2_run_next_sota_meta_learning_context") -and ($planText -match "stage2_run_next_sota_meta_learning_context") -and ($readinessText -match "stage2_run_next_sota_meta_learning_context")) "docs prefer clearly named SOTA meta-learning context runner"
$checks += Assert-Condition (($todoText -match "stage2_run_all_sota_meta_learning_context") -and ($runbookText -match "stage2_run_all_sota_meta_learning_context") -and ($planText -match "stage2_run_all_sota_meta_learning_context") -and ($readmeText -match "stage2_run_all_sota_meta_learning_context")) "docs mention SOTA meta-learning all-row runner"
$checks += Assert-Condition (Test-Path "papers/meta_learning/STAGE2_RUNBOOK.md") "Stage 2 runbook exists"
$checks += Assert-Condition (Select-String -Path "papers/meta_learning/sec/4_experiments.tex" -Pattern "paper_table_scheduler_summary" -Quiet) "paper includes scheduler summary table"
$checks += Assert-Condition (Select-String -Path "papers/meta_learning/sec/4_experiments.tex" -Pattern "paper_table_meta_algorithm_summary" -Quiet) "paper includes SOTA meta-learning context table"
$methodText = Get-Content "papers/meta_learning/sec/3_method.tex" -Raw
$relatedText = Get-Content "papers/meta_learning/sec/2_related_work.tex" -Raw
$bibText = Get-Content "papers/meta_learning/main.bib" -Raw
$schedulerBaselineText = Get-Content "scheduler_baselines.py" -Raw
$datasetText = Get-Content "ds.py" -Raw
$checks += Assert-Condition (($methodText -match "ATS-style") -and ($methodText -match "official reproductions")) "paper states ATS/BASS-style baselines are not official reproductions"
$checks += Assert-Condition (($relatedText -match "DERTS") -and ($bibText -match "zhan2024derts")) "paper cites DERTS task-selection baseline"
$checks += Assert-Condition (($methodText -match "derts\\_proxy") -and ($methodText -match "not an official DERTS")) "paper states derts_proxy is not official DERTS"
$checks += Assert-Condition (($schedulerBaselineText -match "derts_proxy") -and ($datasetText -match "derts_proxy")) "derts_proxy scheduler is wired into controller and dataset"
$checks += Assert-Condition (($trainingScriptText -match "r2d2_ridge") -and ($methodText -match "r2d2\\_ridge")) "r2d2_ridge meta-learning baseline is wired into trainer and paper"
$checks += Assert-Condition (($trainingScriptText -match "proto_net") -and ($methodText -match "proto\\_net") -and ($bibText -match "snell2017prototypical")) "proto_net meta-learning baseline is wired into trainer and paper"
$checks += Assert-Condition (($trainingScriptText -match "matching_net") -and ($methodText -match "matching\\_net") -and ($bibText -match "vinyals2016matching")) "matching_net meta-learning baseline is wired into trainer and paper"
$checks += Assert-Condition (($trainingScriptText -match '"maml"') -and ($methodText -match "second-order MAML") -and ($bibText -match "finn2017maml")) "maml meta-learning baseline is wired into trainer and paper"
$discussionText = Get-Content "papers/meta_learning/sec/5_discussion.tex" -Raw
$checks += Assert-Condition (($discussionText -match "fixed-scheduler SOTA/canonical meta-learning table") -and ($discussionText -match "Matching Networks") -and ($discussionText -match "R2D2")) "discussion prioritizes SOTA/canonical meta-learning baselines"
$conclusionText = Get-Content "papers/meta_learning/sec/6_conclusion.tex" -Raw
$checks += Assert-Condition (($conclusionText -match "SOTA/canonical meta-learning comparison") -and ($conclusionText -match "FOMAML") -and ($conclusionText -match "R2D2-style ridge")) "conclusion prioritizes SOTA/canonical meta-learning comparison"

$checks | Format-Table -AutoSize

$failed = @($checks | Where-Object { $_.status -ne "ok" })
if ($failed.Count -gt 0) {
    throw "Stage 2 benchmark audit failed $($failed.Count) check(s)."
}

Write-Host ""
Write-Host "Stage 2 benchmark audit passed."
