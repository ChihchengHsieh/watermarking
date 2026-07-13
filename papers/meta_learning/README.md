# MetaSpiderMark Paper Draft

This folder contains the CVPR-style draft for the meta-learning extension of
SpiderMark.

Working title:

> MetaSpiderMark: Meta-Learned Verification for Robust Diffusion Watermarking

The paper's current claim is intentionally focused:

> SpiderMark's watermark injection remains unchanged, but verifier training is
> reformulated as downstream attack episodic meta-learning so that the verifier
> generalizes better across transformations and checkpoints.

## Current Paper State

- `main.tex` contains the CVPR entry point.
- `sec/0_abstract.tex` through `sec/6_conclusion.tex` contain the current draft.
- `figures/` contains copied paper-local figures used by the draft.
- `main.bib` contains the minimal citations currently used by the draft.
- `BENCHMARK_PLAN.md` records the scheduler benchmark before we turn it into
  final paper tables.
- `STAGE2_RUNBOOK.md` records the current command sequence for the SOTA/adaptive
  scheduler benchmark.

## Current Evidence

The draft includes the available preliminary downstream comparison:

- non-meta SpiderMark verifier mean accuracy / AUROC: `0.8499 / 0.9293`
- meta-learned epoch116 verifier mean accuracy / AUROC: `0.9182 / 0.9686`

This is treated as preliminary evidence, not as a final benchmark.

## Next Benchmark Milestone

The main benchmark should keep SpiderMark's watermark injection, verifier
architecture, meta-learning update, attack pool, and compute budget fixed, then
compare task schedulers:

- `uniform` seed 0 only as the sanity anchor
- `bandit_ucb` as a simple bandit baseline
- `ats` as an ATS-style adaptive task-scheduler baseline, implemented locally
- `bass` as the BASS-style contextual bandit baseline, implemented locally
- `residual` as the local residual controller
- `derts_proxy` as an exploratory DERTS-inspired online task-selection proxy,
  not an official DERTS reproduction

`cycle`, `hard_task`, and `progress` are implemented but intentionally excluded
from the main compute path unless an appendix sanity check is needed after the
adaptive scheduler comparison is complete.

The currently wired `task_sampling` values are:

- `uniform`
- `cycle`
- `hard_task`
- `progress`
- `bandit_ucb`
- `bandit_thompson`
- `ats`
- `bass`
- `derts_proxy`
- `residual`
- `llm_residual`

The runner also has `meta_algorithm` values (`fomaml`, `maml`, `anil`,
`reptile`, `matching_net`, `proto_net`, `r2d2_ridge`). These are the required
SOTA/canonical meta-learning comparison rows under a fixed scheduler.
`matching_net` is a Matching Networks-style attention metric baseline,
`proto_net` is a Prototypical Networks-style metric baseline, and `r2d2_ridge`
is a differentiable closed-form ridge solver-head baseline inspired by R2D2,
not a full MetaOptNet SVM/QP reproduction.

## Build

From this directory:

```powershell
latexmk -pdf -interaction=nonstopmode main.tex
```

The current build produces `main.pdf`. LaTeX build artifacts and PDFs are
ignored by `.gitignore`.

## Stage 1 Aggregation

From the repository root, regenerate the current Stage 1 paper artifacts with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage1_aggregate_benchmark.ps1
```

This writes CSV summaries and the LaTeX table consumed by the experiment section
under `papers/meta_learning/benchmark_outputs/stage1_current/`.

## Stage 2 Scheduler Manifest

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_manifest.ps1
```

Aggregate completed scheduler runs with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
```

Generate paper-ready scheduler tables after aggregation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_make_scheduler_tables.ps1
```

The aggregator accepts partial progress: missing run CSVs are listed in
`papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/missing_runs.csv`.

## Stage 2 Training

For first-pass scheduler ranking, use the short pilot benchmark before spending
compute on 2000-step formal runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_decision_report.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_pilot_goal_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1 -Execute
```

The unified pilot queue runs the fixed-scheduler
FOMAML/MAML/ANIL/Reptile/MatchingNet/ProtoNet/R2D2-style ridge pilot before
expanding scheduler ablations.

If a long formal run is already active and you decide to abandon it before
starting the pilot, preview and then explicitly execute:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -Execute -ForceNoCheckpoint
powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1 -Execute -ForceNoCheckpoint -StartPilot
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_checkpoint_then_sota_meta_learning.ps1 -EvalLatest -StopAfterCheckpoint -StartPilot
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_checkpoint_then_sota_meta_learning.ps1 -Execute -EvalLatest -StopAfterCheckpoint -StartPilot
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_for_checkpoint.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
```

`eval_stage2_latest_checkpoint.ps1` writes `attack_eval_summary_latest.csv` so
quick-look checkpoint evaluation does not overwrite the final benchmark CSV.

Run a dry-run for every planned scheduler job:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -All -DryRun
```

Check the queue and get the next recommended command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_audit.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_goal_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1
```

Recommended queue: run the next row for the current paper goal. This evaluates
the existing `uniform` seed-0 anchor if needed, then runs the fixed-scheduler
FOMAML/MAML/ANIL/Reptile/MatchingNet/ProtoNet/R2D2-style ridge context rows,
then seed-0 ATS/BASS/UCB/residual scheduler ablations. It refuses to start a new Stage 2 Python job
while another training/evaluation job is active, unless `-AllowConcurrent` is
passed intentionally.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1 -Execute
```

Scheduler-only fallback: run or preview one priority scheduler job at a time:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1 -Execute
```

Preview the strongest seed-0 scheduler baselines:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1
```

Run those priority seed-0 scheduler baselines:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1 -Execute
```

After checkpoints exist, preview/evaluate completed priority seed-0 runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1 -Execute
```

Run a tiny one-step smoke test of the full training pipeline:

```powershell
C:\Users\chihc\miniconda3\envs\pytorch\python.exe scripts\run_stage2_scheduler_training.py --scheduler bass --seed 0 --steps 1 --n-support 1 --n-query 1 --meta-batch-size 1 --tasks-per-epoch 2 --log-interval 1 --save-interval 1 --attack-pool clean,jpeg --run-dir papers\meta_learning\benchmark_outputs\stage2_scheduler_benchmark\smoke_bass_seed0_steps1
```

Dry-run the implemented meta-learning algorithm baselines without starting a
training job:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_learning_algorithms.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_algorithm_units.ps1
```

Use these wrappers instead of assuming `pytest` is installed in the conda
environment.

To run actual one-step training smoke tests, first make sure no long Stage 2 GPU
job is active, then pass `-RunTraining`.

The wrapper uses:

```powershell
C:\Users\chihc\miniconda3\envs\pytorch\python.exe
```

Each run writes its final meta checkpoint to the manifest-aligned path:

```text
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/<run_id>/checkpoints/final.pth
```

## Required SOTA Meta-Learning Context

The fixed-scheduler SOTA/canonical meta-learning context is required paper context. It
defaults to `ats` scheduler with seed 0 only and changes only
`meta_algorithm`:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_completion_gate.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_learning_algorithms.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_algorithm_units.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_derts_proxy_scheduler.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
```

For a low-cost FOMAML/MAML/ANIL/Reptile/MatchingNet/ProtoNet/R2D2-style ridge pilot before full
2000-step context runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_pilot_completion_gate.ps1
```

The pilot fixes the scheduler to `ats`, uses seed 0, and runs 50 steps. It is
for debugging and coarse algorithm ranking only.

