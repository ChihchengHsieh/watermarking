# Stage 2 Runbook

This runbook is the current execution path for the MetaSpiderMark benchmark.
The main benchmark is not a large uniform/cycle sweep. It uses `uniform` only as
a seed-0 anchor, skips `cycle` as a main compute target, and prioritizes
SOTA/canonical meta-learning comparisons under a fixed scheduler.
SOTA-inspired/adaptive task schedulers remain the scheduler ablation.
The current ATS/BASS entries are local implementations of the scheduling ideas,
not official reproductions of the original papers.

## 0. Check Status

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_goal_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1
```

Audit that the benchmark setup still matches the current plan:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_audit.ps1
```

Expected current total-queue step:

```text
Evaluate scheduler_uniform_seed0_steps2000 so delta_vs_uniform.csv has an anchor.
```

Expected first SOTA/canonical meta-learning step after the anchor evaluation:

```text
Next SOTA/canonical meta-learning command:
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_fomaml_ats_seed0_steps2000
```

## Recommended Queue

First-pass ranking should use the short pilot benchmark. The current 2000-step
training speed is roughly minutes per step, so full runs should wait until the
pilot identifies which schedulers are worth compute.

Generate the pilot manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_decision_report.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_pilot_goal_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_pilot_goal.ps1 -Execute
```

Run one pilot row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_pilot.ps1 -Execute
```

The pilot contains seed-0 `uniform`, `ats`, `bass`, `bandit_ucb`, and
`residual` at 50 steps with support/query size 8 and a reduced evaluation suite.
It saves/logs every 10 steps by default so interrupted pilot runs can resume from
`checkpoints/latest.pth`.

If a long 2000-step run is already active and you intentionally want to abandon
it before starting the pilot, preview the stop command first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
```

Only execute after confirming the matched process is correct:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -Execute
```

If no `latest.pth` or `final.pth` exists and discarding current progress is
intentional:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_stop_active_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -Execute -ForceNoCheckpoint
```

Safer switch wrapper for moving directly to the SOTA/canonical meta-learning
pilot:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_switch_to_sota_meta_learning_now.ps1 -Execute -ForceNoCheckpoint -StartPilot
```

To wait until the long run reaches a checkpoint without stopping it:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_for_checkpoint.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_latest_checkpoint.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
```

To wait for the checkpoint, run a quick-look evaluation, stop the active run
after the checkpoint, and then start the SOTA/canonical pilot:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_checkpoint_then_sota_meta_learning.ps1 -EvalLatest -StopAfterCheckpoint -StartPilot
powershell -ExecutionPolicy Bypass -File scripts\stage2_wait_checkpoint_then_sota_meta_learning.ps1 -Execute -EvalLatest -StopAfterCheckpoint -StartPilot
```

The latest-checkpoint evaluation writes `attack_eval_summary_latest.csv`, not
the final manifest CSV.

Use this command to follow the current paper goal automatically. It completes
the existing uniform seed-0 evaluation if needed, then the fixed-scheduler
FOMAML/MAML/ANIL/Reptile/MatchingNet/ProtoNet/R2D2-style ridge meta-learning
context rows, then the seed-0 SOTA/adaptive scheduler rows. The uniform
step is evaluation-only; it does not
train additional uniform runs. The wrapper refuses to start another Stage 2
Python job while scheduler training/evaluation is already active, unless
`-AllowConcurrent` is passed intentionally.

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1
```

Run one next row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1 -Execute
```

## 1. Train SOTA/Canonical Meta-Learning Rows

Run the required fixed-scheduler comparison first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
```

These rows compare FOMAML, MAML, ANIL, Reptile, Matching Networks-style,
Prototypical Networks-style, and R2D2-style ridge under the same ATS scheduler
and seed-0 setting.

To run all seven rows sequentially after the GPU is free:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1 -Execute
```

## 2. Train Priority Seed-0 Schedulers

One-command path: run the next training row, evaluate it, then finalize outputs.

Preview:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_cycle.ps1
```

Run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler_cycle.ps1 -Execute
```

Safest path: run one priority scheduler at a time.

Preview the next run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1
```

Run the next run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1 -Execute
```

Batch path: preview all priority seed-0 runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1
```

Batch path: run all priority seed-0 runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1 -Execute
```

This runs, in order:

```text
scheduler_ats_seed0_steps2000
scheduler_bass_seed0_steps2000
scheduler_bandit_ucb_seed0_steps2000
scheduler_residual_seed0_steps2000
```

The wrapper skips any run that already has `checkpoints/final.pth`.

## 3. Evaluate Completed Priority Seed-0 Schedulers

Safest path: evaluate one ready scheduler at a time.

Preview the next ready evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1
```

Run the next ready evaluation:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1 -Execute
```

Preview ready evaluations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1
```

Run ready evaluations:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1 -Execute
```

To include the uniform seed-0 anchor:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1 -IncludeAnchor -Execute
```

## 4. Aggregate And Generate Tables

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
```

Important outputs:

```text
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/summary_by_scheduler.csv
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/delta_vs_uniform.csv
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_summary.tex
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_delta.tex
```

## 5. Expand Seeds

After seed 0 establishes the ranking, run seeds 1 and 2 for the strongest
schedulers:

```text
scheduler_ats_seed1_steps2000
scheduler_ats_seed2_steps2000
scheduler_bass_seed1_steps2000
scheduler_bass_seed2_steps2000
scheduler_bandit_ucb_seed1_steps2000
scheduler_bandit_ucb_seed2_steps2000
scheduler_residual_seed1_steps2000
scheduler_residual_seed2_steps2000
```

Do not add `cycle`, `hard_task`, or `progress` to the main compute path unless
the adaptive scheduler comparison is already complete.
DERTS-style task subset selection is a relevant recent optional baseline, but it
requires a separate task-pool gradient approximation implementation and is not
part of the current runnable queue.
The runnable `derts_proxy` mode is only a local online-feedback approximation
and should be treated as an exploratory appendix row unless explicitly selected.

## Required: SOTA Meta-Learning Algorithm Context

This is required paper context for comparison against common meta-learning
families. It is fixed to `ats` scheduler and seed 0 by default.
The implemented rows are:

```text
meta_fomaml_ats_seed0_steps2000
meta_maml_ats_seed0_steps2000
meta_anil_ats_seed0_steps2000
meta_reptile_ats_seed0_steps2000
meta_matching_net_ats_seed0_steps2000
meta_proto_net_ats_seed0_steps2000
meta_r2d2_ridge_ats_seed0_steps2000
```

`matching_net` is implemented as a Matching Networks-style attention metric
baseline. `proto_net` is implemented as a Prototypical Networks-style metric baseline.
`r2d2_ridge` is implemented as a differentiable closed-form ridge solver-head
baseline inspired by R2D2. It is not a full MetaOptNet SVM/QP solver-head
reproduction, and full MetaOptNet-style heads should not block the scheduler
benchmark.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_learning_algorithms.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_algorithm_units.ps1
```

Use these wrappers instead of assuming `pytest` is installed in the conda
environment.

Low-cost pilot before full 2000-step context rows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_pilot_completion_gate.ps1
```

Direct runner:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
```

The full context runner refuses to start while another Stage 2 Python job is
active unless `-AllowConcurrent` is passed intentionally.

Run before expanding scheduler seeds. The only reason to delay is an already
active Stage 2 job that should not be interrupted.

