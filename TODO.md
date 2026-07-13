# MetaSpiderMark TODO

Decision:

- For the scheduler benchmark, do not use short runs to select which baselines
  deserve the paper table. Fix the comparison set first, then run each selected
  scheduler with the same 2000-step budget.
- Run all selected scheduler training jobs first, then run all evaluations
  together with the same `BatchSize` and `TestingTimes`.
- Current selected scheduler baselines for seed 0 are:
  `uniform`, `bandit_ucb`, `ats`, `bass`, and `asr`.
- `gcp_proxy` is available as a GCP-style appendix row. It is useful if we want
  to mirror the BASS baseline ecosystem, but it is not as central as ATS, BASS,
  and ASr for the main table.
- The proposed MetaSpiderMark result already exists from
  `[3.a] attack_eval_downstream_meta_checkpoint_sweep.ipynb`.
- Do not run `scheduler_residual_seed0_steps2000` as "ours" unless we confirm
  it is the same method/protocol as the notebook result.
- DERTS remains important, but it still needs a faithful gradient-subset
  implementation before it can be launched as official DERTS.
- External reference zips are in `external/`; the active code uses adapted
  local scheduler implementations rather than importing the external training
  frameworks directly.

Current active run:

- [ ] `scheduler_bandit_ucb_seed0_steps2000`
  - Role: strong bandit scheduler baseline.
  - Current state: running as a formal 2000-step scheduler baseline.
  - Let it finish. Do not start another training or eval job on the same GPU
    while it is active.

Existing anchor:

- [x] `scheduler_uniform_seed0_steps2000`
  - Role: simple seed-0 sanity anchor for deltas.
  - Checkpoint exists.
  - Evaluation is incomplete; one partial attack artifact exists.
  - Do not spend more training compute on uniform unless we explicitly need an
    appendix sanity check.

Reference planning document:

- [x] `papers/meta_learning/SOTA_TASK_SCHEDULERS.md`
  - Lists the SOTA task scheduling / task sampling algorithms to compare:
    ATS, BASS, Bandit-UCB, DERTS, ASr, and the proposed MetaSpiderMark
    scheduler, with BASS-table optional baselines SPL, FOCAL, DAML, GCP, and
    PAML.
  - Use this before launching more scheduler compute so the benchmark table is
    tied to actual papers rather than only local scheduler names.

External scheduler integration:

- [x] `external/Bandit_Task_Scheduler-main.zip`
  - Used as reference for BASS-style neural bandit scheduling.
  - Current runnable row remains `scheduler_bass_seed0_steps2000`.
  - Label as BASS-style unless we later reproduce the official neural exploit
    and exploration networks exactly.
- [x] `external/Adaptive-Sampler-main.zip`
  - Used as reference for ASr diversity/entropy/difficulty task sampling.
  - Added runnable local row: `scheduler_asr_seed0_steps2000`.
- [x] `external/gcp-sampling-master.zip`
  - Used as reference for GCP exponential class/task weight updates.
  - Added optional runnable local row: `scheduler_gcp_proxy_seed0_steps2000`.
- [x] Verification passed:
  - `scheduler_baselines.py`, `ds.py`, and
    `scripts/run_stage2_scheduler_training.py` compile.
  - `scheduler_asr_seed0_steps2000` dry-run passes.
  - `scheduler_gcp_proxy_seed0_steps2000` dry-run passes.

Existing proposed-method evaluation:

- [x] `[3.a] attack_eval_downstream_meta_checkpoint_sweep.ipynb`
  - Role: current MetaSpiderMark / proposed-method downstream attack
    evaluation.
  - Output:
    `eval_results/downstream_meta_checkpoint_sweep/final/attack_eval_summary.csv`
  - Current final-checkpoint summary from that CSV:
    - mean `our_acc`: 0.8959
    - worst `our_acc`: 0.7480
    - mean `our_auc`: 0.9545
    - worst `our_auc`: 0.8389
  - Do not assume `scheduler_residual_seed0_steps2000` is the proposed-method
    row unless we explicitly confirm it is the same training/evaluation
    protocol as the notebook result.

## Main Benchmark Priority

The immediate benchmark priority is the scheduler comparison:

> Under the same SpiderMark verifier, same meta-learning update, same support
> and query construction, same attack pool, same training budget, and same
> downstream evaluation suite, which task scheduler gives the strongest
> downstream watermark verifier?

The SOTA/canonical meta-learning algorithm table remains useful context, but it
should not interrupt the current scheduler-baseline training queue.

Primary scheduler table rows:

- `uniform`: sanity anchor, already trained.
- `bandit_ucb`: running.
- `ats`: next training job.
- `bass`: training job after ATS.
- `asr`: ASr-style adaptive sampler, training job after BASS.
- `ours`: already evaluated in
  `eval_results/downstream_meta_checkpoint_sweep/final/attack_eval_summary.csv`.

## Immediate Execution Order

Training-first policy:

1. Let `scheduler_bandit_ucb_seed0_steps2000` finish.
2. Train `scheduler_ats_seed0_steps2000`.
3. Train `scheduler_bass_seed0_steps2000`.
4. Train `scheduler_asr_seed0_steps2000`.
5. Optional appendix row: train `scheduler_gcp_proxy_seed0_steps2000`.
6. After these training jobs finish, run all scheduler evals together:
   `uniform`, `bandit_ucb`, `ats`, `bass`, and `asr`.
7. Finalize scheduler outputs once after the eval batch.
8. Combine those rows with the already-evaluated notebook `ours` row for the
   paper scheduler table.

Commands after the currently running `bandit_ucb` job finishes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bass_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_asr_seed0_steps2000
```

Optional appendix row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_gcp_proxy_seed0_steps2000
```

Then evaluate all completed scheduler checkpoints:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_uniform_seed0_steps2000 -BatchSize 8 -TestingTimes 5
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -BatchSize 8 -TestingTimes 5
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_ats_seed0_steps2000 -BatchSize 8 -TestingTimes 5
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_bass_seed0_steps2000 -BatchSize 8 -TestingTimes 5
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_asr_seed0_steps2000 -BatchSize 8 -TestingTimes 5
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
```

If `gcp_proxy` is trained, evaluate it before finalizing:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_gcp_proxy_seed0_steps2000 -BatchSize 8 -TestingTimes 5
```

Status check:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_goal_status.ps1
```

Do not use the older mixed queue for the current training-first scheduler run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1 -Execute
```

That wrapper may evaluate anchors or run meta-learning context rows before the
scheduler training queue is complete. For the current scheduler benchmark, use
the explicit training commands above instead.

## Scheduler Baselines To Run

Formal seed-0 scheduler rows:

- [x] `uniform`
  - Current run: `scheduler_uniform_seed0_steps2000`
  - Status: final checkpoint exists; eval is pending.

- [ ] `bandit_ucb`
  - Current run: `scheduler_bandit_ucb_seed0_steps2000`
  - Status: currently training.

- [ ] `ats`
  - Status: train after `bandit_ucb` finishes.

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_ats_seed0_steps2000
  ```

- [ ] `bass`
  - Status: train after `ats` finishes.

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bass_seed0_steps2000
  ```

- [ ] `asr`
  - Status: train after `bass` finishes.
  - Implementation: ASr-style local adaptation from the Adaptive-Sampler repo's
    diversity/entropy/difficulty sampler idea.

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_asr_seed0_steps2000
  ```

- [ ] `gcp_proxy`
  - Status: optional appendix row.
  - Implementation: GCP-style local exponential task-weight update adapted from
    the GCP repo's class-weight update rule.

  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_gcp_proxy_seed0_steps2000
  ```

- [x] `ours`
  - Status: already evaluated in notebook output.
  - Source CSV:
    `eval_results/downstream_meta_checkpoint_sweep/final/attack_eval_summary.csv`

Not in the main compute path:

- `cycle`
- `hard_task`
- `progress`
- `residual`, unless confirmed to match the proposed method
- DERTS-style weighted task subset selection, until implemented
- `derts_proxy`, unless we specifically want a local exploratory appendix row

Do not run these unless the adaptive scheduler benchmark is already complete
and we specifically need an appendix sanity check. DERTS is a recent
task-selection baseline. The runnable `derts_proxy` only approximates the
selection idea with online scalar feedback.

## SOTA Meta-Learning Algorithm Context

This is a required context table for the paper, not another uniform/cycle
benchmark. Use it to show whether MetaSpiderMark is competitive with common
SOTA/canonical meta-learning families under a fixed adaptive scheduler. These runs are
fixed to the `ats` scheduler and seed 0 until the scheduler ranking identifies
a better fixed scheduler.

Current priority: run these SOTA/canonical meta-learning rows before spending
more compute on additional scheduler seeds. The only higher-priority scheduler
work is preserving or quick-looking an already-running checkpoint.

Readiness note:

- `papers/meta_learning/SOTA_META_LEARNING_READINESS.md`
- Only rows in `stage2_meta_learning_benchmark/meta_learning_runs.csv` are
  formal SOTA context rows.
- Legacy `meta_*_uniform_*` directories under the output folder are ignored
  unless they appear in the manifest.

Generate the manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_manifest.ps1
```

Check current meta-learning context queue:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_completion_gate.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1
```

The completion gate should fail until all seven formal SOTA rows are trained,
evaluated, and aggregated into non-placeholder paper tables.

Dry-run smoke check for implemented meta-learning algorithms:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_learning_algorithms.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_meta_algorithm_units.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_smoke_derts_proxy_scheduler.ps1
```

Use these wrappers instead of assuming `pytest` is installed in the conda
environment.

Verified:

- [x] `scripts\stage2_smoke_meta_algorithm_units.ps1` passed for the implemented
  SOTA/canonical meta-learning algorithms.
- [x] `scripts\stage2_smoke_meta_learning_algorithms.ps1` passed in dry-run
  mode for all seven SOTA/canonical meta-learning algorithms with the ATS
  scheduler.
- [x] `scripts\stage2_sota_meta_learning_readiness.ps1` reports seven formal
  ATS/seed-0 rows and ignores legacy `meta_*_uniform_*` directories.
- [ ] Full SOTA/canonical rows still need training and evaluation.

Run one meta-learning context row end-to-end:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute
```

This full context runner also refuses to start while another Stage 2 Python job
is active unless `-AllowConcurrent` is passed intentionally.

Run all formal SOTA/canonical rows sequentially after the GPU is free:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1 -Execute
```

The all-row wrapper finalizes outputs and runs the completion gate after the
seven rows are trained and evaluated.

Low-cost meta-learning algorithm pilot before full 2000-step context rows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_pilot.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_pilot_completion_gate.ps1
```

The pilot fixes the scheduler to `ats`, uses seed 0, and runs 50 steps for
`fomaml`, `maml`, `anil`, `reptile`, `matching_net`, `proto_net`, and `r2d2_ridge`. It is a
debugging/ranking pass, not a replacement for the full matched-compute context
rows.

Low-cost seed 0 rows:

- [ ] `meta_fomaml_ats_seed0_steps2000`
  - Role: default MetaSpiderMark update and delta anchor.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_fomaml_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_fomaml_ats_seed0_steps2000
  ```

- [ ] `meta_maml_ats_seed0_steps2000`
  - Role: second-order MAML baseline.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_maml_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_maml_ats_seed0_steps2000
  ```

- [ ] `meta_anil_ats_seed0_steps2000`
  - Role: ANIL-style head-only adaptation baseline.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_anil_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_anil_ats_seed0_steps2000
  ```

- [ ] `meta_reptile_ats_seed0_steps2000`
  - Role: Reptile-style optimization-based meta-learning baseline.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_reptile_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_reptile_ats_seed0_steps2000
  ```

- [ ] `meta_matching_net_ats_seed0_steps2000`
  - Role: Matching Networks-style attention metric baseline.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_matching_net_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_matching_net_ats_seed0_steps2000
  ```

- [ ] `meta_proto_net_ats_seed0_steps2000`
  - Role: Prototypical Networks-style metric few-shot baseline.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_proto_net_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_proto_net_ats_seed0_steps2000
  ```

- [ ] `meta_r2d2_ridge_ats_seed0_steps2000`
  - Role: R2D2-style differentiable ridge solver-head baseline.
  - Boundary: this is not a full MetaOptNet SVM/QP reproduction.
  ```powershell
  powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_r2d2_ridge_ats_seed0_steps2000
  powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_r2d2_ridge_ats_seed0_steps2000
  ```

Aggregate:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
```

## After Scheduler Training Queue

Do this only after `bandit_ucb`, `ats`, and `bass` seed-0 training are complete.

1. Confirm all selected scheduler checkpoints exist:
   ```powershell
   Test-Path papers\meta_learning\benchmark_outputs\stage2_scheduler_benchmark\scheduler_uniform_seed0_steps2000\checkpoints\final.pth
   Test-Path papers\meta_learning\benchmark_outputs\stage2_scheduler_benchmark\scheduler_bandit_ucb_seed0_steps2000\checkpoints\final.pth
   Test-Path papers\meta_learning\benchmark_outputs\stage2_scheduler_benchmark\scheduler_ats_seed0_steps2000\checkpoints\final.pth
   Test-Path papers\meta_learning\benchmark_outputs\stage2_scheduler_benchmark\scheduler_bass_seed0_steps2000\checkpoints\final.pth
   ```

2. Run downstream attack evaluation for all selected scheduler checkpoints.
   Each output must be:
   ```text
   <run_dir>/attack_eval_summary.csv
   ```
   Commands:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_uniform_seed0_steps2000 -BatchSize 8 -TestingTimes 5
   powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000 -BatchSize 8 -TestingTimes 5
   powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_ats_seed0_steps2000 -BatchSize 8 -TestingTimes 5
   powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_bass_seed0_steps2000 -BatchSize 8 -TestingTimes 5
   ```

3. Aggregate completed scheduler runs once:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
   ```

4. Generate paper tables:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\stage2_make_scheduler_tables.ps1
   ```

5. Check:
   ```text
   papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/summary_by_scheduler.csv
   papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/delta_vs_uniform.csv
   papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_summary.tex
   papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_delta.tex
   ```

## Seed Variance

After seed 0 ranking is known, run seeds 1 and 2 for the top schedulers:

- [ ] `scheduler_bandit_ucb_seed1_steps2000`
- [ ] `scheduler_bandit_ucb_seed2_steps2000`
- [ ] `scheduler_ats_seed1_steps2000`
- [ ] `scheduler_ats_seed2_steps2000`
- [ ] `scheduler_bass_seed1_steps2000`
- [ ] `scheduler_bass_seed2_steps2000`

Do not add residual seeds unless `residual` is confirmed to be the proposed
method and its seed-0 protocol matches the notebook result.

## Required Meta-Learning Algorithm Context

Run this as a required paper comparison against SOTA/canonical meta-learning
methods. Do not wait for large uniform/cycle sweeps:

- [ ] ANIL
- [ ] MAML
- [ ] Reptile
- [ ] Matching Networks-style attention metric baseline
- [ ] Prototypical Networks-style metric baseline
- [ ] R2D2-style ridge solver head
- [ ] MetaOptNet-style SVM/QP solver head, if implemented later

`r2d2_ridge` is implemented as a differentiable ridge solver-head baseline.
Full MetaOptNet-style SVM/QP solver heads are not currently implemented and
should not block the scheduler benchmark.

## Paper Update Checklist

- [x] Frame Stage 2 as SOTA/canonical meta-learning comparison plus scheduler ablation.
- [x] Include ATS-style and BASS-style as strong SOTA-inspired scheduler baselines.
- [x] Keep uniform as a seed-0 sanity anchor, not the main comparison.
- [x] Remove `cycle`, `hard_task`, and `progress` from the main scheduler manifest.
- [ ] Add scheduler results table from `summary_by_scheduler.csv`.
- [ ] Add per-attack deltas from `delta_vs_uniform.csv`.
- [ ] Merge the notebook `ours` row with the scheduler baseline rows.
- [ ] Decide whether the proposed MetaSpiderMark scheduler beats ATS, BASS, and
      UCB under matched evaluation.

