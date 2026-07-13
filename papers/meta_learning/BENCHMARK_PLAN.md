# MetaSpiderMark Benchmark Plan

This note records the experiment matrix for the MetaSpiderMark extension of
SpiderMark.

## Core Question

MetaSpiderMark should primarily answer:

> Under the same SpiderMark verifier architecture, attack task pool,
> scheduler, support/query size, seeds, and compute budget, how does the
> proposed MetaSpiderMark update compare with SOTA/canonical meta-learning
> approaches?

This makes the final benchmark a meta-learning algorithm comparison with a
scheduler ablation. The watermark injection mechanism stays fixed. The verifier
architecture stays fixed. The task scheduler is fixed for the SOTA/canonical
meta-learning table, and then the scheduler is varied separately to explain the
effect of task selection.

Uniform is a seed-0 sanity anchor. The important scheduler baselines are
adaptive task schedulers, especially ATS-style neural task scheduling and BASS
(NeurIPS 2023), plus the current residual scheduler. The current ATS/BASS rows
are SOTA-inspired local implementations, not official reproductions.
Do not spend main compute on large uniform/cycle sweeps.

## Fixed Evaluation Suite

Use the same downstream attacks already used by the checkpoint sweep:

| Attack | Purpose |
|---|---|
| `clean` | no transformation reference |
| `jpeg_strong` | aggressive compression |
| `msg_app_combo` | compound platform-style resampling/recompression |
| `down_up` | resolution downsampling and upsampling |
| `blur` | optical or processing blur |
| `random_crop` | framing and crop robustness |
| `occlusion` | partial content removal |
| `geom_warp` | affine/geometric distortion |
| `train_aug_mix` | mixed augmentation stress test |

Primary metrics:

- fixed-threshold accuracy
- AUROC
- mean accuracy/AUROC across attacks
- worst-attack accuracy/AUROC
- per-attack delta versus uniform scheduler

Secondary metrics:

- threshold stability across attacks
- training time and scheduler overhead
- checkpoint sensitivity
- seed variance
- FID and CLIP similarity for the final selected method

## Stage 1: Anchor Baselines

This stage establishes the fixed anchor numbers.

| ID | Method | Scheduler | Checkpoint Rule | Priority |
|---|---|---|---|---|
| S1-A | improved non-meta SpiderMark verifier | none | best existing/final | required |
| S1-B | current MetaSpiderMark checkpoint | existing residual/LLM run | validation-selected | required |
| S1-C | current MetaSpiderMark checkpoint | existing residual/LLM run | final checkpoint | required |

Decision criterion:

- If meta-learning does not beat non-meta SpiderMark, the scheduler benchmark is
  not enough to support the paper.
- If gains only appear for one checkpoint, checkpoint selection must be a major
  ablation.

## Stage 2: SOTA Meta-Learning Algorithm Benchmark

This is the main paper comparison against SOTA/canonical meta-learning
approaches. Keep the scheduler fixed, preferably to the best known adaptive
scheduler. Until the scheduler ranking is available, use `ats` seed 0 because it
is a strong adaptive scheduler baseline and avoids spending compute on uniform
or cycle scheduling.

Then vary only the meta-learning update.

| ID | Method | Role | Priority | Implementation status |
|---|---|---|---|---|
| S2-A | `fomaml` | current/default first-order MetaSpiderMark update | required anchor | implemented |
| S2-B | `maml` | second-order MAML baseline | required canonical context | implemented |
| S2-C | `anil` | adaptation-head baseline | required canonical context | implemented |
| S2-D | `reptile` | optimization-based meta-learning baseline | required canonical context | implemented |
| S2-E | `matching_net` | Matching Networks-style attention metric baseline | required standard context | implemented |
| S2-F | `proto_net` | Prototypical Networks-style metric baseline | required standard context | implemented |
| S2-G | `r2d2_ridge` | R2D2-style differentiable ridge solver head | required stronger context | implemented |
| S2-H | MetaOptNet-style SVM/QP solver head | stronger external few-shot learner family | optional | not implemented |

Decision criterion:

- If `fomaml` beats or matches the canonical meta-learning rows under matched
  compute, the MetaSpiderMark update is competitive.
- If metric or ridge-style methods dominate, the paper should report that
  algorithm choice is the strongest factor and keep scheduler claims secondary.
- Full MetaOptNet-style solver heads should not block the benchmark unless the
  SVM/QP implementation is added later.

## Stage 3: Scheduler Ablation

This explains the task-selection effect after the meta-learning algorithm
comparison is established.

All rows should use the same attack pool, support/query sizes, meta-learning
algorithm, number of meta-training steps, and checkpoint rule. The sanity
anchor `uniform` only needs seed 0 by default; compute should go to adaptive
and SOTA-inspired scheduler baselines.

| ID | Scheduler | Family | Priority | Implementation note |
|---|---|---|---|---|
| S3-A | `uniform` | minimal sanity anchor | required, one seed first | random task sampling |
| S3-B | `bandit_ucb` | bandit baseline | required | non-contextual UCB over attack tasks |
| S3-C | `ats` | SOTA-inspired scheduler baseline | required | adaptive task scheduler inspired by ATS (NeurIPS 2021), local implementation |
| S3-D | `bass` | SOTA-inspired scheduler baseline | required | contextual-bandit scheduler inspired by BASS (NeurIPS 2023), local implementation |
| S3-E | `residual` | proposed/local controller | required | current residual task controller |
| S3-F | `llm_residual` | diagnostic extension | optional | LLM-assisted residual corrections |
| S3-G | DERTS-style task subset selection | recent task-selection baseline | optional | CVPRW 2024 method; requires task-pool gradient approximation and is not currently implemented |
| S3-H | `derts_proxy` | local exploratory proxy | optional | online scalar-feedback approximation of DERTS-style representative/robust task selection; not an official reproduction |

Decision criterion:

- If `residual` beats `bandit_ucb`, `ats`, and `bass` under matched compute, scheduler
  design is a credible contribution.
- If `ats` or `bass` wins, the paper should position SOTA-inspired task schedulers as the
  strongest method family and describe residual scheduling as an ablation.
- If `uniform` is competitive with all adaptive schedulers, the contribution
  should be framed around episodic meta-training rather than scheduler design.
- If reviewers ask for a more recent task-selection baseline, DERTS-style
  subset selection is the most relevant optional extension. It should be added
  only after the ATS/BASS/UCB/residual comparison is runnable because it needs a
  task-pool subset-selection interface rather than the current online sampler.

## Stage 4: Robustness and Ablations

This stage makes the scheduler benchmark publishable by testing stability,
generalization, and practical cost.

| ID | Ablation | Purpose | Priority |
|---|---|---|---|
| S4-A | seed variance | show rankings are stable | required |
| S4-B | checkpoint rule: final vs validation-selected vs random | separate learning from checkpoint luck | required |
| S4-C | held-out attack or held-out prompt split | test generalization | required |
| S4-D | support/query size | show meta-learning is not tuned to one episode shape | useful |
| S4-E | training cost and scheduler overhead | report practical cost | required |
| S4-F | FID and CLIP for selected final method | verify no quality/semantic regression | useful |

## Current Implementation Status

Available `task_sampling` values:

- `uniform`
- `cycle`
- `hard_task`
- `progress`
- `bandit_ucb`
- `bandit_thompson`
- `ats`
- `bass`
- `residual`
- `llm_residual`
- `derts_proxy`

Relevant but not currently implemented:

- DERTS-style weighted task subset selection

The Stage 2 scheduler manifest includes:

- `uniform` seed 0 only
- `bandit_ucb`
- `ats`
- `bass`
- `residual`

The following implemented schedulers are intentionally excluded from the main
manifest unless an appendix sanity check is needed after the adaptive scheduler
benchmark is complete:

- `cycle`
- `hard_task`
- `progress`

## Commands

Generate and run the short scheduler pilot before spending compute on full
2000-step rows. The pilot is for ranking and debugging only; it does not replace
the final matched-compute benchmark.

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

Generate the Stage 2 scheduler run manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_manifest.ps1
```

Dry-run all scheduler-training jobs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -All -DryRun
```

Check scheduler benchmark status and get the next command:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_goal_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1
```

Run the recommended benchmark queue. This prioritizes the fixed-scheduler
FOMAML/MAML/ANIL/Reptile/MatchingNet/ProtoNet/R2D2-style ridge meta-learning
comparison, then moves to seed-0 ATS/BASS/UCB/residual scheduler ablations.
If the existing uniform seed-0 checkpoint still lacks
evaluation, the queue evaluates it first so scheduler deltas have an anchor; it
does not train additional uniform runs. The queue avoids concurrent Stage 2
Python jobs by default:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_benchmark_goal.ps1 -Execute
```

Run the strongest seed-0 scheduler baselines:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_scheduler.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_priority_seed0.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bandit_ucb_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bass_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_residual_seed0_steps2000
```

The wrapper previews commands by default. Add `-Execute` to run them. It skips
runs that already have `checkpoints/final.pth`; use `-Force` only for an
intentional rerun.

Evaluate completed priority seed-0 scheduler runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_next_scheduler.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_eval_priority_seed0.ps1 -Execute
```

Aggregate completed scheduler runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
```

Generate paper-ready scheduler tables:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_make_scheduler_tables.ps1
```

Generate the SOTA/canonical meta-learning context manifest:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_manifest.ps1
```

Generate and preview the short meta-learning algorithm pilot before spending
compute on full 2000-step context rows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_manifest.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_pilot_status.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_meta_learning_pilot.ps1 -Execute
```

Check SOTA/canonical meta-learning context status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1
```

Run and evaluate one meta-learning algorithm baseline:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_all_sota_meta_learning_context.ps1 -Execute
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_fomaml_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_fomaml_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_maml_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_maml_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_anil_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_anil_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_reptile_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_reptile_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_matching_net_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_matching_net_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_proto_net_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_proto_net_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -ManifestCsv papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv -RunId meta_r2d2_ridge_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_meta_learning_run.ps1 -RunId meta_r2d2_ridge_ats_seed0_steps2000
```

The full context runner refuses to start while another Stage 2 Python job is
active unless `-AllowConcurrent` is passed intentionally.

Aggregate completed meta-learning algorithm runs:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_meta_learning_outputs.ps1
```

Current Stage 2 scheduler outputs:

- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/missing_runs.csv`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/normalized_scheduler_results.csv`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/summary_by_scheduler.csv`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/delta_vs_uniform.csv`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_summary.tex`
- `papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/paper_table_scheduler_delta.tex`

## Not Main-Benchmark Items

These can remain contextual or appendix material:

- external watermarking baselines such as Tree-Ring, DFT-Single, and DWT-DCT
- full MetaOptNet-style SVM/QP solver heads, unless implemented later
- PAC-Bayes meta-learning theory
- LLM residual scheduling as the headline method

