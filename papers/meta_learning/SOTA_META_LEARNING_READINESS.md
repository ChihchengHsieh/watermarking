# SOTA/Canonical Meta-Learning Readiness

This note defines the fixed-scheduler meta-learning comparison for the
MetaSpiderMark paper. It prevents the benchmark from drifting back toward
extra `uniform`, `cycle`, `hard_task`, or `progress` scheduler runs.

## Formal Context Rows

The formal SOTA/canonical meta-learning context is defined only by:

```text
papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark/meta_learning_runs.csv
```

The manifest fixes:

- scheduler: `ats`
- seed: `0`
- training steps: `2000`
- support/query: `16/16`
- attack pool: `clean,downup50,crop,jpeg,blur,msg_app,occlusion`
- evaluation suite: `clean,jpeg_strong,msg_app_combo,down_up,blur,random_crop,occlusion,geom_warp,train_aug_mix`

Only `meta_algorithm` changes across rows.

## Included Baselines

The current required context table includes:

- `fomaml`: default MetaSpiderMark first-order MAML update
- `maml`: second-order MAML baseline
- `anil`: head-only inner adaptation baseline
- `reptile`: first-order optimization-based meta-learning baseline
- `matching_net`: Matching Networks-style attention metric baseline
- `proto_net`: Prototypical Networks-style metric baseline
- `r2d2_ridge`: differentiable ridge solver-head baseline

These baselines cover the main practical families reviewers are likely to ask
about: gradient-based, first-order optimization-based, metric-based, and
closed-form solver-head meta-learning.

## Explicit Non-Goals

The following are not part of the formal SOTA/canonical meta-learning context:

- additional `uniform` meta-learning rows
- `cycle`, `hard_task`, or `progress` scheduler rows
- stale directories named `meta_*_uniform_*` under the benchmark output folder
- full MetaOptNet-style SVM/QP solver heads, unless implemented later

Legacy output directories may remain on disk, but they are ignored unless they
appear in the manifest above.

## Commands

Preview the next required row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1
```

Run the next required row:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_run_next_sota_meta_learning_context.ps1 -Execute
```

The runner refuses to start while another Stage 2 Python job is active unless
`-AllowConcurrent` is passed intentionally.

Check readiness and current status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_benchmark_audit.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_execution_plan.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_readiness.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_sota_meta_learning_completion_gate.ps1
powershell -ExecutionPolicy Bypass -File scripts\stage2_meta_learning_status.ps1
```

The completion gate is expected to fail until all seven formal SOTA
meta-learning rows have checkpoints, evaluation CSVs, and non-placeholder paper
tables.
