# Watermarking

Research code and experiment artifacts for the MetaSpiderMark watermarking and
meta-learning benchmarks.

## Stage 2 scheduler experiments

Help is welcome with the Stage 2 scheduler benchmark. The four paper-facing
adaptive scheduling methods are:

| Paper label | Scheduler ID used by the code | Seed-0 run ID |
|---|---|---|
| ATS-style | `ats` | `scheduler_ats_seed0_steps2000` |
| BASS-style | `bass` | `scheduler_bass_seed0_steps2000` |
| Adaptive Sampler-style | `asr` | `scheduler_asr_seed0_steps2000` |
| GCP-style proxy | `gcp_proxy` | `scheduler_gcp_proxy_seed0_steps2000` |

These are local implementations inspired by the cited scheduling methods, not
official reproductions. `bandit_ucb` is a classical bandit baseline and
`uniform` is the anchor; neither is one of the four paper-facing adaptive rows.

### Running status

Last manually updated: **2026-07-14 (Australia/Brisbane)**.

| Run | Role | Training | Evaluation | Notes |
|---|---|---:|---:|---|
| `scheduler_uniform_seed0_steps2000` | Uniform anchor | Complete | Pending | Final checkpoint exists |
| `scheduler_bandit_ucb_seed0_steps2000` | Classical UCB baseline | Running | Pending | Reported at step 1900/2000; resumable checkpoint exists |
| `scheduler_ats_seed0_steps2000` | ATS-style | Available | Pending | Help wanted |
| `scheduler_bass_seed0_steps2000` | BASS-style | Available | Pending | Help wanted |
| `scheduler_asr_seed0_steps2000` | Adaptive Sampler-style | Available | Pending | Help wanted |
| `scheduler_gcp_proxy_seed0_steps2000` | GCP-style proxy | Available | Pending | Help wanted |

The table is a coordination snapshot, not an automatic job monitor. Before
claiming work, check the current filesystem status:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_scheduler_status.ps1
```

### How to contribute a run

Each full run takes several days on the current GPU, so please coordinate in a
GitHub issue or with the project owner before starting. State the exact run ID
you are claiming. Do not start the same run on two machines.

1. Clone the repository, reproduce the project's working PyTorch environment,
   and activate it. The examples below assume the environment is named
   `pytorch` and that you are in the repository root.
2. Check the status command above and confirm that your selected run has not
   already produced `checkpoints/final.pth`.
3. Preview the selected job with `-DryRun`.
4. Run exactly one claimed run ID. Keep its entire run directory when sharing
   results; it contains checkpoints, logs, and evaluation outputs.

Preview commands:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_ats_seed0_steps2000 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bass_seed0_steps2000 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_asr_seed0_steps2000 -DryRun
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_gcp_proxy_seed0_steps2000 -DryRun
```

Remove `-DryRun` from the one command you have claimed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_ats_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_bass_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_asr_seed0_steps2000
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_scheduler_training.ps1 -RunId scheduler_gcp_proxy_seed0_steps2000
```

Run only one of those commands unless you intentionally have separate GPUs and
have claimed multiple run IDs. Training writes periodic checkpoints beneath:

```text
papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/<run-id>/checkpoints/
```

If a run is interrupted, execute the same command again; the runner can resume
from `checkpoints/latest.pth`. Do not delete or rename a partial run directory.

### Evaluate and return results

After training creates `checkpoints/final.pth`, evaluate the same run ID:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId <run-id>
```

For example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\eval_stage2_scheduler_run.ps1 -RunId scheduler_ats_seed0_steps2000
```

Confirm that the run directory contains both the final checkpoint and
`attack_eval_summary.csv`. Then return the complete directory and report the
GPU model, software environment, run ID, completion time, and any warnings or
interruptions. Checkpoint files may be too large for ordinary Git commits, so
coordinate the transfer method with the project owner rather than committing
large binaries unannounced.

Once returned results are placed in their manifest-defined directories, refresh
the aggregate tables with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\stage2_finalize_scheduler_outputs.ps1
```

For the complete benchmark workflow, seed expansion, and troubleshooting, see
[`papers/meta_learning/STAGE2_RUNBOOK.md`](papers/meta_learning/STAGE2_RUNBOOK.md).
