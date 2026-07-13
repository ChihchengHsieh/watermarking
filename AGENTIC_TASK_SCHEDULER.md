# Agentic Task Scheduler POC

This proof of concept keeps the existing SpiderMark task scheduler as the base
policy and adds a residual controller on top:

```python
final_weights = normalize(base_weights + delta_weights)
```

The implementation is intentionally opt-in. Existing `task_sampling="uniform"`
and `task_sampling="cycle"` behavior is unchanged.

## Enable Local Residual Controller

```python
meta_ds = WatermarkMetaTaskDataset(
    ds=train_ds,
    tasks=tasks,
    n_support=META_N_SUPPORT,
    n_query=META_N_QUERY,
    tasks_per_epoch=META_TASKS_PER_EPOCH,
    seed=SEED,
    task_sampling="residual",
    residual_base_sampling="uniform",
    residual_config={
        "residual_scale": 0.15,
        "lr": 0.05,
        "ema_beta": 0.85,
    },
)
```

## Enable LLM / ChatGPT-like Controller

Set an API key before training. For NVIDIA NIM:

```bash
set NVIDIA_API_KEY=...
```

Then choose the LLM residual scheduler:

```python
meta_ds = WatermarkMetaTaskDataset(
    ds=train_ds,
    tasks=tasks,
    n_support=META_N_SUPPORT,
    n_query=META_N_QUERY,
    tasks_per_epoch=META_TASKS_PER_EPOCH,
    seed=SEED,
    task_sampling="llm_residual",
    residual_base_sampling="uniform",
    residual_config={
        "model": "qwen/qwen3-next-80b-a3b-instruct",
        "api_key_env": "NVIDIA_API_KEY",
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "api_format": "chat_completions",
        "residual_scale": 0.15,
        "call_interval": 5,
        "timeout_sec": 20.0,
        "fallback_on_error": True,
    },
)
```

The LLM receives only compact scheduler state and task-memory statistics. It
returns JSON with one residual correction per task:

```python
{
    "delta_weights": [...],
    "rationale": "..."
}
```

The repository still clips and normalizes the final weights locally, so the LLM
cannot fully replace the base scheduler.

For the current smoke-test notebook configuration (`NUM_ITERS=10`,
`META_ATTACK_TASKS="all"`, currently 7 candidate tasks, and
`META_BATCH_SIZE=3`), this means roughly 30 task decisions. With
`call_interval=5`, the controller makes about 6 LLM calls and reuses cached
residual weights between calls. After API loading, scheduler logs, task
distribution, and loss curve look sane, raise to `NUM_ITERS=2000` and
`call_interval=100` for a first-pass POC.

## Feedback

After each downstream meta-training step, feed the query/outer loss back to the
controller:

```python
outer_loss, task_name, inner_lbl_loss = meta_train_step(...)
meta_ds.update_task_feedback_from_batch(
    task_batch,
    loss=float(outer_loss.detach().cpu().item()),
)
```

Optional signals can also be passed when available:

```python
meta_ds.update_task_feedback_from_batch(
    task_batch,
    loss=query_loss,
    val_gain=val_gain,
    fail_rate=task_fail_rate,
)
```

Inspect controller state:

```python
snapshot = meta_ds.residual_snapshot()
```
