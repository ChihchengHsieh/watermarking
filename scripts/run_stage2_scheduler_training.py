r"""Run one Stage 2 MetaSpiderMark scheduler-training job.

This is a script version of the meta-pretraining notebook loop. It keeps the
SpiderMark verifier, attack-task pool, support/query sizes, and training budget
fixed while varying either ``task_sampling`` or the meta-learning algorithm.

Typical use:

    C:\Users\chihc\miniconda3\envs\pytorch\python.exe \
        scripts/run_stage2_scheduler_training.py \
        --manifest-csv papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv \
        --run-id scheduler_uniform_seed0_steps2000

Use ``--dry-run`` first to validate paths, dataset loading, task construction,
and scheduler setup without running diffusion/model training.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ds import (
    WatermarkMetaTaskDataset,
    WatermarkOnTheFlyDataset,
    build_meta_attack_tasks,
    discover_dataset_files,
)
from meta.meta_model import make_meta_verifier
from watermark import get_watermarking_mask, get_watermarking_pattern


TASK_ALIASES = {
    "down_up": "downup50",
    "random_crop": "crop",
    "jpeg_strong": "jpeg",
    "msg_app_combo": "msg_app",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Stage 2 scheduler benchmark training job.")
    parser.add_argument("--manifest-csv", default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv")
    parser.add_argument("--run-id", help="Run ID to load from the manifest CSV.")
    parser.add_argument("--row-index", type=int, help="Zero-based manifest row index to run if --run-id is omitted.")
    parser.add_argument("--scheduler", default="uniform", help="Scheduler to use when not loading a manifest row.")
    parser.add_argument(
        "--meta-algorithm",
        default="fomaml",
        choices=["fomaml", "maml", "anil", "reptile", "matching_net", "proto_net", "r2d2_ridge"],
        help="Meta-learning algorithm baseline to run.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--n-support", type=int, default=16)
    parser.add_argument("--n-query", type=int, default=16)
    parser.add_argument("--attack-pool", default="clean,downup50,crop,jpeg,blur,msg_app,occlusion")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    parser.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cuda-device", default="0")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--val-batch-size", type=int, default=24)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--meta-batch-size", type=int, default=3)
    parser.add_argument("--tasks-per-epoch", type=int, default=200)
    parser.add_argument("--inner-lr", type=float, default=1e-3)
    parser.add_argument("--inner-steps", type=int, default=1)
    parser.add_argument("--ridge-lambda", type=float, default=1e-3)
    parser.add_argument("--first-order", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Optional checkpoint path to resume from, usually <run_dir>/checkpoints/latest.pth.",
    )
    parser.add_argument("--include-psnr-l1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-mask-patch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--w-mask-shape", default="circle")
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-strength", type=float, default=0.99)
    parser.add_argument("--w-pattern", default="octoweb")
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--residual-lr", type=float, default=0.05)
    parser.add_argument("--residual-base-sampling", default="uniform")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest_row(args: argparse.Namespace) -> dict[str, str] | None:
    if not args.run_id and args.row_index is None:
        return None
    path = Path(args.manifest_csv)
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.run_id:
        for row in rows:
            if row["run_id"] == args.run_id:
                return row
        raise ValueError(f"Run ID not found in manifest: {args.run_id}")
    if args.row_index is None or args.row_index < 0 or args.row_index >= len(rows):
        raise ValueError(f"Invalid --row-index {args.row_index}; manifest has {len(rows)} rows.")
    return rows[args.row_index]


def apply_manifest_row(args: argparse.Namespace, row: dict[str, str] | None) -> argparse.Namespace:
    if row is None:
        if args.run_dir is None:
            args.run_dir = f"papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/manual_{args.scheduler}_seed{args.seed}_steps{args.steps}"
        return args
    args.scheduler = row["scheduler"]
    if "meta_algorithm" in row and row["meta_algorithm"]:
        args.meta_algorithm = row["meta_algorithm"]
    args.seed = int(row["seed"])
    args.steps = int(row["steps"])
    args.n_support = int(row["n_support"])
    args.n_query = int(row["n_query"])
    args.attack_pool = row["attack_pool"]
    args.run_dir = row["run_dir"]
    return args


def normalize_task_names(text: str) -> list[str]:
    names = [name.strip() for name in text.split(",") if name.strip()]
    return [TASK_ALIASES.get(name, name) for name in names]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_task(task_batch: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    support = task_batch["support"]
    query = task_batch["query"]
    for split in (support, query):
        split["x"] = split["x"].squeeze(0)
        split["y"] = split["y"].squeeze(0)
        if "extra" in split:
            for key, value in list(split["extra"].items()):
                if torch.is_tensor(value):
                    split["extra"][key] = value.squeeze(0)
    task_name = task_batch.get("task_name")
    if isinstance(task_name, (list, tuple)):
        task_name = task_name[0]
    return support, query, str(task_name)


def update_adapted_model(
    model,
    *,
    inner_lr: float,
    first_order: bool,
    trainable_prefixes: tuple[str, ...] | None = None,
) -> None:
    for name, param in list(model.named_params()):
        if trainable_prefixes and not any(name.startswith(prefix) for prefix in trainable_prefixes):
            continue
        if param.requires_grad and not param.is_leaf:
            param.retain_grad()
        grad = param.grad
        if grad is None:
            continue
        if first_order:
            grad = grad.detach()
        updated = param - inner_lr * grad
        if updated.requires_grad:
            updated.retain_grad()
        model.set_param(model, name, updated)


def update_adapted_model_from_loss(
    model,
    loss: torch.Tensor,
    *,
    inner_lr: float,
    first_order: bool,
    trainable_prefixes: tuple[str, ...] | None = None,
) -> None:
    named_params = [
        (name, param)
        for name, param in list(model.named_params())
        if param.requires_grad
        and (not trainable_prefixes or any(name.startswith(prefix) for prefix in trainable_prefixes))
    ]
    if not named_params:
        return

    names, params = zip(*named_params)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=not first_order,
        retain_graph=not first_order,
        allow_unused=True,
    )
    for name, param, grad in zip(names, params, grads):
        if grad is None:
            continue
        if first_order:
            grad = grad.detach()
        updated = param - inner_lr * grad
        if updated.requires_grad:
            updated.retain_grad()
        model.set_param(model, name, updated)


def meta_train_step(
    *,
    base_model,
    make_new_model,
    task_batch: dict[str, Any],
    crit,
    device: str,
    include_psnr_l1: bool,
    inner_lr: float,
    inner_steps: int,
    first_order: bool,
    inner_trainable_prefixes: tuple[str, ...] | None = None,
) -> tuple[torch.Tensor, str, torch.Tensor | None]:
    support, query, task_name = unpack_task(task_batch)
    xs = support["x"].to(device, non_blocking=True)
    ys = support["y"].to(device, non_blocking=True)
    xq = query["x"].to(device, non_blocking=True)
    yq = query["y"].to(device, non_blocking=True)

    if include_psnr_l1:
        psnr_s = support["extra"]["psnr"].to(device, non_blocking=True)
        l1_s = support["extra"]["l1"].to(device, non_blocking=True)
        psnr_q = query["extra"]["psnr"].to(device, non_blocking=True)
        l1_q = query["extra"]["l1"].to(device, non_blocking=True)

    new_model = make_new_model().to(device)
    new_model.copy(base_model, same_var=True)
    new_model.train()

    inner_loss = None
    for _ in range(inner_steps):
        logits_s = new_model(xs, psnr_s, l1_s) if include_psnr_l1 else new_model(xs)
        loss_s = crit(logits_s, ys)
        inner_loss = loss_s.detach()
        update_adapted_model_from_loss(
            new_model,
            loss_s,
            inner_lr=inner_lr,
            first_order=first_order,
            trainable_prefixes=inner_trainable_prefixes,
        )

    logits_q = new_model(xq, psnr_q, l1_q) if include_psnr_l1 else new_model(xq)
    return crit(logits_q, yq), task_name, inner_loss


def reptile_task_step(
    *,
    base_model,
    make_new_model,
    task_batch: dict[str, Any],
    crit,
    device: str,
    include_psnr_l1: bool,
    inner_lr: float,
    inner_steps: int,
) -> tuple[float, str, dict[str, torch.Tensor]]:
    support, query, task_name = unpack_task(task_batch)
    xs = support["x"].to(device, non_blocking=True)
    ys = support["y"].to(device, non_blocking=True)
    xq = query["x"].to(device, non_blocking=True)
    yq = query["y"].to(device, non_blocking=True)

    if include_psnr_l1:
        psnr_s = support["extra"]["psnr"].to(device, non_blocking=True)
        l1_s = support["extra"]["l1"].to(device, non_blocking=True)
        psnr_q = query["extra"]["psnr"].to(device, non_blocking=True)
        l1_q = query["extra"]["l1"].to(device, non_blocking=True)

    adapted = make_new_model().to(device)
    adapted.copy(base_model, same_var=False)
    adapted = adapted.to(device)
    adapted.train()

    for _ in range(inner_steps):
        logits_s = adapted(xs, psnr_s, l1_s) if include_psnr_l1 else adapted(xs)
        loss_s = crit(logits_s, ys)
        loss_s.backward(create_graph=False, retain_graph=False)
        update_adapted_model(adapted, inner_lr=inner_lr, first_order=True)
        adapted.zero_grad(set_to_none=True)

    with torch.no_grad():
        logits_q = adapted(xq, psnr_q, l1_q) if include_psnr_l1 else adapted(xq)
        query_loss = float(crit(logits_q, yq).detach().cpu().item())
        adapted_params = {
            name: param.detach().clone()
            for name, param in adapted.named_params()
        }
    return query_loss, task_name, adapted_params


def solver_features(
    model,
    x: torch.Tensor,
    *,
    include_psnr_l1: bool,
    psnr: torch.Tensor | None = None,
    l1: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return differentiable features for closed-form solver baselines."""
    if hasattr(model, "solver_features"):
        feat = model.solver_features(x)
    elif hasattr(model, "backbone"):
        feat = model.backbone(x)
    elif hasattr(model, "get_features"):
        feat = model.get_features(x)
    else:
        feat = x.flatten(1)
    feat = feat.flatten(1)
    if include_psnr_l1:
        if psnr is None or l1 is None:
            raise ValueError("psnr and l1 are required when include_psnr_l1=True")
        if psnr.dim() == 1:
            psnr = psnr.unsqueeze(1)
        if l1.dim() == 1:
            l1 = l1.unsqueeze(1)
        feat = torch.cat([feat, psnr.to(dtype=feat.dtype), l1.to(dtype=feat.dtype)], dim=1)
    return feat


def ridge_classifier_logits(
    support_feat: torch.Tensor,
    support_y: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    num_classes: int,
    ridge_lambda: float,
) -> torch.Tensor:
    support_feat = support_feat.float()
    query_feat = query_feat.float()
    ones_s = torch.ones(support_feat.size(0), 1, device=support_feat.device, dtype=support_feat.dtype)
    ones_q = torch.ones(query_feat.size(0), 1, device=query_feat.device, dtype=query_feat.dtype)
    xs = torch.cat([support_feat, ones_s], dim=1)
    xq = torch.cat([query_feat, ones_q], dim=1)
    y_onehot = torch.nn.functional.one_hot(support_y.long(), num_classes=num_classes).to(dtype=xs.dtype)
    eye = torch.eye(xs.size(1), device=xs.device, dtype=xs.dtype)
    eye[-1, -1] = 0.0
    system = xs.transpose(0, 1) @ xs + float(ridge_lambda) * eye
    rhs = xs.transpose(0, 1) @ y_onehot
    weights = torch.linalg.solve(system, rhs)
    return xq @ weights


def prototypical_classifier_logits(
    support_feat: torch.Tensor,
    support_y: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    support_feat = support_feat.float()
    query_feat = query_feat.float()
    prototypes = []
    for cls in range(num_classes):
        mask = support_y.long() == cls
        if mask.any():
            prototypes.append(support_feat[mask].mean(dim=0))
        else:
            prototypes.append(torch.zeros(support_feat.size(1), device=support_feat.device, dtype=support_feat.dtype))
    prototypes = torch.stack(prototypes, dim=0)
    return -torch.cdist(query_feat, prototypes, p=2).pow(2)


def matching_classifier_logits(
    support_feat: torch.Tensor,
    support_y: torch.Tensor,
    query_feat: torch.Tensor,
    *,
    num_classes: int,
) -> torch.Tensor:
    support_feat = torch.nn.functional.normalize(support_feat.float(), dim=1)
    query_feat = torch.nn.functional.normalize(query_feat.float(), dim=1)
    attention = torch.softmax(query_feat @ support_feat.transpose(0, 1), dim=1)
    y_onehot = torch.nn.functional.one_hot(support_y.long(), num_classes=num_classes).to(dtype=attention.dtype)
    probs = attention @ y_onehot
    return torch.log(probs.clamp_min(1e-8))


def matching_net_task_step(
    *,
    base_model,
    task_batch: dict[str, Any],
    crit,
    device: str,
    include_psnr_l1: bool,
) -> tuple[torch.Tensor, str]:
    support, query, task_name = unpack_task(task_batch)
    xs = support["x"].to(device, non_blocking=True)
    ys = support["y"].to(device, non_blocking=True)
    xq = query["x"].to(device, non_blocking=True)
    yq = query["y"].to(device, non_blocking=True)

    psnr_s = l1_s = psnr_q = l1_q = None
    if include_psnr_l1:
        psnr_s = support["extra"]["psnr"].to(device, non_blocking=True)
        l1_s = support["extra"]["l1"].to(device, non_blocking=True)
        psnr_q = query["extra"]["psnr"].to(device, non_blocking=True)
        l1_q = query["extra"]["l1"].to(device, non_blocking=True)

    support_feat = solver_features(
        base_model,
        xs,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_s,
        l1=l1_s,
    )
    query_feat = solver_features(
        base_model,
        xq,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_q,
        l1=l1_q,
    )
    logits_q = matching_classifier_logits(
        support_feat,
        ys,
        query_feat,
        num_classes=2,
    )
    return crit(logits_q, yq), task_name


def proto_net_task_step(
    *,
    base_model,
    task_batch: dict[str, Any],
    crit,
    device: str,
    include_psnr_l1: bool,
) -> tuple[torch.Tensor, str]:
    support, query, task_name = unpack_task(task_batch)
    xs = support["x"].to(device, non_blocking=True)
    ys = support["y"].to(device, non_blocking=True)
    xq = query["x"].to(device, non_blocking=True)
    yq = query["y"].to(device, non_blocking=True)

    psnr_s = l1_s = psnr_q = l1_q = None
    if include_psnr_l1:
        psnr_s = support["extra"]["psnr"].to(device, non_blocking=True)
        l1_s = support["extra"]["l1"].to(device, non_blocking=True)
        psnr_q = query["extra"]["psnr"].to(device, non_blocking=True)
        l1_q = query["extra"]["l1"].to(device, non_blocking=True)

    support_feat = solver_features(
        base_model,
        xs,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_s,
        l1=l1_s,
    )
    query_feat = solver_features(
        base_model,
        xq,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_q,
        l1=l1_q,
    )
    logits_q = prototypical_classifier_logits(
        support_feat,
        ys,
        query_feat,
        num_classes=2,
    )
    return crit(logits_q, yq), task_name


def r2d2_ridge_task_step(
    *,
    base_model,
    task_batch: dict[str, Any],
    crit,
    device: str,
    include_psnr_l1: bool,
    ridge_lambda: float,
) -> tuple[torch.Tensor, str]:
    support, query, task_name = unpack_task(task_batch)
    xs = support["x"].to(device, non_blocking=True)
    ys = support["y"].to(device, non_blocking=True)
    xq = query["x"].to(device, non_blocking=True)
    yq = query["y"].to(device, non_blocking=True)

    psnr_s = l1_s = psnr_q = l1_q = None
    if include_psnr_l1:
        psnr_s = support["extra"]["psnr"].to(device, non_blocking=True)
        l1_s = support["extra"]["l1"].to(device, non_blocking=True)
        psnr_q = query["extra"]["psnr"].to(device, non_blocking=True)
        l1_q = query["extra"]["l1"].to(device, non_blocking=True)

    support_feat = solver_features(
        base_model,
        xs,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_s,
        l1=l1_s,
    )
    query_feat = solver_features(
        base_model,
        xq,
        include_psnr_l1=include_psnr_l1,
        psnr=psnr_q,
        l1=l1_q,
    )
    logits_q = ridge_classifier_logits(
        support_feat,
        ys,
        query_feat,
        num_classes=2,
        ridge_lambda=ridge_lambda,
    )
    return crit(logits_q, yq), task_name


def scheduler_snapshot(meta_ds) -> dict[str, Any]:
    snap = meta_ds.residual_snapshot() if hasattr(meta_ds, "residual_snapshot") else None
    return snap or {}


def json_safe(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def trim_timing_log(path: Path, max_step: int) -> None:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    kept = []
    for row in rows:
        try:
            step = int(row.get("global_step", "0"))
        except ValueError:
            continue
        if step <= max_step:
            kept.append(row)
    fields = ["global_step", "step_time_sec", "meta_loss", "grad_norm", "tasks"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in kept:
            writer.writerow({field: row.get(field, "") for field in fields})


def total_grad_norm(params) -> float:
    sq_norm = 0.0
    for p in params:
        if p.grad is not None:
            norm = float(p.grad.detach().data.norm(2).cpu().item())
            sq_norm += norm * norm
    return sq_norm ** 0.5


def main() -> None:
    args = parse_args()
    args = apply_manifest_row(args, load_manifest_row(args))

    if args.device is None:
        args.device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    timing_path = run_dir / "timing.csv"
    scheduler_log_path = run_dir / "scheduler.jsonl"
    if args.resume_from is None:
        candidate_resume = ckpt_dir / "latest.pth"
        if candidate_resume.exists():
            args.resume_from = str(candidate_resume)

    set_seed(args.seed)
    task_names = normalize_task_names(args.attack_pool)

    print("=" * 80)
    print("[STAGE2] Scheduler training job")
    print(f"run_dir={run_dir}")
    print(f"scheduler={args.scheduler} seed={args.seed} steps={args.steps}")
    print(f"meta_algorithm={args.meta_algorithm}")
    print(f"tasks={task_names}")
    print(f"device={args.device}")
    print("=" * 80)

    if args.dry_run:
        tasks = build_meta_attack_tasks(args.image_size, task_names=task_names)
        print(f"[DRY-RUN] built tasks: {[task.name for task in tasks]}")
        print("[DRY-RUN] skipping diffusion/model loading and training.")
        return

    import diffusers
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline

    diffusers.utils.logging.disable_progress_bar()

    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        verbose=False,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(args.device)
    text_embeddings = pipe.get_text_embedding("")

    watermarking_mask = get_watermarking_mask(
        pipe.get_random_latents(),
        w_mask_shape=args.w_mask_shape,
        w_channel=args.w_channel,
        w_radius=args.w_radius,
        device=args.device,
    )
    gt_patch = get_watermarking_pattern(
        pipe,
        w_seed=args.seed,
        w_pattern=args.w_pattern,
        w_radius=args.w_radius,
        device=args.device,
        strength=args.w_strength,
        shape=None,
    )

    file_paths, labels = discover_dataset_files(args.data_dir)
    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)
    n_val = int(len(file_paths) * args.validation_split)
    train_paths = file_paths[n_val:]
    train_labels = labels[n_val:]

    train_ds = WatermarkOnTheFlyDataset(
        train_paths,
        train_labels,
        pipe=pipe,
        text_embeddings=text_embeddings,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        image_aug=None,
        image_aug_prob=0.0,
        watermarking_mask=watermarking_mask,
        gt_patch=gt_patch,
        include_mask_patch=args.include_mask_patch,
        include_psnr_l1=args.include_psnr_l1,
        psnr_return_prob=False,
    )

    tasks = build_meta_attack_tasks(args.image_size, task_names=task_names)
    if args.scheduler in {"residual", "llm_residual"}:
        scheduler_config = {
            "residual_scale": args.residual_scale,
            "lr": args.residual_lr,
        }
    else:
        scheduler_config = {}

    meta_ds = WatermarkMetaTaskDataset(
        ds=train_ds,
        tasks=tasks,
        n_support=args.n_support,
        n_query=args.n_query,
        tasks_per_epoch=args.tasks_per_epoch,
        seed=args.seed,
        task_sampling=args.scheduler,
        residual_base_sampling=args.residual_base_sampling,
        residual_config=scheduler_config,
    )
    meta_loader = DataLoader(meta_ds, batch_size=1, shuffle=False, num_workers=0)
    meta_iter = iter(meta_loader)

    sample = train_ds[0]
    x0 = sample[0] if isinstance(sample, (list, tuple)) else sample["x"]
    in_ch = int(x0.shape[0])
    model = make_meta_verifier(in_ch, include_psnr_l1=args.include_psnr_l1).to(args.device)
    meta_params = model.params()
    optimizer = torch.optim.SGD(
        meta_params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    crit = nn.CrossEntropyLoss()
    start_step = 0
    all_losses: list[float] = []
    sampled_tasks: list[str] = []

    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = int(ckpt.get("global_step", 0))
        history = ckpt.get("history", {})
        all_losses = list(history.get("meta_losses", []))
        sampled_tasks = list(history.get("sampled_tasks", []))
        print(f"[RESUME] loaded {resume_path}")
        print(f"[RESUME] continuing from global_step={start_step}")
        trim_timing_log(timing_path, start_step)

    def make_new_model():
        return make_meta_verifier(in_ch, include_psnr_l1=args.include_psnr_l1)

    if start_step == 0 or not timing_path.exists():
        with timing_path.open("w", encoding="utf-8", newline="") as f:
            f.write("global_step,step_time_sec,meta_loss,grad_norm,tasks\n")
    if start_step == 0 or not scheduler_log_path.exists():
        scheduler_log_path.write_text("", encoding="utf-8")

    start = time.time()
    meta_batch = min(int(args.meta_batch_size), len(tasks))

    for step in range(start_step, args.steps):
        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        meta_loss = 0.0
        task_batch_names: list[str] = []
        reptile_targets: dict[str, list[torch.Tensor]] = {}
        for _ in range(meta_batch):
            try:
                task_batch = next(meta_iter)
            except StopIteration:
                meta_iter = iter(meta_loader)
                task_batch = next(meta_iter)

            if args.meta_algorithm == "reptile":
                query_loss, task_name, adapted_params = reptile_task_step(
                    base_model=model,
                    make_new_model=make_new_model,
                    task_batch=task_batch,
                    crit=crit,
                    device=args.device,
                    include_psnr_l1=args.include_psnr_l1,
                    inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps,
                )
                meta_ds.update_task_feedback_from_batch(task_batch, loss=query_loss)
                meta_loss = meta_loss + query_loss
                for name, param in adapted_params.items():
                    reptile_targets.setdefault(name, []).append(param)
            elif args.meta_algorithm == "r2d2_ridge":
                outer_loss, task_name = r2d2_ridge_task_step(
                    base_model=model,
                    task_batch=task_batch,
                    crit=crit,
                    device=args.device,
                    include_psnr_l1=args.include_psnr_l1,
                    ridge_lambda=args.ridge_lambda,
                )
                meta_ds.update_task_feedback_from_batch(
                    task_batch,
                    loss=float(outer_loss.detach().cpu().item()),
                )
                meta_loss = meta_loss + outer_loss
            elif args.meta_algorithm == "matching_net":
                outer_loss, task_name = matching_net_task_step(
                    base_model=model,
                    task_batch=task_batch,
                    crit=crit,
                    device=args.device,
                    include_psnr_l1=args.include_psnr_l1,
                )
                meta_ds.update_task_feedback_from_batch(
                    task_batch,
                    loss=float(outer_loss.detach().cpu().item()),
                )
                meta_loss = meta_loss + outer_loss
            elif args.meta_algorithm == "proto_net":
                outer_loss, task_name = proto_net_task_step(
                    base_model=model,
                    task_batch=task_batch,
                    crit=crit,
                    device=args.device,
                    include_psnr_l1=args.include_psnr_l1,
                )
                meta_ds.update_task_feedback_from_batch(
                    task_batch,
                    loss=float(outer_loss.detach().cpu().item()),
                )
                meta_loss = meta_loss + outer_loss
            else:
                inner_prefixes = ("head.",) if args.meta_algorithm == "anil" else None
                effective_first_order = False if args.meta_algorithm == "maml" else args.first_order
                outer_loss, task_name, _inner_loss = meta_train_step(
                    base_model=model,
                    make_new_model=make_new_model,
                    task_batch=task_batch,
                    crit=crit,
                    device=args.device,
                    include_psnr_l1=args.include_psnr_l1,
                    inner_lr=args.inner_lr,
                    inner_steps=args.inner_steps,
                    first_order=effective_first_order,
                    inner_trainable_prefixes=inner_prefixes,
                )
                meta_ds.update_task_feedback_from_batch(
                    task_batch,
                    loss=float(outer_loss.detach().cpu().item()),
                )
                meta_loss = meta_loss + outer_loss
            task_batch_names.append(task_name)

        if args.meta_algorithm == "reptile":
            loss_val = float(meta_loss) / float(meta_batch)
            for name, param in model.named_params():
                targets = reptile_targets.get(name)
                if not targets:
                    param.grad = torch.zeros_like(param)
                    continue
                target = torch.stack(targets, dim=0).mean(dim=0).to(param.device)
                param.grad = (param.detach() - target).detach()
        else:
            meta_loss = meta_loss / float(meta_batch)
            meta_loss.backward(create_graph=False, retain_graph=False)
            loss_val = float(meta_loss.detach().cpu().item())
        grad_norm = total_grad_norm(meta_params)
        optimizer.step()

        all_losses.append(loss_val)
        sampled_tasks.extend(task_batch_names)
        dt = time.time() - t0

        if hasattr(meta_ds, "update_residual_global_context"):
            meta_ds.update_residual_global_context(
                global_step=step + 1,
                meta_loss=loss_val,
                meta_loss_recent_avg=float(np.mean(all_losses[-20:])),
                grad_norm=grad_norm,
                recent_task_counts=dict(Counter(sampled_tasks[-200:])),
            )

        with timing_path.open("a", encoding="utf-8", newline="") as f:
            f.write(f"{step + 1},{dt:.4f},{loss_val:.6f},{grad_norm:.6f},{'|'.join(task_batch_names)}\n")

        if (step + 1) % args.log_interval == 0 or step == 0:
            elapsed = time.time() - start
            avg = elapsed / float(step + 1)
            eta = avg * max(0, args.steps - step - 1)
            print(
                f"[{step + 1:5d}/{args.steps}] loss={loss_val:.4f} "
                f"grad={grad_norm:.4f} tasks={task_batch_names} "
                f"dt={dt:.1f}s eta={timedelta(seconds=int(eta))}"
            )
            with scheduler_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_safe({
                    "global_step": step + 1,
                    "scheduler": args.scheduler,
                    "sampled_task_names": task_batch_names,
                    "snapshot": scheduler_snapshot(meta_ds),
                })) + "\n")

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(ckpt_dir / "latest.pth", model, optimizer, args, step + 1, tasks, all_losses, sampled_tasks, meta_ds)

    save_checkpoint(ckpt_dir / "final.pth", model, optimizer, args, args.steps, tasks, all_losses, sampled_tasks, meta_ds)
    print(f"[DONE] wrote final checkpoint to {ckpt_dir / 'final.pth'}")


def save_checkpoint(path: Path, model, optimizer, args, step: int, tasks, losses, sampled_tasks, meta_ds) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "meta": {
                "meta_algorithm": args.meta_algorithm,
                "inner_lr": args.inner_lr,
                "inner_steps": args.inner_steps,
                "ridge_lambda": args.ridge_lambda,
                "meta_batch": args.meta_batch_size,
                "n_support": args.n_support,
                "n_query": args.n_query,
            },
            "config": vars(args),
            "scheduler": {
                "mode": args.scheduler,
                "task_names": [task.name for task in tasks],
                "snapshot": scheduler_snapshot(meta_ds),
            },
            "history": {
                "meta_losses": losses,
                "sampled_tasks": sampled_tasks,
            },
        },
        path,
    )


if __name__ == "__main__":
    main()
