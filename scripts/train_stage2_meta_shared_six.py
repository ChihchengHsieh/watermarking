"""Matched, resumable meta-training for five scheduler baselines plus MetaSpiderMark.

All runs use one model initialization, one train split, and common image indices.
For each meta-batch slot, schedulers choose attacks independently.  When two or
more schedulers choose the same attack, its expensive attacked diffusion batch
is generated once and reused by those runs.  Different attacks cannot share an
inversion because attacks are applied before diffusion inversion.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ds import WatermarkMetaTaskDataset, WatermarkOnTheFlyDataset, build_meta_attack_tasks, discover_dataset_files
from meta.meta_model import make_meta_verifier
from scripts.run_stage2_scheduler_training import json_safe, meta_train_step, scheduler_snapshot, total_grad_norm
from watermark import get_watermarking_mask, get_watermarking_pattern


METHODS = [
    ("uniform", "uniform"),
    ("bandit_ucb", "bandit_ucb"),
    ("ats", "ats"),
    ("bass", "bass"),
    ("asr", "asr"),
    ("metaspidermark", "llm_residual"),
]
ATTACKS = ["clean", "downup50", "crop", "jpeg", "blur", "msg_app", "occlusion"]
LLM_MODEL = "openai/gpt-oss-120b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", default="papers/meta_learning/benchmark_outputs/stage2_controlled_six")
    p.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    p.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    p.add_argument("--seed", type=int, default=19980802)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--n-support", type=int, default=8)
    p.add_argument("--n-query", type=int, default=8)
    p.add_argument("--meta-batch-size", type=int, default=3)
    p.add_argument("--inner-lr", type=float, default=1e-3)
    p.add_argument("--inner-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--validation-split", type=float, default=0.15)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-inference-steps", type=int, default=50)
    p.add_argument("--inversion-batch-size", type=int, default=8)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--log-interval", type=int, default=20)
    p.add_argument("--w-mask-shape", default="circle")
    p.add_argument("--w-channel", type=int, default=0)
    p.add_argument("--w-radius", type=int, default=10)
    p.add_argument("--w-strength", type=float, default=0.99)
    p.add_argument("--w-pattern", default="octoweb")
    p.add_argument("--cuda-device", default="0")
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Train only the five scheduler baselines and put the original MetaSpiderMark checkpoint in the manifest.",
    )
    p.add_argument(
        "--original-meta-checkpoint",
        default="outputs/nvidia_meta_20260601_152616/verifier_dataset_stablediff_octoweb_verifier_meta_iter_2000_final.pth",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(payload: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run_id(label: str, args: argparse.Namespace) -> str:
    return f"controlled_{label}_seed{args.seed}_steps{args.steps}_s{args.n_support}_q{args.n_query}"


def training_methods(args: argparse.Namespace):
    return METHODS[:-1] if args.baseline_only else METHODS


def write_manifest(args: argparse.Namespace, methods) -> Path:
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "controlled_six_runs.csv"
    fields = ["run_id", "scheduler", "seed", "steps", "n_support", "n_query", "attack_pool", "run_dir", "checkpoint_path"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for label, scheduler in methods:
            rid = run_id(label, args)
            directory = root / rid
            writer.writerow({
                "run_id": rid,
                "scheduler": scheduler,
                "seed": args.seed,
                "steps": args.steps,
                "n_support": args.n_support,
                "n_query": args.n_query,
                "attack_pool": ",".join(ATTACKS),
                "run_dir": str(directory),
                "checkpoint_path": str(directory / "checkpoints" / "final.pth"),
            })
        if args.baseline_only:
            label = "metaspidermark_original"
            rid = run_id(label, args)
            writer.writerow({
                "run_id": rid,
                "scheduler": "llm_residual",
                "seed": args.seed,
                "steps": args.steps,
                "n_support": args.n_support,
                "n_query": args.n_query,
                "attack_pool": ",".join(ATTACKS),
                "run_dir": str(root / rid),
                "checkpoint_path": args.original_meta_checkpoint,
            })
    return path


def experiment_signature(args: argparse.Namespace, methods) -> dict[str, object]:
    return {
        "methods": methods,
        "attacks": ATTACKS,
        "seed": args.seed,
        "steps": args.steps,
        "n_support": args.n_support,
        "n_query": args.n_query,
        "meta_batch_size": args.meta_batch_size,
        "inner_lr": args.inner_lr,
        "inner_steps": args.inner_steps,
        "lr": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "data_dir": args.data_dir,
        "model_id": args.model_id,
        "num_inference_steps": args.num_inference_steps,
        "inversion_batch_size": args.inversion_batch_size,
        "llm_model": LLM_MODEL,
    }


def initial_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in state_dict.items():
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def llm_config() -> dict[str, object]:
    return {
        # The original qwen3-next endpoint returned HTTP 410 after its
        # 2026-07-27 retirement. This available model uses the same NVIDIA API.
        "model": LLM_MODEL,
        "api_key_env": "NVIDIA_API_KEY",
        "api_url": "https://integrate.api.nvidia.com/v1/chat/completions",
        "api_format": "chat_completions",
        "residual_scale": 0.5,
        "call_interval": 300,
        "timeout_sec": 45.0,
        "max_tokens": 1536,
        "reasoning_effort": "low",
        "fallback_on_error": True,
        "log_errors": True,
    }


def pack_task(ds, task, indices: list[int], n_support: int, packer, inversion_batch_size: int) -> dict[str, Any]:
    old_aug, old_prob = ds.image_aug, ds.image_aug_prob
    ds.image_aug, ds.image_aug_prob = task.image_aug, float(task.image_aug_prob)
    try:
        samples = ds.get_batch(indices, batch_size=inversion_batch_size)
    finally:
        ds.image_aug, ds.image_aug_prob = old_aug, old_prob
    return {
        "support": packer(samples[:n_support]),
        "query": packer(samples[n_support:]),
        "task_name": task.name,
    }


def checkpoint_payload(state, args, step: int, tasks, init_digest: str) -> dict[str, object]:
    return {
        "global_step": step,
        "model_state_dict": state["model"].state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "meta": {
            "inner_lr": args.inner_lr,
            "inner_steps": args.inner_steps,
            "meta_batch": args.meta_batch_size,
            "n_support": args.n_support,
            "n_query": args.n_query,
        },
        "config": {
            "num_iters": args.steps,
            "lr": args.lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "seed": args.seed,
            "include_psnr_l1": True,
            "inversion_batch_size": args.inversion_batch_size,
            "shared_protocol": "matched_indices_grouped_by_selected_attack_v1",
            "initial_state_sha256": init_digest,
        },
        "scheduler": {
            "mode": state["scheduler"],
            "task_names": [task.name for task in tasks],
            "snapshot": scheduler_snapshot(state["meta_ds"]),
            "residual_config": llm_config() if state["scheduler"] == "llm_residual" else {},
        },
        "history": {
            "meta_losses": state["losses"],
            "sampled_tasks": state["sampled_tasks"],
        },
    }


def main() -> None:
    args = parse_args()
    if (args.n_support, args.n_query, args.meta_batch_size) != (8, 8, 3):
        raise ValueError("Controlled protocol requires support=8, query=8, meta-batch=3")
    methods = training_methods(args)
    if args.baseline_only and not Path(args.original_meta_checkpoint).exists():
        raise FileNotFoundError(f"Original MetaSpiderMark checkpoint not found: {args.original_meta_checkpoint}")
    manifest = write_manifest(args, methods)
    signature = experiment_signature(args, methods)
    root = Path(args.output_root)
    print(f"[CONFIG] manifest={manifest}")
    print(json.dumps(signature, indent=2))
    if args.dry_run:
        print("[DRY-RUN] six matched mappings written; no model or diffusion work performed")
        return

    if any(scheduler == "llm_residual" for _, scheduler in methods) and not os.environ.get("NVIDIA_API_KEY"):
        try:
            from secret import NVIDIA_API_KEY  # type: ignore
            os.environ["NVIDIA_API_KEY"] = NVIDIA_API_KEY
        except Exception as exc:
            raise RuntimeError("NVIDIA_API_KEY is required for the matched MetaSpiderMark controller") from exc

    device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    import diffusers
    from diffusers import DPMSolverMultistepScheduler
    from inverse_stable_diffusion import InversableStableDiffusionPipeline

    diffusers.utils.logging.disable_progress_bar()
    diffusion_scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder="scheduler")
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id, scheduler=diffusion_scheduler, torch_dtype=torch.float16, verbose=False
    ).to(device)
    pipe.set_progress_bar_config(disable=True)
    text_embeddings = pipe.get_text_embedding("")
    watermarking_mask = get_watermarking_mask(
        pipe.get_random_latents(), w_mask_shape=args.w_mask_shape, w_channel=args.w_channel,
        w_radius=args.w_radius, device=device,
    )
    gt_patch = get_watermarking_pattern(
        pipe, w_seed=args.seed, w_pattern=args.w_pattern, w_radius=args.w_radius,
        device=device, strength=args.w_strength, shape=None,
    )

    paths, labels = discover_dataset_files(args.data_dir)
    combined = list(zip(paths, labels))
    random.shuffle(combined)
    n_val = int(len(combined) * args.validation_split)
    train = combined[n_val:]
    train_paths, train_labels = zip(*train)
    train_ds = WatermarkOnTheFlyDataset(
        train_paths, train_labels, pipe=pipe, text_embeddings=text_embeddings,
        num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
        device=device, image_aug=None, image_aug_prob=0.0,
        watermarking_mask=watermarking_mask, gt_patch=gt_patch,
        include_mask_patch=False, include_psnr_l1=True, psnr_return_prob=False,
        inversion_batch_size=args.inversion_batch_size,
    )
    tasks = build_meta_attack_tasks(args.image_size, task_names=ATTACKS)
    sample = train_ds[0]
    in_ch = int(sample[0].shape[0])

    # Reset after pipeline/sample setup: model initialization is explicit and identical.
    set_seed(args.seed)
    reference = make_meta_verifier(in_ch, include_psnr_l1=True)
    reference_state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    init_digest = initial_hash(reference_state)
    del reference

    states = []
    for label, scheduler in methods:
        model = make_meta_verifier(in_ch, include_psnr_l1=True).to(device)
        model.load_state_dict(reference_state, strict=True)
        optimizer = torch.optim.SGD(
            model.params(), lr=args.lr, momentum=args.momentum,
            weight_decay=args.weight_decay, nesterov=True,
        )
        residual_config = llm_config() if scheduler == "llm_residual" else {}
        meta_ds = WatermarkMetaTaskDataset(
            ds=train_ds, tasks=tasks, n_support=args.n_support, n_query=args.n_query,
            tasks_per_epoch=200, seed=args.seed, task_sampling=scheduler,
            residual_base_sampling="uniform", residual_config=residual_config,
        )
        states.append({
            "label": label, "scheduler": scheduler, "model": model,
            "optimizer": optimizer, "meta_ds": meta_ds, "losses": [], "sampled_tasks": [],
        })

    shared_path = root / "shared_meta_state.pth"
    start_step = 0
    index_rng = random.Random(args.seed ^ 0x5A17)
    if shared_path.exists():
        shared = torch.load(shared_path, map_location="cpu", weights_only=False)
        if shared["signature"] != signature:
            raise RuntimeError(f"Resume configuration mismatch: {shared_path}")
        start_step = int(shared["global_step"])
        for state in states:
            rid = run_id(state["label"], args)
            checkpoint = torch.load(root / rid / "checkpoints" / "latest.pth", map_location=device, weights_only=False)
            if int(checkpoint["global_step"]) != start_step:
                raise RuntimeError(f"Unaligned resume checkpoint for {rid}")
            state["model"].load_state_dict(checkpoint["model_state_dict"], strict=True)
            state["optimizer"].load_state_dict(checkpoint["optimizer_state_dict"])
            state["losses"] = list(checkpoint.get("history", {}).get("meta_losses", []))
            state["sampled_tasks"] = list(checkpoint.get("history", {}).get("sampled_tasks", []))
            saved_scheduler = shared["schedulers"][state["label"]]
            state["meta_ds"].rng.setstate(saved_scheduler["task_rng_state"])
            state["meta_ds"].residual_agent = saved_scheduler["controller"]
        random.setstate(shared["python_rng_state"])
        np.random.set_state(shared["numpy_rng_state"])
        torch.set_rng_state(shared["torch_rng_state"])
        index_rng.setstate(shared["index_rng_state"])
        if torch.cuda.is_available() and "cuda_rng_state_all" in shared:
            torch.cuda.set_rng_state_all(shared["cuda_rng_state_all"])
        print(f"[RESUME] continuing all six models from step {start_step}")
    else:
        set_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    started = time.time()
    for step in range(start_step, args.steps):
        step_started = time.time()
        for state in states:
            state["model"].train()
            state["optimizer"].zero_grad(set_to_none=True)
        step_losses = [0.0] * len(states)
        step_tasks = [[] for _ in states]

        for slot in range(args.meta_batch_size):
            chosen = [state["meta_ds"]._choose_task_id(step * args.meta_batch_size + slot) for state in states]
            total = args.n_support + args.n_query
            indices = index_rng.sample(range(len(train_ds)), total) if total <= len(train_ds) else [index_rng.randrange(len(train_ds)) for _ in range(total)]
            grouped: dict[int, list[int]] = defaultdict(list)
            for state_index, task_id in enumerate(chosen):
                grouped[task_id].append(state_index)
            batches = {
                task_id: pack_task(
                    train_ds, tasks[task_id], indices, args.n_support,
                    states[0]["meta_ds"]._pack, args.inversion_batch_size,
                )
                for task_id in grouped
            }
            for state_index, task_id in enumerate(chosen):
                state = states[state_index]
                batch = dict(batches[task_id])
                batch["task_id"] = task_id
                outer_loss, task_name, _ = meta_train_step(
                    base_model=state["model"],
                    make_new_model=lambda: make_meta_verifier(in_ch, include_psnr_l1=True),
                    task_batch=batch, crit=criterion, device=device, include_psnr_l1=True,
                    inner_lr=args.inner_lr, inner_steps=args.inner_steps, first_order=True,
                )
                (outer_loss / args.meta_batch_size).backward()
                loss_value = float(outer_loss.detach().cpu().item())
                state["meta_ds"].update_task_feedback(task_id, loss=loss_value)
                step_losses[state_index] += loss_value / args.meta_batch_size
                step_tasks[state_index].append(task_name)
            del batches

        for index, state in enumerate(states):
            grad_norm = total_grad_norm(state["model"].params())
            state["optimizer"].step()
            state["losses"].append(step_losses[index])
            state["sampled_tasks"].extend(step_tasks[index])
            state["meta_ds"].update_residual_global_context(
                global_step=step + 1, meta_loss=step_losses[index],
                meta_loss_recent_avg=float(np.mean(state["losses"][-20:])),
                grad_norm=grad_norm, learning_rate=args.lr,
                recent_task_counts=dict(Counter(state["sampled_tasks"][-200:])),
            )

        completed = step + 1
        if completed % args.log_interval == 0 or step == start_step:
            elapsed = time.time() - started
            eta = elapsed / max(1, completed - start_step) * (args.steps - completed)
            summary = " | ".join(
                f"{state['label']}={step_losses[i]:.4f}:{','.join(step_tasks[i])}"
                for i, state in enumerate(states)
            )
            print(f"[{completed:04d}/{args.steps}] dt={time.time()-step_started:.1f}s eta={eta/3600:.1f}h | {summary}")

        if completed % args.save_interval == 0 or completed == args.steps:
            for state in states:
                rid = run_id(state["label"], args)
                payload = checkpoint_payload(state, args, completed, tasks, init_digest)
                atomic_save(payload, root / rid / "checkpoints" / "latest.pth")
                if completed == args.steps:
                    atomic_save(payload, root / rid / "checkpoints" / "final.pth")
            shared = {
                "global_step": completed,
                "signature": signature,
                "initial_state_sha256": init_digest,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "index_rng_state": index_rng.getstate(),
                "schedulers": {
                    state["label"]: {
                        "task_rng_state": state["meta_ds"].rng.getstate(),
                        "controller": state["meta_ds"].residual_agent,
                        "snapshot": json_safe(scheduler_snapshot(state["meta_ds"])),
                    }
                    for state in states
                },
            }
            if torch.cuda.is_available():
                shared["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
            atomic_save(shared, shared_path)
            print(f"[CHECKPOINT] synchronized all {len(states)} trained models at step {completed}")

    print(f"[DONE] matched shared meta-training complete for {len(states)} models")


if __name__ == "__main__":
    main()
