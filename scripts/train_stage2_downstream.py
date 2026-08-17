r"""Downstream-train Stage 2 scheduler checkpoints with one matched protocol.

The Stage 2 checkpoints are meta-verifier initializations. This runner maps each
initialization into the ordinary SpiderMark verifier used by the existing
downstream notebook, then fine-tunes it with the same seed, split, optimizer,
augmentation, and validation-selection settings. Runs are resumable per epoch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from attack import make_train_image_augmentations  # noqa: E402
from ds import WatermarkOnTheFlyDataset, discover_dataset_files  # noqa: E402
from engine import safe_eval_call_psnr_l1, train_epoch_psnr_l1  # noqa: E402
from eval_downstream_meta_checkpoints import load_pipe  # noqa: E402
from model import make_model  # noqa: E402
from watermark import get_watermarking_mask, get_watermarking_pattern  # noqa: E402


DEFAULT_RUN_IDS = [
    "scheduler_uniform_seed0_steps2000",
    "scheduler_bandit_ucb_seed0_steps2000",
    "scheduler_ats_seed0_steps2000",
    "scheduler_bass_seed0_steps2000",
    "scheduler_asr_seed0_steps2000",
]

HISTORY_FIELDS = [
    "epoch",
    "train_loss",
    "val_loss_aug",
    "val_loss_noaug",
    "acc_aug",
    "acc_noaug",
    "auc_aug",
    "auc_noaug",
    "epoch_time_sec",
    "best_val_loss",
    "best_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-csv",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    )
    parser.add_argument("--run-ids", nargs="+", default=DEFAULT_RUN_IDS)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    parser.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--inversion-batch-size", type=int, default=8)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--w-mask-shape", default="circle")
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-strength", type=float, default=0.99)
    parser.add_argument("--w-pattern", default="octoweb")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest_rows(path: Path, run_ids: list[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        indexed = {row["run_id"]: row for row in csv.DictReader(handle)}
    missing = [run_id for run_id in run_ids if run_id not in indexed]
    if missing:
        raise ValueError(f"Run ID(s) missing from manifest: {', '.join(missing)}")
    return [indexed[run_id] for run_id in run_ids]


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], object]:
    if not path.exists():
        raise FileNotFoundError(f"Meta checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )
    if not isinstance(state, dict) or "backbone.conv1.weight" not in state:
        raise ValueError(f"Not a supported MetaVerifier checkpoint: {path}")
    return state, checkpoint


def map_meta_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("backbone."):
            mapped[key.replace("backbone.", "base_model.", 1)] = value
        elif key.startswith("head.fc1."):
            mapped[key.replace("head.fc1.", "fc.0.", 1)] = value
        elif key.startswith("head.fc2."):
            mapped[key.replace("head.fc2.", "fc.2.", 1)] = value
    return mapped


def make_downstream_model(meta_checkpoint: Path, device: str):
    state, source_checkpoint = checkpoint_state(meta_checkpoint)
    in_channels = int(state["backbone.conv1.weight"].shape[1])
    config = source_checkpoint.get("config", {}) if isinstance(source_checkpoint, dict) else {}
    if not bool(config.get("include_psnr_l1", True)):
        raise ValueError(f"Checkpoint does not use the required PSNR/L1 head: {meta_checkpoint}")
    model = make_model(in_channels, include_psnr_l1=True).to(device)
    result = model.load_state_dict(map_meta_state(state), strict=False)
    unexpected = list(result.unexpected_keys)
    missing = [key for key in result.missing_keys if not key.endswith("num_batches_tracked")]
    if missing or unexpected:
        raise RuntimeError(
            f"Meta-to-downstream mapping mismatch for {meta_checkpoint}: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return model, source_checkpoint


def save_checkpoint_atomic(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_history(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_history(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_datasets(args, pipe, text_embeddings, watermarking_mask, gt_patch):
    file_paths, labels = discover_dataset_files(args.data_dir)
    combined = list(zip(file_paths, labels))
    random.Random(args.seed).shuffle(combined)
    file_paths, labels = zip(*combined)
    n_val = int(len(file_paths) * args.validation_split)
    val_paths, val_labels = file_paths[:n_val], labels[:n_val]
    train_paths, train_labels = file_paths[n_val:], labels[n_val:]
    common = dict(
        pipe=pipe,
        text_embeddings=text_embeddings,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=args.device,
        watermarking_mask=watermarking_mask,
        gt_patch=gt_patch,
        image_size=args.image_size,
        include_mask_patch=False,
        include_psnr_l1=True,
        psnr_return_prob=False,
        inversion_batch_size=args.inversion_batch_size,
    )
    train_ds = WatermarkOnTheFlyDataset(
        train_paths,
        train_labels,
        image_aug=make_train_image_augmentations(args.image_size),
        **common,
    )
    val_aug_ds = WatermarkOnTheFlyDataset(
        val_paths,
        val_labels,
        image_aug=make_train_image_augmentations(args.image_size),
        **common,
    )
    val_clean_ds = WatermarkOnTheFlyDataset(
        val_paths,
        val_labels,
        image_aug=None,
        **common,
    )
    return train_ds, val_aug_ds, val_clean_ds


def checkpoint_payload(model, optimizer, epoch, best_val_loss, best_epoch, row, args):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "source_run_id": row["run_id"],
        "source_meta_checkpoint": row["checkpoint_path"],
        "downstream_config": {
            "epochs": args.epochs,
            "seed": args.seed,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "validation_split": args.validation_split,
        },
    }


def train_one(row, args, datasets) -> None:
    run_dir = Path(row["run_dir"])
    output_dir = run_dir / "downstream"
    latest_path = output_dir / "latest.pth"
    best_path = output_dir / "best.pth"
    final_path = output_dir / "final.pth"
    history_path = output_dir / "history.csv"
    model, _ = make_downstream_model(Path(row["checkpoint_path"]), args.device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.base_model.parameters(), "lr": args.backbone_lr},
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    start_epoch, best_val_loss, best_epoch = 1, float("inf"), None
    history = load_history(history_path)

    if final_path.exists():
        final = torch.load(final_path, map_location="cpu", weights_only=False)
        if int(final.get("epoch", 0)) >= args.epochs:
            print(f"[SKIP] {row['run_id']} already completed {args.epochs} epochs")
            return
    if latest_path.exists():
        resume = torch.load(latest_path, map_location=args.device, weights_only=False)
        if resume.get("source_run_id") != row["run_id"]:
            raise RuntimeError(f"Resume checkpoint belongs to another run: {latest_path}")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_val_loss = float(resume.get("best_val_loss", best_val_loss))
        best_epoch = resume.get("best_epoch")
        history = [item for item in history if int(item["epoch"]) < start_epoch]
        print(f"[RESUME] {row['run_id']} from epoch {start_epoch - 1}")

    train_ds, val_aug_ds, val_clean_ds = datasets
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    val_aug_loader = DataLoader(
        val_aug_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_clean_loader = DataLoader(
        val_clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    print(f"[TRAIN] {row['run_id']} epochs {start_epoch}..{args.epochs}")
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_loss = train_epoch_psnr_l1(model, train_loader, optimizer, criterion, args.device)
        acc_aug, auc_aug, val_loss_aug = safe_eval_call_psnr_l1(
            model, val_aug_loader, criterion, args.device
        )
        acc_clean, auc_clean, val_loss_clean = safe_eval_call_psnr_l1(
            model, val_clean_loader, criterion, args.device
        )
        if val_loss_clean < best_val_loss:
            best_val_loss, best_epoch = float(val_loss_clean), epoch
            save_checkpoint_atomic(
                checkpoint_payload(
                    model, optimizer, epoch, best_val_loss, best_epoch, row, args
                ),
                best_path,
            )
        payload = checkpoint_payload(
            model, optimizer, epoch, best_val_loss, best_epoch, row, args
        )
        save_checkpoint_atomic(payload, latest_path)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss_aug": val_loss_aug,
                "val_loss_noaug": val_loss_clean,
                "acc_aug": acc_aug,
                "acc_noaug": acc_clean,
                "auc_aug": auc_aug,
                "auc_noaug": auc_clean,
                "epoch_time_sec": time.time() - started,
                "best_val_loss": best_val_loss,
                "best_epoch": best_epoch,
            }
        )
        write_history(history, history_path)
        print(
            f"[EPOCH {epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss_clean:.4f} acc={acc_clean:.4f} auc={auc_clean:.4f} "
            f"best_epoch={best_epoch}"
        )

    save_checkpoint_atomic(
        checkpoint_payload(
            model, optimizer, args.epochs, best_val_loss, best_epoch, row, args
        ),
        final_path,
    )
    print(f"[DONE] {row['run_id']} downstream checkpoints: {output_dir}")


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    args.device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest_rows(Path(args.manifest_csv), args.run_ids)

    for row in rows:
        model, _ = make_downstream_model(Path(row["checkpoint_path"]), args.device)
        print(f"[READY] {row['run_id']} -> {Path(row['run_dir']) / 'downstream'}")
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if args.dry_run:
        print("[DRY-RUN] all meta-to-downstream checkpoint mappings validated")
        return

    set_seed(args.seed)
    pipe_args = argparse.Namespace(**vars(args))
    pipe, text_embeddings = load_pipe(pipe_args, args.device)
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
    datasets = make_datasets(args, pipe, text_embeddings, watermarking_mask, gt_patch)
    for row in rows:
        set_seed(args.seed)
        train_one(row, args, datasets)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
