"""Train all Stage 2 downstream baselines from one shared on-the-fly batch stream.

Every generated FFT/PSNR/L1 batch is reused by every model before the next batch
is generated.  This makes the downstream data exactly matched across baselines
and avoids repeating the expensive Stable Diffusion inversion five times.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_downstream_meta_checkpoints import load_pipe  # noqa: E402
from scripts.train_stage2_downstream import (  # noqa: E402
    DEFAULT_RUN_IDS,
    load_manifest_rows,
    make_datasets,
    make_downstream_model,
    save_checkpoint_atomic,
    set_seed,
)
from watermark import get_watermarking_mask, get_watermarking_pattern  # noqa: E402


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
    "best_auc_aug",
    "best_epoch",
    "best_acc_aug",
    "best_acc_auc_aug",
    "best_acc_epoch",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-csv",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    )
    parser.add_argument("--run-ids", nargs="+", default=DEFAULT_RUN_IDS)
    parser.add_argument("--epochs", type=int, default=120)
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
    parser.add_argument("--output-name", default="downstream_shared120")
    parser.add_argument("--snapshot-epochs", nargs="+", type=int, default=[110, 116, 120])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def write_history(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_history(path: Path, before_epoch: int) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [row for row in csv.DictReader(handle) if int(row["epoch"]) < before_epoch]


def optimizer_for(model, args):
    return torch.optim.AdamW(
        [
            {"params": model.base_model.parameters(), "lr": args.backbone_lr},
            {"params": model.fc.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )


def model_payload(
    state, epoch: int, args, selection_metric: str = "latest_training_state"
) -> dict[str, object]:
    return {
        "epoch": epoch,
        "model_state_dict": state["model"].state_dict(),
        "optimizer_state_dict": state["optimizer"].state_dict(),
        "best_auc_aug": state["best_auc"],
        "best_epoch": state["best_epoch"],
        "best_acc_aug": state["best_acc"],
        "best_acc_auc_aug": state["best_acc_auc"],
        "best_acc_epoch": state["best_acc_epoch"],
        "source_run_id": state["row"]["run_id"],
        "source_meta_checkpoint": state["row"]["checkpoint_path"],
        "selection_metric": selection_metric,
        "shared_batch_protocol": "shared_on_the_fly_v1",
        "downstream_config": {
            "epochs": args.epochs,
            "seed": args.seed,
            "backbone_lr": args.backbone_lr,
            "head_lr": args.head_lr,
            "weight_decay": args.weight_decay,
            "validation_split": args.validation_split,
            "learning_rate_scheduler": None,
            "inversion_batch_size": args.inversion_batch_size,
        },
    }


def rng_payload(epoch: int, shuffle_generator: torch.Generator, args) -> dict[str, object]:
    payload = {
        "epoch": epoch,
        "run_ids": list(args.run_ids),
        "epochs": args.epochs,
        "seed": args.seed,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "shuffle_generator_state": shuffle_generator.get_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng(payload: dict[str, object], shuffle_generator: torch.Generator) -> None:
    random.setstate(payload["python_rng_state"])
    np.random.set_state(payload["numpy_rng_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    shuffle_generator.set_state(payload["shuffle_generator_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])


def shared_train_epoch(states, loader, criterion, device: str) -> list[float]:
    totals = [0.0] * len(states)
    for state in states:
        state["model"].train()
    for fft, psnr, l1, labels in tqdm(loader, desc="shared train", leave=False):
        fft = fft.to(device)
        psnr = psnr.to(device)
        l1 = l1.to(device)
        labels = labels.to(device)
        for index, state in enumerate(states):
            optimizer = state["optimizer"]
            optimizer.zero_grad(set_to_none=True)
            logits = state["model"](fft, psnr, l1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            totals[index] += float(loss.item()) * fft.size(0)
    return [total / len(loader.dataset) for total in totals]


def shared_evaluate(states, loader, criterion, device: str):
    totals = [0.0] * len(states)
    probabilities = [[] for _ in states]
    truths: list[int] = []
    for state in states:
        state["model"].eval()
    with torch.no_grad():
        for fft, psnr, l1, labels in tqdm(loader, desc="shared eval", leave=False):
            fft = fft.to(device)
            psnr = psnr.to(device)
            l1 = l1.to(device)
            labels = labels.to(device)
            truths.extend(labels.cpu().tolist())
            for index, state in enumerate(states):
                logits = state["model"](fft, psnr, l1)
                totals[index] += float(criterion(logits, labels).item()) * fft.size(0)
                probabilities[index].extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    results = []
    for index, probs in enumerate(probabilities):
        accuracy = accuracy_score(truths, [int(prob > 0.5) for prob in probs])
        try:
            auc = roc_auc_score(truths, probs)
        except ValueError:
            auc = float("nan")
        results.append((float(accuracy), float(auc), totals[index] / len(loader.dataset)))
    return results


def build_states(rows, args):
    states = []
    for row in rows:
        model, _ = make_downstream_model(Path(row["checkpoint_path"]), args.device)
        output_dir = Path(row["run_dir"]) / args.output_name
        states.append(
            {
                "row": row,
                "output_dir": output_dir,
                "model": model,
                "optimizer": optimizer_for(model, args),
                "best_auc": float("-inf"),
                "best_epoch": None,
                "best_acc": float("-inf"),
                "best_acc_auc": float("-inf"),
                "best_acc_epoch": None,
                "history": [],
            }
        )
        print(f"[READY] {row['run_id']} -> {output_dir}")
    return states


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("epochs must be positive")
    args.device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest_rows(Path(args.manifest_csv), args.run_ids)
    states = build_states(rows, args)
    if args.dry_run:
        print(f"[DRY-RUN] all {len(states)} mappings validated; no training data generated")
        return

    set_seed(args.seed)
    pipe, text_embeddings = load_pipe(argparse.Namespace(**vars(args)), args.device)
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
    train_ds, val_aug_ds, val_clean_ds = make_datasets(
        args, pipe, text_embeddings, watermarking_mask, gt_patch
    )
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=shuffle_generator,
    )
    val_aug_loader = DataLoader(
        val_aug_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    val_clean_loader = DataLoader(
        val_clean_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    shared_dir = Path(rows[0]["run_dir"]).parent / args.output_name
    shared_state_path = shared_dir / "shared_state.pth"
    start_epoch = 1
    if shared_state_path.exists():
        shared = torch.load(shared_state_path, map_location="cpu", weights_only=False)
        if shared.get("run_ids") != list(args.run_ids):
            raise RuntimeError(f"Shared resume configuration mismatch: {shared_state_path}")
        completed_epoch = int(shared["epoch"])
        previous_target = int(shared.get("epochs", completed_epoch))
        if args.epochs < completed_epoch:
            raise RuntimeError(
                f"Requested {args.epochs} epochs, but the shared state already completed "
                f"epoch {completed_epoch}: {shared_state_path}"
            )
        if args.epochs > previous_target:
            print(
                f"[EXTEND] shared target {previous_target} -> {args.epochs} epochs "
                f"from completed epoch {completed_epoch}"
            )
        for state in states:
            latest_path = state["output_dir"] / "latest.pth"
            resume = torch.load(latest_path, map_location=args.device, weights_only=False)
            if int(resume["epoch"]) != completed_epoch:
                raise RuntimeError(f"Unaligned resume checkpoint: {latest_path}")
            state["model"].load_state_dict(resume["model_state_dict"], strict=True)
            state["optimizer"].load_state_dict(resume["optimizer_state_dict"])
            state["best_auc"] = float(resume.get("best_auc_aug", float("-inf")))
            state["best_epoch"] = resume.get("best_epoch")
            state["best_acc"] = float(resume.get("best_acc_aug", float("-inf")))
            state["best_acc_auc"] = float(
                resume.get("best_acc_auc_aug", float("-inf"))
            )
            state["best_acc_epoch"] = resume.get("best_acc_epoch")
            state["history"] = read_history(
                state["output_dir"] / "history.csv", completed_epoch + 1
            )
        restore_rng(shared, shuffle_generator)
        start_epoch = completed_epoch + 1
        print(f"[RESUME] all models from completed epoch {completed_epoch}")
    else:
        # Model construction and pipeline loading consume RNG; reset immediately
        # before the first shared epoch so the data stream has a stable origin.
        set_seed(args.seed)

    criterion = nn.CrossEntropyLoss()
    snapshots = set(args.snapshot_epochs)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        train_losses = shared_train_epoch(states, train_loader, criterion, args.device)
        aug_results = shared_evaluate(states, val_aug_loader, criterion, args.device)
        clean_results = shared_evaluate(states, val_clean_loader, criterion, args.device)

        for index, state in enumerate(states):
            acc_aug, auc_aug, loss_aug = aug_results[index]
            acc_clean, auc_clean, loss_clean = clean_results[index]
            if not math.isnan(auc_aug) and auc_aug > state["best_auc"]:
                state["best_auc"] = auc_aug
                state["best_epoch"] = epoch
                save_checkpoint_atomic(
                    model_payload(
                        state, epoch, args, "augmented_validation_auroc"
                    ),
                    state["output_dir"] / "best_auc.pth",
                )
            auc_for_tie_break = auc_aug if not math.isnan(auc_aug) else float("-inf")
            improved_accuracy = acc_aug > state["best_acc"]
            tied_accuracy = math.isclose(
                acc_aug, state["best_acc"], rel_tol=0.0, abs_tol=1e-12
            )
            if improved_accuracy or (
                tied_accuracy and auc_for_tie_break > state["best_acc_auc"]
            ):
                state["best_acc"] = acc_aug
                state["best_acc_auc"] = auc_for_tie_break
                state["best_acc_epoch"] = epoch
                save_checkpoint_atomic(
                    model_payload(
                        state,
                        epoch,
                        args,
                        "augmented_validation_accuracy_then_auroc",
                    ),
                    state["output_dir"] / "best_acc.pth",
                )
            row = {
                "epoch": epoch,
                "train_loss": train_losses[index],
                "val_loss_aug": loss_aug,
                "val_loss_noaug": loss_clean,
                "acc_aug": acc_aug,
                "acc_noaug": acc_clean,
                "auc_aug": auc_aug,
                "auc_noaug": auc_clean,
                "epoch_time_sec": time.time() - started,
                "best_auc_aug": state["best_auc"],
                "best_epoch": state["best_epoch"],
                "best_acc_aug": state["best_acc"],
                "best_acc_auc_aug": state["best_acc_auc"],
                "best_acc_epoch": state["best_acc_epoch"],
            }
            state["history"].append(row)
            write_history(state["history"], state["output_dir"] / "history.csv")
            payload = model_payload(state, epoch, args, "latest_training_state")
            save_checkpoint_atomic(payload, state["output_dir"] / "latest.pth")
            if epoch in snapshots:
                snapshot_payload = model_payload(
                    state, epoch, args, "fixed_epoch_snapshot"
                )
                save_checkpoint_atomic(
                    snapshot_payload, state["output_dir"] / f"epoch{epoch}.pth"
                )
            print(
                f"[{epoch:03d}/{args.epochs}] {state['row']['run_id']} "
                f"loss={train_losses[index]:.4f} val_auc_aug={auc_aug:.4f} "
                f"best_auc={state['best_auc']:.4f}@{state['best_epoch']} "
                f"best_acc={state['best_acc']:.4f}@{state['best_acc_epoch']}"
            )

        save_checkpoint_atomic(rng_payload(epoch, shuffle_generator, args), shared_state_path)
        print(f"[EPOCH DONE] {epoch}/{args.epochs} shared time={(time.time() - started) / 60:.1f} min")

    for state in states:
        save_checkpoint_atomic(
            model_payload(state, args.epochs, args, "final_epoch"),
            state["output_dir"] / "final.pth",
        )
    print("[DONE] shared downstream training complete for all models")


if __name__ == "__main__":
    main()
