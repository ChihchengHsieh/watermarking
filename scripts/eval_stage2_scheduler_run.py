r"""Evaluate one Stage 2 benchmark run.

This script is the evaluation counterpart of ``run_stage2_scheduler_training.py``.
It reads one row from the scheduler manifest, loads the run checkpoint, evaluates
the MetaSpiderMark verifier on the fixed attack suite, and writes:

    <run_dir>/attack_eval_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_downstream_meta_checkpoints import (  # noqa: E402
    add_threshold_accuracies,
    build_attack_factories,
    build_validation_dataset,
    load_pipe,
)
from engine import eval_model_psnr_l1  # noqa: E402
from meta.meta_model import make_meta_verifier  # noqa: E402
from scripts.run_stage2_scheduler_training import update_adapted_model_from_loss  # noqa: E402
from utils.thr import get_best_thrs  # noqa: E402
from watermark import get_watermarking_mask, get_watermarking_pattern  # noqa: E402


EVALUATION_PROTOCOL = "episodic_adaptation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one Stage 2 benchmark run.")
    parser.add_argument(
        "--manifest-csv",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--checkpoint", default=None, help="Override the manifest checkpoint path.")
    parser.add_argument("--output-csv", default=None, help="Override <run_dir>/attack_eval_summary.csv.")
    parser.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    parser.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--testing-times", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=None, help="Override the manifest seed.")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--w-mask-shape", default="circle")
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-strength", type=float, default=0.99)
    parser.add_argument("--w-pattern", default="octoweb")
    parser.add_argument("--include-psnr-l1", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--adaptation-support-size",
        type=int,
        default=None,
        help="Labeled support samples per evaluation episode (default: checkpoint n_support).",
    )
    parser.add_argument(
        "--adaptation-inner-lr",
        type=float,
        default=None,
        help="Support adaptation learning rate (default: checkpoint inner_lr).",
    )
    parser.add_argument(
        "--adaptation-inner-steps",
        type=int,
        default=None,
        help="Support adaptation steps (default: checkpoint inner_steps).",
    )
    parser.add_argument(
        "--fail-on-degenerate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop after an attack if every verifier score is effectively identical.",
    )
    parser.add_argument(
        "--resume-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse existing per-attack eval_results.pt files when resuming an interrupted run.",
    )
    parser.add_argument(
        "--write-partial",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write <output>.partial.csv after each completed attack.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate manifest/checkpoint loading only.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_manifest_row(manifest_csv: str, run_id: str) -> dict[str, str]:
    path = Path(manifest_csv)
    if not path.exists():
        raise FileNotFoundError(f"Manifest CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run_id:
                return row
    raise ValueError(f"Run ID not found in manifest: {run_id}")


def split_csv_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def build_output_row(
    row: dict[str, str],
    checkpoint_path: Path,
    attack_name: str,
    attack_result: dict[str, object],
    seed: int,
    clean_best_l1_thr: float,
    clean_best_psnr_thr: float,
    ckpt: object,
) -> dict[str, object]:
    return {
        "run_id": row["run_id"],
        "scheduler": row.get("scheduler", ""),
        "meta_algorithm": row.get("meta_algorithm", ""),
        "seed": seed,
        "checkpoint_label": row["run_id"],
        "checkpoint": str(checkpoint_path),
        "attack": attack_name,
        "our_acc": attack_result["our_acc"],
        "our_auc": attack_result["our_auc"],
        "best_our_thr": attack_result["best_our_thr"],
        "l1_acc": attack_result["l1_acc"],
        "l1_auc": attack_result["l1_auc"],
        "best_l1_thr": attack_result["best_l1_thr"],
        "psnr_acc": attack_result["psnr_acc"],
        "psnr_auc": attack_result["psnr_auc"],
        "best_psnr_thr": attack_result["best_psnr_thr"],
        "clean_best_l1_thr": clean_best_l1_thr,
        "clean_best_psnr_thr": clean_best_psnr_thr,
        "elapsed_sec": attack_result.get("elapsed_sec", 0.0),
        "evaluation_protocol": attack_result["evaluation_protocol"],
        "adaptation_support_size": attack_result["adaptation_support_size"],
        "adaptation_inner_lr": attack_result["adaptation_inner_lr"],
        "adaptation_inner_steps": attack_result["adaptation_inner_steps"],
    }


def load_scheduler_checkpoint(checkpoint_path: Path, device: str, include_psnr_l1: bool):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    in_channels = int(config.get("in_channels", 8))
    if "include_psnr_l1" in config:
        include_psnr_l1 = bool(config["include_psnr_l1"])

    model = make_meta_verifier(in_channels, include_psnr_l1=include_psnr_l1).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, ckpt


def resolve_adaptation_config(args: argparse.Namespace, ckpt: object) -> dict[str, object]:
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    support_size = int(
        args.adaptation_support_size
        if args.adaptation_support_size is not None
        else config.get("n_support", 16)
    )
    inner_lr = float(
        args.adaptation_inner_lr
        if args.adaptation_inner_lr is not None
        else config.get("inner_lr", 1e-3)
    )
    inner_steps = int(
        args.adaptation_inner_steps
        if args.adaptation_inner_steps is not None
        else config.get("inner_steps", 1)
    )
    if support_size <= 0:
        raise ValueError("adaptation support size must be positive")
    if inner_lr <= 0:
        raise ValueError("adaptation inner learning rate must be positive")
    if inner_steps <= 0:
        raise ValueError("adaptation inner steps must be positive")
    return {
        "evaluation_protocol": EVALUATION_PROTOCOL,
        "adaptation_support_size": support_size,
        "adaptation_inner_lr": inner_lr,
        "adaptation_inner_steps": inner_steps,
    }


def artifact_matches_protocol(artifact: object, adaptation: dict[str, object]) -> bool:
    if not isinstance(artifact, dict):
        return False
    return all(artifact.get(key) == value for key, value in adaptation.items())


def balanced_support_indices(dataset, support_size: int, rng: random.Random) -> tuple[list[int], list[int]]:
    if support_size >= len(dataset):
        raise ValueError(
            f"adaptation support size ({support_size}) must be smaller than validation size ({len(dataset)})"
        )
    labels = [int(label) for label in dataset.labels]
    by_label: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    if len(by_label) < 2:
        raise ValueError("episodic evaluation requires at least two labels in the validation set")

    support: list[int] = []
    label_groups = list(by_label.values())
    base_count, remainder = divmod(support_size, len(label_groups))
    for group_index, candidates in enumerate(label_groups):
        take = base_count + (1 if group_index < remainder else 0)
        if take > len(candidates):
            raise ValueError("not enough validation samples to build a balanced support set")
        support.extend(rng.sample(candidates, take))
    rng.shuffle(support)
    support_set = set(support)
    query = [index for index in range(len(dataset)) if index not in support_set]
    return support, query


def adapt_model(
    base_model,
    support_batch,
    *,
    device: str,
    include_psnr_l1: bool,
    inner_lr: float,
    inner_steps: int,
    meta_algorithm: str,
):
    adapted = make_meta_verifier(
        base_model.in_channels,
        include_psnr_l1=base_model.include_psnr_l1,
    ).to(device)
    # This mirrors meta_train_step. MetaModule.copy intentionally copies learned
    # leaves while each episode starts with fresh BatchNorm running statistics.
    adapted.copy(base_model, same_var=True)
    adapted.train()

    if include_psnr_l1:
        x, psnr, l1, y = support_batch
        psnr = psnr.to(device, non_blocking=True)
        l1 = l1.to(device, non_blocking=True)
    else:
        x, y = support_batch
        psnr = l1 = None
    x = x.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    criterion = nn.CrossEntropyLoss()
    trainable_prefixes = ("head.",) if meta_algorithm == "anil" else None

    for _ in range(inner_steps):
        logits = adapted(x, psnr, l1) if include_psnr_l1 else adapted(x)
        loss = criterion(logits, y)
        update_adapted_model_from_loss(
            adapted,
            loss,
            inner_lr=inner_lr,
            first_order=True,
            trainable_prefixes=trainable_prefixes,
        )
    adapted.eval()
    return adapted


def eval_attack_with_adaptation(
    base_model,
    dataset,
    attack_name: str,
    aug_builder,
    args: argparse.Namespace,
    device: str,
    adaptation: dict[str, object],
    meta_algorithm: str,
    seed: int,
) -> dict[str, object]:
    dataset.image_aug = aug_builder()
    dataset.image_aug_prob = 1.0
    criterion = nn.CrossEntropyLoss()
    all_preds, all_gts, all_psnrs, all_l1s = [], [], [], []
    attack_seed = seed * 100_003 + sum((i + 1) * ord(ch) for i, ch in enumerate(attack_name))
    rng = random.Random(attack_seed)

    for _ in range(args.testing_times):
        support_indices, query_indices = balanced_support_indices(
            dataset, int(adaptation["adaptation_support_size"]), rng
        )
        support_loader = DataLoader(
            Subset(dataset, support_indices),
            batch_size=len(support_indices),
            shuffle=False,
            num_workers=args.num_workers,
        )
        query_loader = DataLoader(
            Subset(dataset, query_indices),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        adapted = adapt_model(
            base_model,
            next(iter(support_loader)),
            device=device,
            include_psnr_l1=base_model.include_psnr_l1,
            inner_lr=float(adaptation["adaptation_inner_lr"]),
            inner_steps=int(adaptation["adaptation_inner_steps"]),
            meta_algorithm=meta_algorithm,
        )
        probs, gts, _, psnrs, l1s = eval_model_psnr_l1(
            adapted, query_loader, criterion, device
        )
        all_preds.extend(probs)
        all_gts.extend(gts)
        all_psnrs.extend(psnrs)
        all_l1s.extend(l1s)
        del adapted

    l1_fpr, l1_tpr, l1_thresholds = metrics.roc_curve(
        all_gts, [-x for x in all_l1s], pos_label=1
    )
    psnr_fpr, psnr_tpr, psnr_thresholds = metrics.roc_curve(
        all_gts, all_psnrs, pos_label=1
    )
    our_fpr, our_tpr, our_thresholds = metrics.roc_curve(
        all_gts, all_preds, pos_label=1
    )
    result = {
        "attack_name": attack_name,
        "preds": all_preds,
        "gts": all_gts,
        "psnrs": all_psnrs,
        "l1s": all_l1s,
        "best_l1_thr": get_best_thrs(l1_fpr, l1_tpr, l1_thresholds),
        "best_psnr_thr": get_best_thrs(psnr_fpr, psnr_tpr, psnr_thresholds),
        "best_our_thr": get_best_thrs(our_fpr, our_tpr, our_thresholds),
        "l1_auc": metrics.auc(l1_fpr, l1_tpr),
        "psnr_auc": metrics.auc(psnr_fpr, psnr_tpr),
        "our_auc": metrics.auc(our_fpr, our_tpr),
    }
    result.update(adaptation)
    return result


def predictions_are_degenerate(preds: object, tolerance: float = 1e-8) -> bool:
    values = np.asarray(preds, dtype=np.float64)
    return values.size == 0 or not np.isfinite(values).all() or np.ptp(values) <= tolerance


def evaluate_run(args: argparse.Namespace, row: dict[str, str]) -> pd.DataFrame:
    seed = int(args.seed if args.seed is not None else row["seed"])
    set_seed(seed)

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint or row["checkpoint_path"])
    output_csv = Path(args.output_csv or row["eval_csv"])
    result_dir = output_csv.parent
    result_dir.mkdir(parents=True, exist_ok=True)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[INFO] run_id={row['run_id']}")
    print(f"[INFO] scheduler={row.get('scheduler', 'unknown')} seed={seed}")
    if row.get("meta_algorithm"):
        print(f"[INFO] meta_algorithm={row['meta_algorithm']}")
    print(f"[INFO] checkpoint={checkpoint_path}")
    print(f"[INFO] output_csv={output_csv}")

    model, ckpt = load_scheduler_checkpoint(checkpoint_path, device, args.include_psnr_l1)
    args.include_psnr_l1 = model.include_psnr_l1
    adaptation = resolve_adaptation_config(args, ckpt)
    print(
        "[INFO] evaluation_protocol="
        f"{adaptation['evaluation_protocol']} "
        f"support={adaptation['adaptation_support_size']} "
        f"inner_lr={adaptation['adaptation_inner_lr']} "
        f"inner_steps={adaptation['adaptation_inner_steps']}"
    )
    if args.dry_run:
        print("[DRY-RUN] checkpoint and adaptation config loaded; skipping diffusion/evaluation.")
        return pd.DataFrame()

    pipe, text_embeddings = load_pipe(args, device)
    watermarking_mask = get_watermarking_mask(
        pipe.get_random_latents(),
        w_mask_shape=args.w_mask_shape,
        w_channel=args.w_channel,
        w_radius=args.w_radius,
        device=device,
    )
    gt_patch = get_watermarking_pattern(
        pipe,
        w_seed=seed,
        w_pattern=args.w_pattern,
        w_radius=args.w_radius,
        device=device,
        strength=args.w_strength,
        shape=None,
    )

    dataset = build_validation_dataset(
        args, pipe, text_embeddings, watermarking_mask, gt_patch, device
    )

    attack_factories = build_attack_factories(args.image_size)
    requested_attacks = split_csv_list(row.get("eval_attack_suite", "")) or list(attack_factories)
    missing_attacks = [name for name in requested_attacks if name not in attack_factories]
    if missing_attacks:
        raise ValueError(f"Unknown eval attack(s) in manifest: {missing_attacks}")
    if "clean" not in requested_attacks:
        requested_attacks = ["clean", *requested_attacks]

    clean_best_l1_thr = None
    clean_best_psnr_thr = None
    rows: list[dict[str, object]] = []
    partial_csv = output_csv.with_suffix(".partial.csv")

    for attack_name in requested_attacks:
        attack_dir = result_dir / "eval_artifacts" / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)
        attack_result_path = attack_dir / "eval_results.pt"

        attack_result = None
        if args.resume_existing and attack_result_path.exists():
            candidate = torch.load(attack_result_path, map_location="cpu", weights_only=False)
            if artifact_matches_protocol(candidate, adaptation):
                attack_result = candidate
                print(f"[RESUME] loaded compatible attack result: {attack_result_path}")
            else:
                print(f"[STALE] ignoring pre-adaptation/incompatible artifact: {attack_result_path}")
        if attack_result is None:
            started = time.time()
            attack_result = eval_attack_with_adaptation(
                model,
                dataset,
                attack_name,
                attack_factories[attack_name],
                args,
                device,
                adaptation,
                row.get("meta_algorithm", "fomaml") or "fomaml",
                seed,
            )
            attack_result["elapsed_sec"] = time.time() - started

        if args.fail_on_degenerate and predictions_are_degenerate(attack_result["preds"]):
            raise RuntimeError(
                f"Degenerate verifier scores for {attack_name}: all probabilities are identical or invalid. "
                "Stopping before the remaining expensive attacks."
            )

        if attack_name == "clean":
            clean_best_l1_thr = attack_result["best_l1_thr"]
            clean_best_psnr_thr = attack_result["best_psnr_thr"]
        if clean_best_l1_thr is None or clean_best_psnr_thr is None:
            raise RuntimeError("Clean attack must run before thresholded attacks.")

        attack_result = add_threshold_accuracies(
            attack_result, clean_best_l1_thr, clean_best_psnr_thr
        )
        attack_result["checkpoint"] = str(checkpoint_path)
        attack_result["checkpoint_label"] = row["run_id"]
        attack_result["scheduler"] = row.get("scheduler", "")
        attack_result["meta_algorithm"] = row.get("meta_algorithm", "")
        attack_result["seed"] = seed
        attack_result["global_step_from_ckpt"] = ckpt.get("global_step") if isinstance(ckpt, dict) else None
        torch.save(attack_result, attack_result_path)

        out_row = build_output_row(
            row,
            checkpoint_path,
            attack_name,
            attack_result,
            seed,
            clean_best_l1_thr,
            clean_best_psnr_thr,
            ckpt,
        )
        rows.append(out_row)
        if args.write_partial:
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
        method_label = row.get("meta_algorithm") or row.get("scheduler", "method")
        print(
            f"{method_label:<12} | seed={seed} | {attack_name:<14} "
            f"our_acc={out_row['our_acc']:.4f} our_auc={out_row['our_auc']:.4f}"
        )

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()
    print(f"[DONE] wrote {output_csv}")
    return df


def main() -> None:
    args = parse_args()
    row = load_manifest_row(args.manifest_csv, args.run_id)
    evaluate_run(args, row)


if __name__ == "__main__":
    main()
