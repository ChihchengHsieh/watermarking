"""Evaluate downstream Stage 2 models on one shared nine-attack data stream."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_downstream_meta_checkpoints import (
    add_threshold_accuracies,
    build_attack_factories,
    build_validation_dataset,
    load_pipe,
    set_seed,
)
from model import make_model
from scripts.train_stage2_downstream import DEFAULT_RUN_IDS, load_manifest_rows
from utils.thr import get_best_thrs
from watermark import get_watermarking_mask, get_watermarking_pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-csv",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark/scheduler_runs.csv",
    )
    parser.add_argument("--run-ids", nargs="+", default=DEFAULT_RUN_IDS)
    parser.add_argument("--downstream-dir", default="downstream_shared120")
    parser.add_argument("--checkpoint-name", default="best_auc.pth")
    parser.add_argument(
        "--extra-checkpoints",
        nargs="*",
        default=[],
        metavar="LABEL=PATH",
        help="Additional downstream checkpoints to include in the same shared evaluation.",
    )
    parser.add_argument(
        "--output-dir", default="eval_results/stage2_downstream_shared120_best_auc"
    )
    parser.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    parser.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--testing-times", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def short_label(run_id: str) -> str:
    controlled = re.fullmatch(
        r"controlled_(.+)_seed\d+_steps\d+_s\d+_q\d+", run_id
    )
    if controlled:
        return controlled.group(1)
    label = run_id.removeprefix("scheduler_").removesuffix("_seed0_steps2000")
    return label


def load_states(rows, args):
    states = []

    def append_state(run_id: str, label: str, checkpoint_path: Path) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"Unsafe checkpoint label: {label!r}")
        if any(state["label"] == label for state in states):
            raise ValueError(f"Duplicate checkpoint label: {label}")
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing downstream checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        model = make_model(8, include_psnr_l1=True).to(args.device)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        checkpoint_epoch = int(checkpoint.get("epoch", -1)) if isinstance(checkpoint, dict) else -1
        best_auc_aug = checkpoint.get("best_auc_aug") if isinstance(checkpoint, dict) else None
        states.append(
            {
                "run_id": run_id,
                "label": label,
                "checkpoint": checkpoint_path,
                "checkpoint_epoch": checkpoint_epoch,
                "best_auc_aug": best_auc_aug,
                "model": model,
                "output_dir": Path(args.output_dir) / label,
            }
        )
        print(
            f"[READY] {label}: {checkpoint_path} "
            f"(epoch {checkpoint_epoch}, val AUROC {best_auc_aug})"
        )

    for row in rows:
        checkpoint_path = Path(row["run_dir"]) / args.downstream_dir / args.checkpoint_name
        label = short_label(row["run_id"])
        append_state(row["run_id"], label, checkpoint_path)

    for specification in args.extra_checkpoints:
        if "=" not in specification:
            raise ValueError(
                f"Invalid --extra-checkpoints value {specification!r}; expected LABEL=PATH"
            )
        label, raw_path = specification.split("=", 1)
        append_state(label, label, Path(raw_path))
    return states


def all_attack_artifacts_exist(states, attack_name: str) -> bool:
    return all((state["output_dir"] / attack_name / "eval_results.pt").exists() for state in states)


def load_attack_artifacts(states, attack_name: str):
    return [
        torch.load(
            state["output_dir"] / attack_name / "eval_results.pt",
            map_location="cpu",
            weights_only=False,
        )
        for state in states
    ]


def evaluate_attack_shared(states, loader, dataset, attack_name, aug_builder, args):
    dataset.image_aug = aug_builder()
    dataset.image_aug_prob = 1.0
    all_probs = [[] for _ in states]
    all_gts: list[int] = []
    all_psnrs: list[float] = []
    all_l1s: list[float] = []

    for _ in tqdm(range(args.testing_times), desc=f"{attack_name} | shared runs"):
        for fft, psnr, l1, labels in tqdm(loader, desc="shared eval", leave=False):
            fft = fft.to(args.device)
            psnr_device = psnr.to(args.device)
            l1_device = l1.to(args.device)
            all_gts.extend(labels.tolist())
            all_psnrs.extend(psnr.tolist())
            all_l1s.extend(l1.tolist())
            with torch.no_grad():
                for index, state in enumerate(states):
                    logits = state["model"](fft, psnr_device, l1_device)
                    all_probs[index].extend(
                        torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
                    )

    l1_fpr, l1_tpr, l1_thresholds = metrics.roc_curve(
        all_gts, [-value for value in all_l1s], pos_label=1
    )
    psnr_fpr, psnr_tpr, psnr_thresholds = metrics.roc_curve(
        all_gts, all_psnrs, pos_label=1
    )
    results = []
    for state, probs in zip(states, all_probs):
        our_fpr, our_tpr, our_thresholds = metrics.roc_curve(all_gts, probs, pos_label=1)
        results.append(
            {
                "attack_name": attack_name,
                "preds": probs,
                "gts": all_gts,
                "psnrs": all_psnrs,
                "l1s": all_l1s,
                "best_l1_thr": get_best_thrs(l1_fpr, l1_tpr, l1_thresholds),
                "best_psnr_thr": get_best_thrs(psnr_fpr, psnr_tpr, psnr_thresholds),
                "best_our_thr": get_best_thrs(our_fpr, our_tpr, our_thresholds),
                "l1_auc": metrics.auc(l1_fpr, l1_tpr),
                "psnr_auc": metrics.auc(psnr_fpr, psnr_tpr),
                "our_auc": metrics.auc(our_fpr, our_tpr),
                "checkpoint": str(state["checkpoint"]),
                "checkpoint_label": state["label"],
                "checkpoint_epoch": state["checkpoint_epoch"],
                "best_auc_aug_from_ckpt": state["best_auc_aug"],
                "evaluation_protocol": "shared_downstream_attack_eval_v1",
            }
        )
    return results


def summary_row(result: dict[str, object]) -> dict[str, object]:
    return {
        "checkpoint_label": result["checkpoint_label"],
        "checkpoint": result["checkpoint"],
        "checkpoint_epoch": result["checkpoint_epoch"],
        "best_auc_aug_from_ckpt": result["best_auc_aug_from_ckpt"],
        "attack": result["attack_name"],
        "our_acc": result["our_acc"],
        "our_auc": result["our_auc"],
        "best_our_thr": result["best_our_thr"],
        "l1_acc": result["l1_acc"],
        "l1_auc": result["l1_auc"],
        "best_l1_thr": result["best_l1_thr"],
        "psnr_acc": result["psnr_acc"],
        "psnr_auc": result["psnr_auc"],
        "best_psnr_thr": result["best_psnr_thr"],
        "clean_best_l1_thr": result["clean_best_l1_thr"],
        "clean_best_psnr_thr": result["clean_best_psnr_thr"],
        "elapsed_sec": result.get("elapsed_sec"),
        "evaluation_protocol": result["evaluation_protocol"],
    }


def main() -> None:
    args = parse_args()
    args.device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_manifest_rows(Path(args.manifest_csv), args.run_ids)
    states = load_states(rows, args)
    if args.dry_run:
        print("[DRY-RUN] all downstream checkpoints loaded successfully")
        return

    set_seed(args.seed)
    pipe, text_embeddings = load_pipe(args, args.device)
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
    dataset = build_validation_dataset(
        args, pipe, text_embeddings, watermarking_mask, gt_patch, args.device
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    rows_by_label = {state["label"]: [] for state in states}
    clean_thresholds = None
    for attack_index, (attack_name, aug_builder) in enumerate(
        build_attack_factories(args.image_size).items()
    ):
        started = time.time()
        if not args.force and all_attack_artifacts_exist(states, attack_name):
            results = load_attack_artifacts(states, attack_name)
            print(f"[RESUME] reused completed shared attack: {attack_name}")
        else:
            # Attack-local seeding makes resumed and uninterrupted evaluations identical.
            set_seed(args.seed + attack_index * 1000)
            results = evaluate_attack_shared(
                states, loader, dataset, attack_name, aug_builder, args
            )

        if attack_name == "clean":
            clean_thresholds = (results[0]["best_l1_thr"], results[0]["best_psnr_thr"])
        if clean_thresholds is None:
            raise RuntimeError("Clean must be evaluated before thresholded attacks")

        for state, result in zip(states, results):
            result = add_threshold_accuracies(result, *clean_thresholds)
            result["elapsed_sec"] = time.time() - started
            attack_dir = state["output_dir"] / attack_name
            attack_dir.mkdir(parents=True, exist_ok=True)
            torch.save(result, attack_dir / "eval_results.pt")
            row = summary_row(result)
            rows_by_label[state["label"]].append(row)
            print(
                f"{state['label']:>12} | {attack_name:<14} "
                f"acc={row['our_acc']:.4f} AUROC={row['our_auc']:.4f}"
            )

        for state in states:
            pd.DataFrame(rows_by_label[state["label"]]).to_csv(
                state["output_dir"] / "attack_eval_summary.csv", index=False
            )
        combined = pd.concat(
            [pd.DataFrame(rows_by_label[state["label"]]) for state in states],
            ignore_index=True,
        )
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        combined.to_csv(Path(args.output_dir) / "combined_attack_eval_summary.csv", index=False)

    print(f"[DONE] {Path(args.output_dir) / 'combined_attack_eval_summary.csv'}")


if __name__ == "__main__":
    main()
