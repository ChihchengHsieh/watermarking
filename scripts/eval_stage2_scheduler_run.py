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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_downstream_meta_checkpoints import (  # noqa: E402
    add_threshold_accuracies,
    build_attack_factories,
    build_validation_dataset,
    eval_attack,
    load_pipe,
)
from meta.meta_model import make_meta_verifier  # noqa: E402
from watermark import get_watermarking_mask, get_watermarking_pattern  # noqa: E402


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
    if args.dry_run:
        print("[DRY-RUN] checkpoint loaded successfully; skipping diffusion/evaluation.")
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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

        if args.resume_existing and attack_result_path.exists():
            attack_result = torch.load(attack_result_path, map_location="cpu", weights_only=False)
            print(f"[RESUME] loaded existing attack result: {attack_result_path}")
        else:
            started = time.time()
            attack_result = eval_attack(
                model,
                loader,
                dataset,
                attack_name,
                attack_factories[attack_name],
                args,
                device,
            )
            attack_result["elapsed_sec"] = time.time() - started

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
