import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
from torch.utils.data import DataLoader
from tqdm import tqdm

from attack import (
    make_blur_aug,
    make_clean_aug,
    make_down_up_attack,
    make_geom_aug,
    make_jpeg_aug,
    make_msg_app_combo,
    make_occlusion_block,
    make_random_crop_attack,
    make_train_image_augmentations,
)
from diffusers import DPMSolverMultistepScheduler
from ds import WatermarkOnTheFlyDataset, discover_dataset_files
from engine import eval_model_psnr_l1
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from model import load_checkpoint, make_model
from utils.thr import get_best_thrs
from watermark import get_watermarking_mask, get_watermarking_pattern


DEFAULT_CHECKPOINTS = [
    "verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_epoch116.pth",
    "verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_epoch110.pth",
    "verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_final.pth",
]

LOCAL_CHECKPOINT_DIR = Path("local_checkpoints")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate downstream meta verifier checkpoints under attack transforms."
    )
    parser.add_argument("--data-dir", default="./verifier_dataset_stablediff_octoweb")
    parser.add_argument("--output-dir", default="./eval_results/downstream_meta_checkpoint_sweep")
    parser.add_argument("--checkpoints", nargs="+", default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--model-id", default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--testing-times", type=int, default=5)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--w-mask-shape", default="circle")
    parser.add_argument("--w-channel", type=int, default=0)
    parser.add_argument("--w-radius", type=int, default=10)
    parser.add_argument("--w-strength", type=float, default=0.99)
    parser.add_argument("--w-pattern", default="octoweb")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_label(path):
    stem = Path(path).stem
    prefix = "verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_"
    if stem.startswith(prefix):
        return stem[len(prefix) :]
    return stem


def resolve_checkpoint_path(path):
    checkpoint_path = Path(path)
    if checkpoint_path.exists():
        return checkpoint_path

    local_checkpoint_path = LOCAL_CHECKPOINT_DIR / checkpoint_path.name
    if local_checkpoint_path.exists():
        return local_checkpoint_path

    return checkpoint_path


def build_attack_factories(image_size):
    return {
        "clean": lambda: make_clean_aug(image_size),
        "jpeg_strong": lambda: make_jpeg_aug(image_size, q_low=40, q_high=60),
        "msg_app_combo": lambda: make_msg_app_combo(image_size),
        "down_up": lambda: make_down_up_attack(image_size, downscale_frac=0.5),
        "blur": lambda: make_blur_aug(image_size),
        "random_crop": lambda: make_random_crop_attack(image_size, scale=(0.5, 0.9)),
        "occlusion": lambda: make_occlusion_block(image_size, box_frac=0.25),
        "geom_warp": lambda: make_geom_aug(image_size),
        "train_aug_mix": lambda: make_train_image_augmentations(image_size),
    }


def build_validation_dataset(args, pipe, text_embeddings, watermarking_mask, gt_patch, device):
    file_paths, labels = discover_dataset_files(args.data_dir)
    combined = list(zip(file_paths, labels))
    random.shuffle(combined)
    file_paths, labels = zip(*combined)

    n_val = int(len(file_paths) * args.validation_split)
    val_paths = file_paths[:n_val]
    val_labels = labels[:n_val]
    print(f"Validation samples: {len(val_paths)} / {len(file_paths)}")
    print(f"First validation labels: {list(val_labels[:50])}")

    val_ds = WatermarkOnTheFlyDataset(
        val_paths,
        val_labels,
        pipe=pipe,
        text_embeddings=text_embeddings,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        device=device,
        image_aug=None,
        image_aug_prob=1.0,
        image_size=args.image_size,
        include_mask_patch=False,
        include_psnr_l1=True,
        watermarking_mask=watermarking_mask,
        gt_patch=gt_patch,
        psnr_return_prob=False,
        inversion_batch_size=getattr(args, "inversion_batch_size", 8),
    )
    return val_ds


def load_pipe(args, device):
    scheduler = DPMSolverMultistepScheduler.from_pretrained(
        args.model_id, subfolder="scheduler"
    )
    torch_dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
        verbose=False,
    )
    try:
        import diffusers

        diffusers.utils.logging.disable_progress_bar()
    except Exception:
        pass
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    text_embeddings = pipe.get_text_embedding("")
    return pipe, text_embeddings


def eval_attack(model, loader, dataset, attack_name, aug_builder, args, device):
    dataset.image_aug = aug_builder()
    dataset.image_aug_prob = 1.0

    crit = nn.CrossEntropyLoss()
    all_preds, all_gts, all_psnrs, all_l1s = [], [], [], []

    for _ in tqdm(range(args.testing_times), desc=f"{attack_name} | Testing runs"):
        probs, gts, _, psnrs, l1s = eval_model_psnr_l1(model, loader, crit, device)
        all_preds.extend(probs)
        all_gts.extend(gts)
        all_psnrs.extend(psnrs)
        all_l1s.extend(l1s)

    l1_fpr, l1_tpr, l1_thresholds = metrics.roc_curve(
        all_gts, [-x for x in all_l1s], pos_label=1
    )
    psnr_fpr, psnr_tpr, psnr_thresholds = metrics.roc_curve(
        all_gts, all_psnrs, pos_label=1
    )
    our_fpr, our_tpr, our_thresholds = metrics.roc_curve(
        all_gts, all_preds, pos_label=1
    )

    best_l1_thr = get_best_thrs(l1_fpr, l1_tpr, l1_thresholds)
    best_psnr_thr = get_best_thrs(psnr_fpr, psnr_tpr, psnr_thresholds)
    best_our_thr = get_best_thrs(our_fpr, our_tpr, our_thresholds)

    return {
        "attack_name": attack_name,
        "preds": all_preds,
        "gts": all_gts,
        "psnrs": all_psnrs,
        "l1s": all_l1s,
        "best_l1_thr": best_l1_thr,
        "best_psnr_thr": best_psnr_thr,
        "best_our_thr": best_our_thr,
        "l1_auc": metrics.auc(l1_fpr, l1_tpr),
        "psnr_auc": metrics.auc(psnr_fpr, psnr_tpr),
        "our_auc": metrics.auc(our_fpr, our_tpr),
    }


def add_threshold_accuracies(result, clean_best_l1_thr, clean_best_psnr_thr):
    all_gts = result["gts"]
    all_l1s = result["l1s"]
    all_psnrs = result["psnrs"]
    all_preds = result["preds"]

    l1_preds = [1 if -l1 >= clean_best_l1_thr else 0 for l1 in all_l1s]
    psnr_preds = [1 if psnr >= clean_best_psnr_thr else 0 for psnr in all_psnrs]
    our_preds = [1 if pred >= 0.5 else 0 for pred in all_preds]

    result["clean_best_l1_thr"] = clean_best_l1_thr
    result["clean_best_psnr_thr"] = clean_best_psnr_thr
    result["l1_acc"] = sum(p == gt for p, gt in zip(l1_preds, all_gts)) / len(all_gts)
    result["psnr_acc"] = sum(p == gt for p, gt in zip(psnr_preds, all_gts)) / len(all_gts)
    result["our_acc"] = sum(p == gt for p, gt in zip(our_preds, all_gts)) / len(all_gts)
    return result


def evaluate_checkpoint(checkpoint_path, args, pipe, dataset, loader, device):
    label = checkpoint_label(checkpoint_path)
    result_dir = Path(args.output_dir) / label
    result_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"Evaluating checkpoint: {checkpoint_path}")
    print(f"Output dir: {result_dir}")
    print("=" * 80)

    model = make_model(8, include_psnr_l1=True).to(device)
    model, _, best_val_loss, best_epoch = load_checkpoint(
        model, checkpoint_path, device, opt=None
    )
    model.eval()

    attack_factories = build_attack_factories(args.image_size)
    rows = []
    clean_best_l1_thr = None
    clean_best_psnr_thr = None

    for attack_name, aug_builder in attack_factories.items():
        attack_dir = result_dir / attack_name
        attack_dir.mkdir(parents=True, exist_ok=True)

        started = time.time()
        attack_result = eval_attack(
            model, loader, dataset, attack_name, aug_builder, args, device
        )

        if attack_name == "clean":
            clean_best_l1_thr = attack_result["best_l1_thr"]
            clean_best_psnr_thr = attack_result["best_psnr_thr"]
        if clean_best_l1_thr is None or clean_best_psnr_thr is None:
            raise RuntimeError("Clean attack must run before thresholded attacks.")

        attack_result = add_threshold_accuracies(
            attack_result, clean_best_l1_thr, clean_best_psnr_thr
        )
        attack_result["checkpoint"] = checkpoint_path
        attack_result["checkpoint_label"] = label
        attack_result["best_val_loss_from_ckpt"] = best_val_loss
        attack_result["best_epoch_from_ckpt"] = best_epoch
        attack_result["elapsed_sec"] = time.time() - started

        torch.save(attack_result, attack_dir / "eval_results.pt")

        row = {
            "checkpoint_label": label,
            "checkpoint": checkpoint_path,
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
            "elapsed_sec": attack_result["elapsed_sec"],
        }
        rows.append(row)

        print(
            f"{label:>12} | {attack_name:<14} "
            f"our_acc={row['our_acc']:.4f} our_auc={row['our_auc']:.4f} "
            f"l1_acc={row['l1_acc']:.4f} psnr_acc={row['psnr_acc']:.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = result_dir / "attack_eval_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved checkpoint summary: {csv_path}")
    return df


def main():
    args = parse_args()
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    args.checkpoints = [resolve_checkpoint_path(path) for path in args.checkpoints]
    missing = [str(path) for path in args.checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoint(s): {missing}")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
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
        w_seed=args.seed,
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

    all_dfs = []
    total_start = time.time()
    for checkpoint_path in args.checkpoints:
        all_dfs.append(
            evaluate_checkpoint(checkpoint_path, args, pipe, dataset, loader, device)
        )

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_csv = Path(args.output_dir) / "combined_attack_eval_summary.csv"
    combined.to_csv(combined_csv, index=False)

    print("\n" + "=" * 80)
    print("All checkpoint evaluations complete.")
    print(f"Combined summary: {combined_csv}")
    print(f"Total elapsed minutes: {(time.time() - total_start) / 60:.2f}")
    print("=" * 80)
    print(
        combined.sort_values(["attack", "our_acc"], ascending=[True, False])[
            ["attack", "checkpoint_label", "our_acc", "our_auc", "l1_acc", "psnr_acc"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
