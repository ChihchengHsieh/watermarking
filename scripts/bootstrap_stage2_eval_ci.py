"""Paired image-level bootstrap CIs for a shared Stage 2 evaluation.

The evaluator applies every checkpoint and attack to the same ordered examples.
This script resamples those example indices jointly, preserving the pairing across
methods and attacks. By default, clean performance is excluded from the macro
robustness summary but is retained in the per-attack output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument(
        "--include-clean",
        action="store_true",
        help="Include clean performance in the overall macro average.",
    )
    return parser.parse_args()


def load_results(eval_dir: Path):
    loaded: dict[str, dict[str, dict]] = {}
    for path in sorted(eval_dir.glob("*/*/eval_results.pt")):
        result = torch.load(path, map_location="cpu", weights_only=False)
        label = str(result.get("checkpoint_label", path.parents[1].name))
        attack = str(result.get("attack_name", path.parent.name))
        loaded.setdefault(label, {})[attack] = result
    if not loaded:
        raise FileNotFoundError(f"No */*/eval_results.pt files found under {eval_dir}")
    return loaded


def auc_or_nan(labels: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def intervals(values: np.ndarray) -> tuple[float, float]:
    valid = values[~np.isnan(values)]
    if valid.size == 0:
        return float("nan"), float("nan")
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high)


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")

    loaded = load_results(args.eval_dir)
    labels = sorted(loaded)
    all_attacks = sorted(set.intersection(*(set(loaded[label]) for label in labels)))
    if not all_attacks:
        raise RuntimeError("The checkpoints do not share any completed attacks")
    summary_attacks = all_attacks if args.include_clean else [
        attack for attack in all_attacks if attack != "clean"
    ]
    if not summary_attacks:
        raise RuntimeError("No attacks remain in the overall summary")

    reference_gts = np.asarray(loaded[labels[0]][all_attacks[0]]["gts"], dtype=np.int64)
    n_images = reference_gts.size
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for label in labels:
        arrays[label] = {}
        for attack in all_attacks:
            result = loaded[label][attack]
            gts = np.asarray(result["gts"], dtype=np.int64)
            probs = np.asarray(result["preds"], dtype=np.float64)
            if not np.array_equal(gts, reference_gts):
                raise RuntimeError(f"Ground-truth order mismatch for {label}/{attack}")
            if probs.size != n_images:
                raise RuntimeError(f"Prediction length mismatch for {label}/{attack}")
            arrays[label][attack] = probs

    rng = np.random.default_rng(args.seed)
    bootstrap_indices = rng.integers(
        0, n_images, size=(args.bootstrap_samples, n_images), endpoint=False
    )

    overall_rows = []
    attack_rows = []
    macro_correctness_by_label: dict[str, np.ndarray] = {}
    for label in labels:
        point_acc_by_attack = []
        point_auc_by_attack = []
        boot_acc_by_attack: dict[str, np.ndarray] = {}
        boot_auc_by_attack: dict[str, np.ndarray] = {}

        for attack in all_attacks:
            probs = arrays[label][attack]
            point_accuracy = float(np.mean((probs >= 0.5) == reference_gts))
            point_auc = auc_or_nan(reference_gts, probs)
            boot_accuracy = np.empty(args.bootstrap_samples, dtype=np.float64)
            boot_auc = np.empty(args.bootstrap_samples, dtype=np.float64)
            for index, sampled in enumerate(bootstrap_indices):
                sampled_gts = reference_gts[sampled]
                sampled_probs = probs[sampled]
                boot_accuracy[index] = np.mean((sampled_probs >= 0.5) == sampled_gts)
                boot_auc[index] = auc_or_nan(sampled_gts, sampled_probs)
            boot_acc_by_attack[attack] = boot_accuracy
            boot_auc_by_attack[attack] = boot_auc
            acc_low, acc_high = intervals(boot_accuracy)
            auc_low, auc_high = intervals(boot_auc)
            attack_rows.append(
                {
                    "checkpoint_label": label,
                    "attack": attack,
                    "n_images": n_images,
                    "accuracy": point_accuracy,
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "auroc": point_auc,
                    "auroc_ci_low": auc_low,
                    "auroc_ci_high": auc_high,
                }
            )
            if attack in summary_attacks:
                point_acc_by_attack.append(point_accuracy)
                point_auc_by_attack.append(point_auc)

        macro_boot_acc = np.mean(
            np.vstack([boot_acc_by_attack[attack] for attack in summary_attacks]), axis=0
        )
        macro_boot_auc = np.nanmean(
            np.vstack([boot_auc_by_attack[attack] for attack in summary_attacks]), axis=0
        )
        macro_correctness_by_label[label] = np.mean(
            np.vstack(
                [
                    (arrays[label][attack] >= 0.5) == reference_gts
                    for attack in summary_attacks
                ]
            ),
            axis=0,
        )
        acc_low, acc_high = intervals(macro_boot_acc)
        auc_low, auc_high = intervals(macro_boot_auc)
        overall_rows.append(
            {
                "checkpoint_label": label,
                "scope": "all_conditions" if args.include_clean else "attacks_only",
                "n_images": n_images,
                "n_attacks": len(summary_attacks),
                "bootstrap_samples": args.bootstrap_samples,
                "mean_accuracy": float(np.mean(point_acc_by_attack)),
                "accuracy_ci_low": acc_low,
                "accuracy_ci_high": acc_high,
                "mean_auroc": float(np.mean(point_auc_by_attack)),
                "auroc_ci_low": auc_low,
                "auroc_ci_high": auc_high,
            }
        )

    overall = pd.DataFrame(overall_rows).sort_values(
        ["mean_accuracy", "mean_auroc"], ascending=False
    )
    by_attack = pd.DataFrame(attack_rows).sort_values(
        ["attack", "accuracy", "auroc"], ascending=[True, False, False]
    )
    accuracy_leader = str(overall.iloc[0]["checkpoint_label"])
    paired_rows = []
    for label in labels:
        if label == accuracy_leader:
            continue
        differences = macro_correctness_by_label[accuracy_leader] - macro_correctness_by_label[label]
        boot_differences = differences[bootstrap_indices].mean(axis=1)
        difference_low, difference_high = intervals(boot_differences)
        paired_rows.append(
            {
                "accuracy_leader": accuracy_leader,
                "comparison_method": label,
                "mean_accuracy_difference": float(differences.mean()),
                "difference_ci_low": difference_low,
                "difference_ci_high": difference_high,
                "bootstrap_samples": args.bootstrap_samples,
            }
        )
    paired = pd.DataFrame(paired_rows).sort_values(
        "mean_accuracy_difference", ascending=True
    )
    overall_path = args.eval_dir / "bootstrap_ci_summary.csv"
    attacks_path = args.eval_dir / "bootstrap_ci_by_attack.csv"
    paired_path = args.eval_dir / "bootstrap_accuracy_paired_differences.csv"
    overall.to_csv(overall_path, index=False)
    by_attack.to_csv(attacks_path, index=False)
    paired.to_csv(paired_path, index=False)
    print(overall.to_string(index=False))
    print(f"[DONE] {overall_path}")
    print(f"[DONE] {attacks_path}")
    print(f"[DONE] {paired_path}")


if __name__ == "__main__":
    main()
