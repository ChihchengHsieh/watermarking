"""Aggregate Stage 1 MetaSpiderMark benchmark CSVs.

This script standardizes existing attack-evaluation summaries into a single
paper-oriented table. It does not run model evaluation or training; it consumes
CSV outputs that already exist in the repository.

Default inputs match the current MetaSpiderMark draft:
  - non-meta SpiderMark verifier results
  - downstream meta checkpoint sweep results

Outputs:
  - normalized_results.csv: one row per method/checkpoint/attack
  - summary_by_method.csv: mean accuracy/AUROC by method
  - delta_vs_baseline.csv: per-attack deltas against the baseline method
  - paper_table_meta_vs_nometa.tex: compact LaTeX table for the paper draft
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


ATTACK_ORDER = [
    "clean",
    "jpeg_strong",
    "msg_app_combo",
    "down_up",
    "blur",
    "random_crop",
    "occlusion",
    "geom_warp",
    "train_aug_mix",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate existing Stage 1 MetaSpiderMark benchmark CSVs."
    )
    parser.add_argument(
        "--nonmeta-csv",
        default="eval_results/our_improved_final/attack_eval_summary.csv",
        help="CSV for the improved non-meta SpiderMark verifier.",
    )
    parser.add_argument(
        "--meta-sweep-csv",
        default="eval_results/downstream_meta_checkpoint_sweep/combined_attack_eval_summary.csv",
        help="CSV for the downstream meta checkpoint sweep.",
    )
    parser.add_argument(
        "--selected-meta-checkpoint",
        default="epoch116",
        help="Meta checkpoint label to use in the main paper comparison table.",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/meta_learning/benchmark_outputs/stage1_current",
        help="Directory for aggregate outputs.",
    )
    parser.add_argument(
        "--baseline-method",
        default="SpiderMark-no-meta",
        help="Method name used as the delta baseline.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def float_value(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of these columns exist with values: {names}")


def normalize_nonmeta(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        attack = row["attack"]
        normalized.append(
            {
                "method": "SpiderMark-no-meta",
                "family": "SpiderMark verifier",
                "meta_learning": "no",
                "scheduler": "none",
                "checkpoint_rule": "existing_final",
                "checkpoint_label": "nonmeta_final",
                "attack": attack,
                "accuracy": float_value(row, "our_acc"),
                "auroc": float_value(row, "our_auc", "our_auroc"),
                "source_csv": "eval_results/our_improved_final/attack_eval_summary.csv",
            }
        )
    return normalized


def normalize_meta_sweep(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    normalized = []
    for row in rows:
        checkpoint_label = row.get("checkpoint_label", "unknown")
        normalized.append(
            {
                "method": f"MetaSpiderMark-{checkpoint_label}",
                "family": "MetaSpiderMark verifier",
                "meta_learning": "yes",
                "scheduler": "llm_residual",
                "checkpoint_rule": checkpoint_label,
                "checkpoint_label": checkpoint_label,
                "attack": row["attack"],
                "accuracy": float_value(row, "our_acc"),
                "auroc": float_value(row, "our_auc", "our_auroc"),
                "source_csv": "eval_results/downstream_meta_checkpoint_sweep/combined_attack_eval_summary.csv",
            }
        )
    return normalized


def attack_sort_key(row: dict[str, object]) -> tuple[int, str]:
    attack = str(row["attack"])
    try:
        idx = ATTACK_ORDER.index(attack)
    except ValueError:
        idx = len(ATTACK_ORDER)
    return idx, attack


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["method"])].append(row)

    summary = []
    for method, method_rows in sorted(groups.items()):
        summary.append(
            {
                "method": method,
                "family": method_rows[0]["family"],
                "meta_learning": method_rows[0]["meta_learning"],
                "scheduler": method_rows[0]["scheduler"],
                "checkpoint_rule": method_rows[0]["checkpoint_rule"],
                "num_attacks": len({row["attack"] for row in method_rows}),
                "mean_accuracy": mean(float(row["accuracy"]) for row in method_rows),
                "mean_auroc": mean(float(row["auroc"]) for row in method_rows),
            }
        )
    return summary


def compute_deltas(rows: list[dict[str, object]], baseline_method: str) -> list[dict[str, object]]:
    baseline = {
        str(row["attack"]): row
        for row in rows
        if str(row["method"]) == baseline_method
    }
    if not baseline:
        raise ValueError(f"Baseline method not found: {baseline_method}")

    deltas = []
    for row in rows:
        method = str(row["method"])
        attack = str(row["attack"])
        if method == baseline_method:
            continue
        base = baseline.get(attack)
        if base is None:
            continue
        deltas.append(
            {
                "method": method,
                "baseline_method": baseline_method,
                "attack": attack,
                "accuracy": float(row["accuracy"]),
                "baseline_accuracy": float(base["accuracy"]),
                "delta_accuracy": float(row["accuracy"]) - float(base["accuracy"]),
                "auroc": float(row["auroc"]),
                "baseline_auroc": float(base["auroc"]),
                "delta_auroc": float(row["auroc"]) - float(base["auroc"]),
            }
        )
    return sorted(deltas, key=lambda row: (str(row["method"]), attack_sort_key(row)))


def selected_comparison(
    rows: list[dict[str, object]], baseline_method: str, selected_meta_checkpoint: str
) -> list[dict[str, object]]:
    selected_method = f"MetaSpiderMark-{selected_meta_checkpoint}"
    wanted = {baseline_method, selected_method}
    return [
        row
        for row in rows
        if str(row["method"]) in wanted and str(row["attack"]) in ATTACK_ORDER
    ]


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_latex_table(
    path: Path,
    rows: list[dict[str, object]],
    baseline_method: str,
    selected_meta_checkpoint: str,
) -> None:
    selected_method = f"MetaSpiderMark-{selected_meta_checkpoint}"
    by_method_attack = {
        (str(row["method"]), str(row["attack"])): row
        for row in rows
    }

    lines = [
        "% Auto-generated by scripts/stage1_aggregate_benchmark.py",
        "\\begin{table*}[t]",
        "\\centering",
        "\\caption{Stage 1 comparison between non-meta SpiderMark and the selected MetaSpiderMark verifier.}",
        "\\label{tab:stage1_generated_meta_vs_nometa}",
        "\\resizebox{\\textwidth}{!}{",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\textbf{Attack} & \\textbf{No-meta Acc} & \\textbf{No-meta AUROC} & \\textbf{Meta Acc} & \\textbf{Meta AUROC} & \\textbf{$\\Delta$ Acc} & \\textbf{$\\Delta$ AUROC} \\\\",
        "\\midrule",
    ]

    acc_deltas = []
    auc_deltas = []
    base_accs = []
    base_aucs = []
    meta_accs = []
    meta_aucs = []
    for attack in ATTACK_ORDER:
        base = by_method_attack[(baseline_method, attack)]
        meta = by_method_attack[(selected_method, attack)]
        escaped_attack = attack.replace("_", "\\_")
        delta_acc = float(meta["accuracy"]) - float(base["accuracy"])
        delta_auc = float(meta["auroc"]) - float(base["auroc"])
        base_accs.append(float(base["accuracy"]))
        base_aucs.append(float(base["auroc"]))
        meta_accs.append(float(meta["accuracy"]))
        meta_aucs.append(float(meta["auroc"]))
        acc_deltas.append(delta_acc)
        auc_deltas.append(delta_auc)
        lines.append(
            f"{escaped_attack} & {fmt(float(base['accuracy']))} & {fmt(float(base['auroc']))} "
            f"& {fmt(float(meta['accuracy']))} & {fmt(float(meta['auroc']))} "
            f"& {delta_acc:+.4f} & {delta_auc:+.4f} \\\\"
        )

    lines.extend(
        [
            "\\midrule",
            f"\\textbf{{Mean}} & \\textbf{{{fmt(mean(base_accs))}}} & \\textbf{{{fmt(mean(base_aucs))}}} "
            f"& \\textbf{{{fmt(mean(meta_accs))}}} & \\textbf{{{fmt(mean(meta_aucs))}}} "
            f"& \\textbf{{{mean(acc_deltas):+.4f}}} & \\textbf{{{mean(auc_deltas):+.4f}}} \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    rows.extend(normalize_nonmeta(read_csv(Path(args.nonmeta_csv))))
    rows.extend(normalize_meta_sweep(read_csv(Path(args.meta_sweep_csv))))
    rows = sorted(rows, key=lambda row: (str(row["method"]), attack_sort_key(row)))

    normalized_fields = [
        "method",
        "family",
        "meta_learning",
        "scheduler",
        "checkpoint_rule",
        "checkpoint_label",
        "attack",
        "accuracy",
        "auroc",
        "source_csv",
    ]
    write_csv(output_dir / "normalized_results.csv", rows, normalized_fields)

    summary_rows = summarize(rows)
    write_csv(
        output_dir / "summary_by_method.csv",
        summary_rows,
        [
            "method",
            "family",
            "meta_learning",
            "scheduler",
            "checkpoint_rule",
            "num_attacks",
            "mean_accuracy",
            "mean_auroc",
        ],
    )

    delta_rows = compute_deltas(rows, args.baseline_method)
    write_csv(
        output_dir / "delta_vs_baseline.csv",
        delta_rows,
        [
            "method",
            "baseline_method",
            "attack",
            "accuracy",
            "baseline_accuracy",
            "delta_accuracy",
            "auroc",
            "baseline_auroc",
            "delta_auroc",
        ],
    )

    table_rows = selected_comparison(
        rows, args.baseline_method, args.selected_meta_checkpoint
    )
    write_latex_table(
        output_dir / "paper_table_meta_vs_nometa.tex",
        table_rows,
        args.baseline_method,
        args.selected_meta_checkpoint,
    )

    print(f"Wrote normalized results to {output_dir / 'normalized_results.csv'}")
    print(f"Wrote method summary to {output_dir / 'summary_by_method.csv'}")
    print(f"Wrote deltas to {output_dir / 'delta_vs_baseline.csv'}")
    print(f"Wrote LaTeX table to {output_dir / 'paper_table_meta_vs_nometa.tex'}")


if __name__ == "__main__":
    main()
