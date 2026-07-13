"""Create LaTeX tables from Stage 2 scheduler benchmark aggregates.

This script consumes outputs from ``stage2_aggregate_scheduler_benchmark.ps1``.
It is intentionally read-only with respect to experiment results: it does not
train or evaluate models.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 2 scheduler LaTeX tables.")
    parser.add_argument(
        "--input-dir",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/meta_learning/benchmark_outputs/stage2_scheduler_benchmark",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fmt(value: str) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "--"


def escape_latex(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
    )


def write_placeholder(path: Path, caption: str, label: str) -> None:
    path.write_text(
        "\n".join(
            [
                "\\begin{table}[t]",
                "\\centering",
                "\\small",
                "\\begin{tabular}{lc}",
                "\\toprule",
                "Status & Value \\\\",
                "\\midrule",
                "Benchmark results & Pending \\\\",
                "\\bottomrule",
                "\\end{tabular}",
                f"\\caption{{{caption}. Results are pending because no evaluated scheduler CSVs are available yet.}}",
                f"\\label{{{label}}}",
                "\\end{table}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_summary_table(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        write_placeholder(
            path,
            "Stage 2 scheduler benchmark summary",
            "tab:stage2_scheduler_summary",
        )
        return

    rows = sorted(rows, key=lambda r: float(r.get("mean_accuracy", "nan")), reverse=True)
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Scheduler & Seeds & Mean Acc. & Mean AUROC & Worst Acc. & Worst AUROC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row.get('scheduler', ''))} & "
            f"{row.get('num_seeds', '--')} & "
            f"{fmt(row.get('mean_accuracy', ''))} & "
            f"{fmt(row.get('mean_auroc', ''))} & "
            f"{fmt(row.get('mean_worst_accuracy', ''))} & "
            f"{fmt(row.get('mean_worst_auroc', ''))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Stage 2 scheduler benchmark summary under matched attack pools, support/query sizes, and meta-training budget. Uniform is a seed-0 anchor; adaptive scheduler rows receive the main seed budget.}",
            "\\label{tab:stage2_scheduler_summary}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_table(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        write_placeholder(
            path,
            "Per-attack scheduler deltas versus uniform",
            "tab:stage2_scheduler_delta",
        )
        return

    priority = {"ats": 0, "bass": 1, "bandit_ucb": 2, "residual": 3}
    rows = sorted(
        rows,
        key=lambda r: (
            priority.get(r.get("scheduler", ""), 99),
            int(r.get("seed", 999)),
            r.get("attack", ""),
        ),
    )
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Scheduler & Attack & Acc. & $\\Delta$ Acc. & AUROC & $\\Delta$ AUROC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row.get('scheduler', ''))} & "
            f"{escape_latex(row.get('attack', ''))} & "
            f"{fmt(row.get('accuracy', ''))} & "
            f"{fmt(row.get('delta_accuracy', ''))} & "
            f"{fmt(row.get('auroc', ''))} & "
            f"{fmt(row.get('delta_auroc', ''))} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\caption{Per-attack scheduler deltas versus the uniform seed-0 anchor.}",
            "\\label{tab:stage2_scheduler_delta}",
            "\\end{table*}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = read_csv(input_dir / "summary_by_scheduler.csv")
    delta_rows = read_csv(input_dir / "delta_vs_uniform.csv")

    summary_path = output_dir / "paper_table_scheduler_summary.tex"
    delta_path = output_dir / "paper_table_scheduler_delta.tex"
    write_summary_table(summary_rows, summary_path)
    write_delta_table(delta_rows, delta_path)

    print(f"Wrote scheduler summary table to {summary_path}")
    print(f"Wrote scheduler delta table to {delta_path}")


if __name__ == "__main__":
    main()
