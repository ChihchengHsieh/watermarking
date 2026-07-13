"""Create LaTeX tables from Stage 2 SOTA/canonical meta-learning aggregates."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Stage 2 meta-learning LaTeX tables.")
    parser.add_argument(
        "--input-dir",
        default="papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark",
    )
    parser.add_argument(
        "--output-dir",
        default="papers/meta_learning/benchmark_outputs/stage2_meta_learning_benchmark",
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
                f"\\caption{{{caption}. Results are pending because no evaluated meta-learning CSVs are available yet.}}",
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
            "SOTA/canonical meta-learning algorithm context",
            "tab:stage2_meta_algorithm_summary",
        )
        return

    priority = {"fomaml": 0, "maml": 1, "anil": 2, "reptile": 3, "matching_net": 4, "proto_net": 5, "r2d2_ridge": 6}
    rows = sorted(rows, key=lambda r: priority.get(r.get("meta_algorithm", ""), 99))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "Meta Algorithm & Seeds & Mean Acc. & Mean AUROC & Worst Acc. & Worst AUROC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row.get('meta_algorithm', ''))} & "
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
            "\\caption{SOTA/canonical meta-learning algorithm context under a fixed ATS scheduler and seed-0 setting.}",
            "\\label{tab:stage2_meta_algorithm_summary}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_delta_table(rows: list[dict[str, str]], path: Path) -> None:
    if not rows:
        write_placeholder(
            path,
            "Per-attack meta-learning algorithm deltas versus FOMAML",
            "tab:stage2_meta_algorithm_delta",
        )
        return

    priority = {"maml": 0, "anil": 1, "reptile": 2, "matching_net": 3, "proto_net": 4, "r2d2_ridge": 5}
    rows = sorted(
        rows,
        key=lambda r: (
            priority.get(r.get("meta_algorithm", ""), 99),
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
        "Meta Algorithm & Attack & Acc. & $\\Delta$ Acc. & AUROC & $\\Delta$ AUROC \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{escape_latex(row.get('meta_algorithm', ''))} & "
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
            "\\caption{Per-attack meta-learning algorithm deltas versus the FOMAML anchor under a fixed ATS scheduler.}",
            "\\label{tab:stage2_meta_algorithm_delta}",
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

    summary_rows = read_csv(input_dir / "summary_by_meta_algorithm.csv")
    delta_rows = read_csv(input_dir / "delta_vs_fomaml.csv")

    summary_path = output_dir / "paper_table_meta_algorithm_summary.tex"
    delta_path = output_dir / "paper_table_meta_algorithm_delta.tex"
    write_summary_table(summary_rows, summary_path)
    write_delta_table(delta_rows, delta_path)

    print(f"Wrote meta-learning summary table to {summary_path}")
    print(f"Wrote meta-learning delta table to {delta_path}")


if __name__ == "__main__":
    main()
