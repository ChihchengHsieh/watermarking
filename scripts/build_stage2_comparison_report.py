"""Build the portable Stage 2 downstream comparison report artifact."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd


BASELINE_CSV = Path(
    "eval_results/stage2_downstream_shared120_best_auc/combined_attack_eval_summary.csv"
)
META_CSV = Path("eval_results/downstream_meta_checkpoint_sweep/combined_attack_eval_summary.csv")
DEFAULT_OUTPUT = Path(
    "papers/meta_learning/reports/downstream_comparison_20260811/artifact.json"
)
MODEL_LABELS = {
    "uniform": "Uniform",
    "bandit_ucb": "Bandit-UCB",
    "ats": "ATS",
    "bass": "BASS",
    "asr": "ASR",
    "epoch116": "MetaSpiderMark 116",
    "epoch110": "MetaSpiderMark 110",
    "final": "MetaSpiderMark final (300)",
}
SELECTION = {
    "uniform": "Best augmented-validation AUROC through epoch 120",
    "bandit_ucb": "Best augmented-validation AUROC through epoch 120",
    "ats": "Best augmented-validation AUROC through epoch 120",
    "bass": "Best augmented-validation AUROC through epoch 120",
    "asr": "Best augmented-validation AUROC through epoch 120",
    "epoch116": "Highest augmented-validation accuracy in MetaSpiderMark history",
    "epoch110": "Highest augmented-validation AUROC in MetaSpiderMark history",
    "final": "Epoch-300 endpoint",
}
META_EPOCHS = {"epoch116": 116, "epoch110": 110, "final": 300}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-csv", type=Path, default=BASELINE_CSV)
    parser.add_argument("--meta-csv", type=Path, default=META_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def clean_number(value: float) -> float:
    return round(float(value), 8)


def source(source_id: str, label: str, path: str, description: str) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": description,
            "sql": f"SELECT * FROM read_csv_auto('{path}')",
            "tables_used": [path],
            "filters": ["Nine named attacks", "Five testing repetitions per attack"],
            "metric_definitions": [
                "mean_auc: arithmetic mean of model AUROC across the nine attacks.",
                "mean_acc: arithmetic mean of fixed-threshold accuracy across the nine attacks.",
                "worst_auc: minimum model AUROC among the nine attacks.",
                "worst_acc: minimum fixed-threshold accuracy among the nine attacks.",
            ],
        },
    }


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_csv)
    meta = pd.read_csv(args.meta_csv)
    expected_attacks = {
        "clean",
        "jpeg_strong",
        "msg_app_combo",
        "down_up",
        "blur",
        "random_crop",
        "occlusion",
        "geom_warp",
        "train_aug_mix",
    }
    if len(baseline) != 45 or set(baseline["attack"]) != expected_attacks:
        raise ValueError("Baseline evaluation must contain five models x nine attacks")
    if len(meta) != 27 or set(meta["attack"]) != expected_attacks:
        raise ValueError("Meta evaluation must contain three checkpoints x nine attacks")

    baseline = baseline.copy()
    meta = meta.copy()
    baseline["model_key"] = baseline["checkpoint_label"]
    meta["model_key"] = meta["checkpoint_label"]
    combined = pd.concat([baseline, meta], ignore_index=True)

    aggregate = (
        combined.groupby("model_key", as_index=False)
        .agg(
            mean_acc=("our_acc", "mean"),
            mean_auc=("our_auc", "mean"),
            worst_acc=("our_acc", "min"),
            worst_auc=("our_auc", "min"),
            attack_count=("attack", "nunique"),
        )
    )
    epoch_by_key = (
        baseline.groupby("model_key")["checkpoint_epoch"].first().astype(int).to_dict()
    )
    epoch_by_key.update(META_EPOCHS)
    aggregate["model"] = aggregate["model_key"].map(MODEL_LABELS)
    aggregate["checkpoint_epoch"] = aggregate["model_key"].map(epoch_by_key)
    aggregate["selection_basis"] = aggregate["model_key"].map(SELECTION)
    meta116 = aggregate.loc[aggregate["model_key"] == "epoch116"].iloc[0]
    aggregate["delta_mean_auc_vs_meta116"] = aggregate["mean_auc"] - meta116["mean_auc"]
    aggregate["delta_mean_acc_vs_meta116"] = aggregate["mean_acc"] - meta116["mean_acc"]
    aggregate["mean_auc_rank"] = aggregate["mean_auc"].rank(method="min", ascending=False).astype(int)
    aggregate["mean_acc_rank"] = aggregate["mean_acc"].rank(method="min", ascending=False).astype(int)
    aggregate = aggregate.sort_values("mean_auc", ascending=False)

    aggregate_rows = []
    profile_rows = []
    for row in aggregate.to_dict("records"):
        aggregate_rows.append(
            {
                "model": row["model"],
                "checkpoint_epoch": int(row["checkpoint_epoch"]),
                "selection_basis": row["selection_basis"],
                "attacks": int(row["attack_count"]),
                "mean_accuracy": clean_number(row["mean_acc"]),
                "mean_auroc": clean_number(row["mean_auc"]),
                "worst_accuracy": clean_number(row["worst_acc"]),
                "worst_auroc": clean_number(row["worst_auc"]),
                "delta_mean_accuracy_vs_meta116": clean_number(
                    row["delta_mean_acc_vs_meta116"]
                ),
                "delta_mean_auroc_vs_meta116": clean_number(
                    row["delta_mean_auc_vs_meta116"]
                ),
                "mean_auroc_rank": int(row["mean_auc_rank"]),
                "mean_accuracy_rank": int(row["mean_acc_rank"]),
            }
        )
        profile_rows.extend(
            [
                {
                    "model": row["model"],
                    "scope": "Mean AUROC",
                    "auroc": clean_number(row["mean_auc"]),
                    "checkpoint_epoch": int(row["checkpoint_epoch"]),
                    "mean_accuracy": clean_number(row["mean_acc"]),
                    "worst_accuracy": clean_number(row["worst_acc"]),
                    "rank": int(row["mean_auc_rank"]),
                },
                {
                    "model": row["model"],
                    "scope": "Worst-attack AUROC",
                    "auroc": clean_number(row["worst_auc"]),
                    "checkpoint_epoch": int(row["checkpoint_epoch"]),
                    "mean_accuracy": clean_number(row["mean_acc"]),
                    "worst_accuracy": clean_number(row["worst_acc"]),
                    "rank": int(row["mean_auc_rank"]),
                },
            ]
        )

    delta_rows = [
        row
        for row in aggregate_rows
        if row["model"] != "MetaSpiderMark 116"
    ]
    uniform = baseline[baseline["checkpoint_label"] == "uniform"].set_index("attack")
    meta116_attacks = meta[meta["checkpoint_label"] == "epoch116"].set_index("attack")
    attack_rows = []
    attack_order = [
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
    for attack in attack_order:
        attack_rows.append(
            {
                "attack": attack.replace("_", " ").title(),
                "attack_key": attack,
                "uniform_accuracy": clean_number(uniform.loc[attack, "our_acc"]),
                "meta116_accuracy": clean_number(meta116_attacks.loc[attack, "our_acc"]),
                "delta_accuracy": clean_number(
                    uniform.loc[attack, "our_acc"] - meta116_attacks.loc[attack, "our_acc"]
                ),
                "uniform_auroc": clean_number(uniform.loc[attack, "our_auc"]),
                "meta116_auroc": clean_number(meta116_attacks.loc[attack, "our_auc"]),
                "delta_auroc": clean_number(
                    uniform.loc[attack, "our_auc"] - meta116_attacks.loc[attack, "our_auc"]
                ),
                "baseline_predictions": 750,
                "meta_predictions": 750,
                "paired_realized_tensors": "No",
            }
        )

    uniform_summary = next(row for row in aggregate_rows if row["model"] == "Uniform")
    meta_summary = next(
        row for row in aggregate_rows if row["model"] == "MetaSpiderMark 116"
    )
    summary_rows = [
        {
            "uniform_mean_auroc": uniform_summary["mean_auroc"],
            "uniform_mean_auroc_delta": uniform_summary["delta_mean_auroc_vs_meta116"],
            "uniform_mean_accuracy": uniform_summary["mean_accuracy"],
            "uniform_mean_accuracy_delta": uniform_summary[
                "delta_mean_accuracy_vs_meta116"
            ],
            "meta116_worst_auroc": meta_summary["worst_auroc"],
            "uniform_worst_auroc": uniform_summary["worst_auroc"],
            "meta_floor_advantage": clean_number(
                meta_summary["worst_auroc"] - uniform_summary["worst_auroc"]
            ),
        }
    ]

    comparison_source = {
        "id": "comparison_analysis",
        "label": "Reproducible Stage 2 comparison analysis",
        "path": "scripts/build_stage2_comparison_report.py",
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": "Combines the reviewed baseline and MetaSpiderMark attack summaries and computes unweighted attack-level aggregates and deltas.",
            "sql": (
                "WITH combined AS (\n"
                "  SELECT checkpoint_label, attack, our_acc, our_auc FROM read_csv_auto('eval_results/stage2_downstream_shared120_best_auc/combined_attack_eval_summary.csv')\n"
                "  UNION ALL\n"
                "  SELECT checkpoint_label, attack, our_acc, our_auc FROM read_csv_auto('eval_results/downstream_meta_checkpoint_sweep/combined_attack_eval_summary.csv')\n"
                ")\n"
                "SELECT checkpoint_label, AVG(our_acc) AS mean_acc, AVG(our_auc) AS mean_auc, MIN(our_acc) AS worst_acc, MIN(our_auc) AS worst_auc\n"
                "FROM combined GROUP BY checkpoint_label"
            ),
            "tables_used": [str(args.baseline_csv).replace("\\", "/"), str(args.meta_csv).replace("\\", "/")],
            "filters": [
                "Nine common attacks only",
                "Five test repetitions per attack",
                "MetaSpiderMark 116 is the primary comparison baseline",
            ],
            "metric_definitions": [
                "Mean AUROC: unweighted arithmetic mean of AUROC across the nine common attacks.",
                "Mean accuracy: unweighted arithmetic mean of fixed 0.5-threshold accuracy across the nine common attacks.",
                "Worst-attack AUROC: minimum AUROC across the nine common attacks.",
                "Delta versus MetaSpiderMark 116: candidate metric minus the corresponding MetaSpiderMark epoch-116 metric.",
            ],
        },
    }
    baseline_source = source(
        "baseline_eval",
        "Shared downstream baseline evaluation",
        str(args.baseline_csv).replace("\\", "/"),
        "Five downstream baselines evaluated on a shared generated attack stream.",
    )
    meta_source = source(
        "meta_eval",
        "Historical MetaSpiderMark checkpoint sweep",
        str(args.meta_csv).replace("\\", "/"),
        "MetaSpiderMark epochs 110, 116, and 300 evaluated in the earlier checkpoint sweep.",
    )

    title = "Downstream Robustness: MetaSpiderMark vs Scheduler Baselines"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical comparison of nine-attack downstream robustness results.",
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "cards": [
                {
                    "id": "uniform_auc",
                    "description": "Uniform has the highest mean AUROC in the current directional comparison.",
                    "dataset": "summary",
                    "sourceId": "comparison_analysis",
                    "metrics": [
                        {"label": "Uniform mean AUROC", "field": "uniform_mean_auroc", "format": "number"},
                        {"label": "vs Meta 116", "field": "uniform_mean_auroc_delta", "format": "number", "signed": True},
                    ],
                },
                {
                    "id": "uniform_accuracy",
                    "description": "Uniform also leads MetaSpiderMark 116 on average fixed-threshold accuracy.",
                    "dataset": "summary",
                    "sourceId": "comparison_analysis",
                    "metrics": [
                        {"label": "Uniform mean accuracy", "field": "uniform_mean_accuracy", "format": "percent"},
                        {"label": "vs Meta 116", "field": "uniform_mean_accuracy_delta", "format": "percent", "signed": True},
                    ],
                },
                {
                    "id": "meta_floor",
                    "description": "MetaSpiderMark 116 preserves the strongest minimum AUROC across attacks.",
                    "dataset": "summary",
                    "sourceId": "comparison_analysis",
                    "metrics": [
                        {"label": "Meta 116 worst AUROC", "field": "meta116_worst_auroc", "format": "number"},
                        {"label": "advantage vs Uniform", "field": "meta_floor_advantage", "format": "number", "signed": True},
                    ],
                },
            ],
            "charts": [
                {
                    "id": "auroc_profile",
                    "title": "Mean and worst-attack AUROC by checkpoint",
                    "subtitle": "Nine attacks; higher is better. Mean captures overall performance, while the minimum captures the robustness floor.",
                    "type": "bar",
                    "dataset": "auroc_profile",
                    "sourceId": "comparison_analysis",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Checkpoint"},
                        "y": {"field": "auroc", "type": "quantitative", "label": "AUROC", "format": "number"},
                        "color": {"field": "scope", "type": "nominal", "label": "Metric"},
                    },
                    "valueFormat": "number",
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "settings": {"groupMode": "grouped", "showValues": True, "categoryLabelPolicy": "wrap"},
                    "labels": {"values": "all"},
                    "legend": {"position": "bottom", "sort": "spec", "title": "Metric"},
                    "layout": "full",
                },
                {
                    "id": "mean_auc_delta",
                    "title": "Mean AUROC difference versus MetaSpiderMark 116",
                    "subtitle": "Candidate minus MetaSpiderMark 116 across nine attacks; positive values favor the candidate.",
                    "type": "horizontalBar",
                    "dataset": "aggregate_delta",
                    "sourceId": "comparison_analysis",
                    "encodings": {
                        "x": {"field": "model", "type": "nominal", "label": "Checkpoint"},
                        "y": {"field": "delta_mean_auroc_vs_meta116", "type": "quantitative", "label": "AUROC difference", "format": "number"},
                    },
                    "valueFormat": "number",
                    "palette": {"kind": "diverging", "name": "blue-orange", "midpoint": 0},
                    "referenceLines": [{"axis": "y", "value": 0, "label": "MetaSpiderMark 116", "color": "neutral", "lineStyle": "dashed"}],
                    "settings": {"orientation": "horizontal", "showValues": True, "sort": "descending", "categoryLabelPolicy": "wrap"},
                    "labels": {"values": "all"},
                    "layout": "full",
                },
                {
                    "id": "uniform_attack_delta",
                    "title": "Uniform AUROC difference by attack",
                    "subtitle": "Uniform minus historical MetaSpiderMark 116; each side contains 750 predictions per attack, but tensors were generated in separate runs.",
                    "type": "horizontalBar",
                    "dataset": "attack_delta",
                    "sourceId": "comparison_analysis",
                    "encodings": {
                        "x": {"field": "attack", "type": "nominal", "label": "Attack"},
                        "y": {"field": "delta_auroc", "type": "quantitative", "label": "AUROC difference", "format": "number"},
                    },
                    "valueFormat": "number",
                    "palette": {"kind": "diverging", "name": "blue-orange", "midpoint": 0},
                    "referenceLines": [{"axis": "y", "value": 0, "label": "Equal AUROC", "color": "neutral", "lineStyle": "dashed"}],
                    "settings": {"orientation": "horizontal", "showValues": True, "sort": "descending", "categoryLabelPolicy": "wrap"},
                    "labels": {"values": "all"},
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "aggregate_table",
                    "title": "Aggregate checkpoint comparison",
                    "subtitle": "Unweighted results across the same nine attack categories.",
                    "dataset": "aggregate",
                    "sourceId": "comparison_analysis",
                    "defaultSort": {"field": "mean_auroc", "direction": "desc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "model", "label": "Model", "type": "text"},
                        {"field": "checkpoint_epoch", "label": "Epoch", "format": "number"},
                        {"field": "mean_accuracy", "label": "Mean accuracy", "format": "percent"},
                        {"field": "mean_auroc", "label": "Mean AUROC", "format": "number"},
                        {"field": "worst_accuracy", "label": "Worst accuracy", "format": "percent"},
                        {"field": "worst_auroc", "label": "Worst AUROC", "format": "number"},
                        {"field": "delta_mean_auroc_vs_meta116", "label": "AUROC delta vs Meta 116", "format": "number", "movement": True},
                    ],
                },
                {
                    "id": "attack_table",
                    "title": "Uniform and MetaSpiderMark 116 by attack",
                    "subtitle": "Accuracy and AUROC from nine common attacks; deltas are Uniform minus MetaSpiderMark 116.",
                    "dataset": "attack_delta",
                    "sourceId": "comparison_analysis",
                    "defaultSort": {"field": "delta_auroc", "direction": "desc"},
                    "density": "dense",
                    "layout": "full",
                    "columns": [
                        {"field": "attack", "label": "Attack", "type": "text"},
                        {"field": "uniform_accuracy", "label": "Uniform accuracy", "format": "percent"},
                        {"field": "meta116_accuracy", "label": "Meta 116 accuracy", "format": "percent"},
                        {"field": "delta_accuracy", "label": "Accuracy delta", "format": "percent", "movement": True},
                        {"field": "uniform_auroc", "label": "Uniform AUROC", "format": "number"},
                        {"field": "meta116_auroc", "label": "Meta 116 AUROC", "format": "number"},
                        {"field": "delta_auroc", "label": "AUROC delta", "format": "number", "movement": True},
                    ],
                },
            ],
            "sources": [
                comparison_source,
                baseline_source,
                meta_source,
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "body": (
                        "## Technical summary\n\n"
                        "**Uniform is the current average-performance leader**, reaching 0.9736 mean AUROC and 92.86% mean accuracy, ahead of MetaSpiderMark 116 by 0.0050 AUROC and 1.04 percentage points of accuracy. "
                        "**MetaSpiderMark 116 remains the strongest robustness-floor model**, with 0.8830 worst-attack AUROC versus Uniform's 0.8809. "
                        "Bandit-UCB also exceeds MetaSpiderMark 116 on mean AUROC, but not on mean accuracy. The remaining scheduler baselines trail MetaSpiderMark 116 on average AUROC.\n\n"
                        "These differences are **directional rather than final paired estimates**: all five scheduler baselines shared identical realized attack samples, whereas the historical MetaSpiderMark sweep used a separate random stream. A seeded MetaSpiderMark re-evaluation is required before making a publication claim about small gaps."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["uniform_auc", "uniform_accuracy", "meta_floor"]},
                {
                    "id": "profile_finding",
                    "type": "markdown",
                    "body": (
                        "## Uniform leads on average; MetaSpiderMark protects the floor\n\n"
                        "The paired bars separate overall discrimination from the weakest attack. Uniform ranks first on mean AUROC, but MetaSpiderMark 116 has the highest minimum AUROC and the highest minimum accuracy. This makes Uniform the strongest current all-attack average, while MetaSpiderMark remains the safer choice when the hardest transformation is the decision criterion."
                    ),
                },
                {"id": "profile_chart_block", "type": "chart", "chartId": "auroc_profile", "layout": "full"},
                {"id": "aggregate_table_block", "type": "table", "tableId": "aggregate_table", "layout": "full"},
                {
                    "id": "delta_finding",
                    "type": "markdown",
                    "body": (
                        "## Only Uniform clearly improves both headline averages\n\n"
                        "Uniform improves both mean accuracy and mean AUROC versus MetaSpiderMark 116. Bandit-UCB gains 0.0020 mean AUROC but loses 1.20 percentage points of accuracy, indicating stronger ranking quality with weaker calibration at the fixed 0.5 threshold. ASR is close on accuracy but lower on AUROC; ATS and BASS are lower on both aggregate measures."
                    ),
                },
                {"id": "delta_chart_block", "type": "chart", "chartId": "mean_auc_delta", "layout": "full"},
                {
                    "id": "attack_finding",
                    "type": "markdown",
                    "body": (
                        "## Uniform's gain is broad, but not universal\n\n"
                        "Uniform exceeds the historical MetaSpiderMark 116 AUROC on eight of nine attacks. Its largest visible gains occur under blur and strong JPEG compression. MetaSpiderMark 116 remains slightly better on the hardest training-augmentation mixture and has higher fixed-threshold accuracy under the messaging-app combination. The signed bars show where the aggregate advantage originates and where it reverses."
                    ),
                },
                {"id": "attack_delta_chart_block", "type": "chart", "chartId": "uniform_attack_delta", "layout": "full"},
                {"id": "attack_table_block", "type": "table", "tableId": "attack_table", "layout": "full"},
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        "The comparison covers the same nine attack categories: clean, strong JPEG, messaging-app combination, down/up sampling, blur, random crop, occlusion, geometric warp, and the training-augmentation mixture. Each attack result pools five testing repetitions over the 15% validation split.\n\n"
                        "**Accuracy** uses the verifier's fixed 0.5 decision threshold and therefore reflects discrimination plus calibration. **AUROC** measures threshold-free ranking quality. **Mean** metrics are unweighted arithmetic averages across the nine attacks. **Worst** metrics are the minimum across attacks and represent the observed robustness floor."
                    ),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "body": (
                        "## Evaluation and checkpoint methodology\n\n"
                        "The five scheduler baselines were downstream-trained together to epoch 120 using a shared on-the-fly batch stream, fixed MetaSpiderMark learning rates, and no learning-rate scheduler. Their reported checkpoints maximize augmented-validation AUROC: Uniform epoch 5, Bandit-UCB epoch 68, ATS epoch 49, BASS epoch 68, and ASR epoch 35.\n\n"
                        "MetaSpiderMark 116 is the primary historical comparator because it achieved the best attack-sweep result and highest augmented-validation accuracy; MetaSpiderMark 110 is retained as the validation-AUROC-selected comparator. The report also includes the epoch-300 endpoint for context. No attack-test result was used to select any of the five new baseline checkpoints."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations and confidence assessment\n\n"
                        "**Assessment: share with caveats.** The attack names, dataset split, testing count, transforms, and metrics align, but the historical MetaSpiderMark run did not evaluate the exact realized tensors used by the shared baseline run. Small differences—especially the 0.0020–0.0050 AUROC gaps—may therefore include random attack-sampling variation.\n\n"
                        "Checkpoint-selection criteria also differ: the five new baselines use best augmented-validation AUROC, MetaSpiderMark 116 was identified by augmented-validation accuracy and downstream attack performance, and MetaSpiderMark 110 is its best augmented-validation-AUROC checkpoint. The results are descriptive; no confidence intervals or statistical significance tests are available from the summary files alone."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        "1. Re-evaluate MetaSpiderMark 116 and 110 with the same attack-local seeds used by the shared baseline evaluator; no retraining is required.\n"
                        "2. Rebuild this report from the paired outputs and treat that version as the publication-ready comparison.\n"
                        "3. Report both average and worst-attack metrics: average-only reporting would hide MetaSpiderMark's stronger robustness floor.\n"
                        "4. If model ranking remains close, retain per-example predictions and use paired bootstrap intervals for AUROC and accuracy deltas."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does Uniform's advantage persist when MetaSpiderMark is evaluated on the exact same seeded tensors?\n"
                        "- Are the fixed-threshold accuracy differences mainly calibration effects that disappear after validation-only threshold calibration?\n"
                        "- Do the scheduler rankings remain stable across additional training seeds, or are they specific to seed 0?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": datetime.now().astimezone().isoformat(timespec="seconds"),
            "status": "ready",
            "datasets": {
                "summary": summary_rows,
                "aggregate": aggregate_rows,
                "auroc_profile": profile_rows,
                "aggregate_delta": delta_rows,
                "attack_delta": attack_rows,
            },
        },
        "sources": [comparison_source, baseline_source, meta_source],
        "package_info": {
            "root": "papers/meta_learning/reports/downstream_comparison_20260811",
            "manifestPath": "artifact.json",
            "snapshotPath": "artifact.json",
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
