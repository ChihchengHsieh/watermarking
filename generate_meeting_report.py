from __future__ import annotations

import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parent
SOURCE_REL = "eval_results/stage2_controlled_six_paired/combined_attack_eval_summary.csv"
PREDICTION_ROOT_REL = "eval_results/stage2_controlled_six_paired"
SOURCE = ROOT / SOURCE_REL
PREDICTION_ROOT = ROOT / PREDICTION_ROOT_REL
OUTPUT = ROOT / "artifact.json"
CI_OUTPUT = ROOT / "evaluation_confidence_intervals.csv"

BOOTSTRAP_SEED = 19980802
ACCURACY_BOOTSTRAPS = 10_000
AUROC_BOOTSTRAPS = 3_000

SOURCE_SQL = (
    "SELECT checkpoint_label, attack, CAST(our_auc AS REAL) AS our_auc, "
    "CAST(our_acc AS REAL) AS our_acc, CAST(best_our_thr AS REAL) AS best_our_thr, "
    "CAST(checkpoint_epoch AS INTEGER) AS checkpoint_epoch, evaluation_protocol "
    "FROM controlled_six_results"
)
CI_SQL = (
    "SELECT method_id, method, accuracy_rank, auc_rank, avg_accuracy_attacks, "
    "accuracy_ci_low, accuracy_ci_high, worst_accuracy, avg_auc_attacks, "
    "auc_ci_low, auc_ci_high, worst_auc FROM paired_bootstrap_results"
)

METHOD_LABELS = {
    "metaspidermark_original": "MetaSpiderMark (original)",
    "uniform": "Uniform",
    "bandit_ucb": "Bandit UCB",
    "ats": "ATS",
    "bass": "BASS",
    "asr": "ASR",
}

ATTACK_LABELS = {
    "clean": "Clean",
    "jpeg_strong": "Strong JPEG",
    "msg_app_combo": "Messaging-app combo",
    "down_up": "Down/up sampling",
    "blur": "Blur",
    "random_crop": "Random crop",
    "occlusion": "Occlusion",
    "geom_warp": "Geometric warp",
    "train_aug_mix": "Training augmentation mix",
}
ATTACKS = list(ATTACK_LABELS)
ATTACK_ONLY = [attack for attack in ATTACKS if attack != "clean"]
METHODS = list(METHOD_LABELS)


def avg(values: list[float]) -> float:
    return sum(values) / len(values)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
    raw_rows = list(csv.DictReader(handle))

if len(raw_rows) != 54:
    raise ValueError(f"Expected 54 rows (6 methods x 9 conditions), found {len(raw_rows)}")

required = {"checkpoint_label", "attack", "our_auc", "our_acc", "checkpoint_epoch"}
missing = required - set(raw_rows[0])
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

connection = sqlite3.connect(":memory:")
columns = list(raw_rows[0])
quoted_columns = ", ".join(f'"{column}" TEXT' for column in columns)
connection.execute(f"CREATE TABLE controlled_six_results ({quoted_columns})")
placeholders = ", ".join("?" for _ in columns)
connection.executemany(
    f"INSERT INTO controlled_six_results VALUES ({placeholders})",
    [[source_row[column] for column in columns] for source_row in raw_rows],
)
queried_rows = connection.execute(SOURCE_SQL).fetchall()

rows: list[dict] = []
by_method: dict[str, list[dict]] = defaultdict(list)
for checkpoint_label, attack, our_auc, our_acc, best_our_thr, checkpoint_epoch, protocol in queried_rows:
    row = {
        "method_id": checkpoint_label,
        "method": METHOD_LABELS[checkpoint_label],
        "attack_id": attack,
        "condition": ATTACK_LABELS[attack],
        "auc": our_auc,
        "accuracy": our_acc,
        "best_threshold": best_our_thr,
        "checkpoint_epoch": checkpoint_epoch,
        "evaluation_protocol": protocol,
    }
    rows.append(row)
    by_method[checkpoint_label].append(row)

auc_wins = defaultdict(int)
accuracy_wins = defaultdict(int)
for attack in ATTACKS:
    subset = [row for row in rows if row["attack_id"] == attack]
    max_auc = max(row["auc"] for row in subset)
    max_accuracy = max(row["accuracy"] for row in subset)
    for row in subset:
        if abs(row["auc"] - max_auc) < 1e-12:
            auc_wins[row["method_id"]] += 1
        if abs(row["accuracy"] - max_accuracy) < 1e-12:
            accuracy_wins[row["method_id"]] += 1

summary: list[dict] = []
for method_id, method_rows in by_method.items():
    attacked = [row for row in method_rows if row["attack_id"] != "clean"]
    clean = next(row for row in method_rows if row["attack_id"] == "clean")
    summary.append(
        {
            "method_id": method_id,
            "method": METHOD_LABELS[method_id],
            "avg_accuracy_attacks": avg([row["accuracy"] for row in attacked]),
            "avg_accuracy_all": avg([row["accuracy"] for row in method_rows]),
            "worst_accuracy": min(row["accuracy"] for row in method_rows),
            "clean_accuracy": clean["accuracy"],
            "avg_auc_attacks": avg([row["auc"] for row in attacked]),
            "avg_auc_all": avg([row["auc"] for row in method_rows]),
            "worst_auc": min(row["auc"] for row in method_rows),
            "clean_auc": clean["auc"],
            "accuracy_wins": accuracy_wins[method_id],
            "auc_wins": auc_wins[method_id],
            "checkpoint_epoch": clean["checkpoint_epoch"],
        }
    )

# Load aligned per-example predictions and verify the paired design.
prediction_scores: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
correct_predictions: dict[str, dict[str, np.ndarray]] = defaultdict(dict)
reference_gts: np.ndarray | None = None
for attack in ATTACKS:
    attack_gts: np.ndarray | None = None
    reference_psnr: np.ndarray | None = None
    reference_l1: np.ndarray | None = None
    for method in METHODS:
        result_path = PREDICTION_ROOT / method / attack / "eval_results.pt"
        result = torch.load(result_path, map_location="cpu", weights_only=False)
        scores = np.asarray(result["preds"], dtype=np.float64)
        gts = np.asarray(result["gts"], dtype=np.int64)
        psnrs = np.asarray(result["psnrs"], dtype=np.float64)
        l1s = np.asarray(result["l1s"], dtype=np.float64)
        if len(scores) != 750 or len(gts) != 750:
            raise ValueError(f"Expected 750 paired examples in {result_path}")
        if attack_gts is None:
            attack_gts = gts
            reference_psnr = psnrs
            reference_l1 = l1s
        elif not np.array_equal(gts, attack_gts):
            raise ValueError(f"Ground-truth order is not paired for {attack}")
        elif not np.allclose(psnrs, reference_psnr) or not np.allclose(l1s, reference_l1):
            raise ValueError(f"Example order is not paired for {attack}")
        prediction_scores[method][attack] = scores
        correct_predictions[method][attack] = (scores >= 0.5) == gts
    if reference_gts is None:
        reference_gts = attack_gts
    elif not np.array_equal(reference_gts, attack_gts):
        raise ValueError(f"Ground-truth order differs across attacks at {attack}")

assert reference_gts is not None
positive_indices = np.flatnonzero(reference_gts == 1)
negative_indices = np.flatnonzero(reference_gts == 0)

# Accuracy bootstrap: same stratified resample across methods and attacks.
accuracy_rng = np.random.default_rng(BOOTSTRAP_SEED)
accuracy_bootstrap = np.empty((ACCURACY_BOOTSTRAPS, len(METHODS)), dtype=np.float64)
for bootstrap_index in range(ACCURACY_BOOTSTRAPS):
    sampled = np.concatenate(
        [
            accuracy_rng.choice(positive_indices, len(positive_indices), replace=True),
            accuracy_rng.choice(negative_indices, len(negative_indices), replace=True),
        ]
    )
    for method_index, method in enumerate(METHODS):
        accuracy_bootstrap[bootstrap_index, method_index] = np.mean(
            [correct_predictions[method][attack][sampled].mean() for attack in ATTACK_ONLY]
        )

# AUROC bootstrap: same paired resample, preserving the eight-attack equal-weight metric.
auc_rng = np.random.default_rng(BOOTSTRAP_SEED)
auc_bootstrap = np.empty((AUROC_BOOTSTRAPS, len(METHODS)), dtype=np.float64)
for bootstrap_index in range(AUROC_BOOTSTRAPS):
    sampled = np.concatenate(
        [
            auc_rng.choice(positive_indices, len(positive_indices), replace=True),
            auc_rng.choice(negative_indices, len(negative_indices), replace=True),
        ]
    )
    sampled_gts = reference_gts[sampled]
    for method_index, method in enumerate(METHODS):
        auc_bootstrap[bootstrap_index, method_index] = np.mean(
            [
                roc_auc_score(sampled_gts, prediction_scores[method][attack][sampled])
                for attack in ATTACK_ONLY
            ]
        )

for method_row in summary:
    method_index = METHODS.index(method_row["method_id"])
    accuracy_low, accuracy_high = percentile_interval(accuracy_bootstrap[:, method_index])
    auc_low, auc_high = percentile_interval(auc_bootstrap[:, method_index])
    method_row["accuracy_ci_low"] = accuracy_low
    method_row["accuracy_ci_high"] = accuracy_high
    method_row["auc_ci_low"] = auc_low
    method_row["auc_ci_high"] = auc_high

accuracy_order = sorted(
    summary,
    key=lambda row: (-round(row["avg_accuracy_attacks"], 12), row["method"]),
)
previous_accuracy: float | None = None
previous_rank = 0
for position, row in enumerate(accuracy_order, start=1):
    if previous_accuracy is None or abs(row["avg_accuracy_attacks"] - previous_accuracy) >= 1e-12:
        previous_rank = position
        previous_accuracy = row["avg_accuracy_attacks"]
    row["accuracy_rank"] = previous_rank
auc_order = sorted(summary, key=lambda row: (-row["avg_auc_attacks"], row["method"]))
for rank, row in enumerate(auc_order, start=1):
    row["auc_rank"] = rank
summary = accuracy_order

meta = next(row for row in summary if row["method_id"] == "metaspidermark_original")
asr = next(row for row in summary if row["method_id"] == "asr")
ats = next(row for row in summary if row["method_id"] == "ats")
uniform = next(row for row in summary if row["method_id"] == "uniform")

meta_index = METHODS.index("metaspidermark_original")
asr_index = METHODS.index("asr")
ats_index = METHODS.index("ats")
accuracy_difference = accuracy_bootstrap[:, meta_index] - accuracy_bootstrap[:, asr_index]
auc_difference = auc_bootstrap[:, meta_index] - auc_bootstrap[:, ats_index]
accuracy_diff_low, accuracy_diff_high = percentile_interval(accuracy_difference)
auc_diff_low, auc_diff_high = percentile_interval(auc_difference)
accuracy_diff_point = meta["avg_accuracy_attacks"] - asr["avg_accuracy_attacks"]
auc_diff_point = meta["avg_auc_attacks"] - ats["avg_auc_attacks"]
if abs(accuracy_diff_point) < 1e-12:
    accuracy_diff_point = 0.0

headline_metrics = [
    {
        "meta_accuracy": meta["avg_accuracy_attacks"],
        "asr_accuracy": asr["avg_accuracy_attacks"],
        "accuracy_diff_pp": accuracy_diff_point * 100,
        "accuracy_diff_ci_low_pp": accuracy_diff_low * 100,
        "accuracy_diff_ci_high_pp": accuracy_diff_high * 100,
        "meta_auc": meta["avg_auc_attacks"],
        "ats_auc": ats["avg_auc_attacks"],
        "auc_diff_pp": auc_diff_point * 100,
        "auc_diff_ci_low_pp": auc_diff_low * 100,
        "auc_diff_ci_high_pp": auc_diff_high * 100,
        "uniform_worst_accuracy": uniform["worst_accuracy"],
    }
]

paired_comparison = [
    {
        "metric": "Accuracy (primary)",
        "candidate": "MetaSpiderMark (original)",
        "comparator": "ASR",
        "candidate_value": meta["avg_accuracy_attacks"],
        "comparator_value": asr["avg_accuracy_attacks"],
        "difference_pp": accuracy_diff_point * 100,
        "ci_low_pp": accuracy_diff_low * 100,
        "ci_high_pp": accuracy_diff_high * 100,
        "conclusion": "No distinguishable difference",
    },
    {
        "metric": "AUROC (secondary)",
        "candidate": "MetaSpiderMark (original)",
        "comparator": "ATS",
        "candidate_value": meta["avg_auc_attacks"],
        "comparator_value": ats["avg_auc_attacks"],
        "difference_pp": auc_diff_point * 100,
        "ci_low_pp": auc_diff_low * 100,
        "ci_high_pp": auc_diff_high * 100,
        "conclusion": "MetaSpiderMark leads on this fixed-checkpoint evaluation",
    },
]

# Save an exact, reusable CI summary beside the report.
ci_fields = [
    "method_id", "method", "accuracy_rank", "avg_accuracy_attacks", "accuracy_ci_low",
    "accuracy_ci_high", "worst_accuracy", "auc_rank", "avg_auc_attacks", "auc_ci_low",
    "auc_ci_high", "worst_auc",
]
with CI_OUTPUT.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=ci_fields)
    writer.writeheader()
    writer.writerows([{field: row[field] for field in ci_fields} for row in summary])

# Materialize and read the bootstrap summary through the SQL recorded in provenance.
connection.execute(
    "CREATE TABLE paired_bootstrap_results ("
    "method_id TEXT, method TEXT, accuracy_rank INTEGER, auc_rank INTEGER, "
    "avg_accuracy_attacks REAL, accuracy_ci_low REAL, accuracy_ci_high REAL, worst_accuracy REAL, "
    "avg_auc_attacks REAL, auc_ci_low REAL, auc_ci_high REAL, worst_auc REAL)"
)
connection.executemany(
    "INSERT INTO paired_bootstrap_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
    [
        (
            row["method_id"], row["method"], row["accuracy_rank"], row["auc_rank"],
            row["avg_accuracy_attacks"], row["accuracy_ci_low"], row["accuracy_ci_high"],
            row["worst_accuracy"], row["avg_auc_attacks"], row["auc_ci_low"],
            row["auc_ci_high"], row["worst_auc"],
        )
        for row in summary
    ],
)
if len(connection.execute(CI_SQL).fetchall()) != 6:
    raise ValueError("Bootstrap summary provenance query did not return six methods")

source_mtime = datetime.fromtimestamp(SOURCE.stat().st_mtime, tz=timezone.utc)
generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")

summary_source = {
    "id": "controlled_six_results",
    "label": "Controlled six-method paired evaluation summary",
    "path": SOURCE_REL,
    "query": {
        "engine": "sqlite",
        "sql": SOURCE_SQL,
        "description": "Loads the completed 54-row summary after importing the CSV into SQLite.",
        "executed_at": generated_at,
        "language": "sql",
        "filters": ["All six methods", "All nine conditions", "Fixed learned-model threshold of 0.5"],
        "metric_definitions": [
            "Attack mean accuracy: equal-weight arithmetic mean of fixed-0.5-threshold accuracy across eight non-clean attacks.",
            "Attack mean AUROC: equal-weight arithmetic mean of AUROC across eight non-clean attacks.",
            "Worst-case accuracy and AUROC: minimum value across all nine evaluation conditions.",
        ],
        "tables_used": ["controlled_six_results"],
    },
}

bootstrap_source = {
    "id": "paired_bootstrap_results",
    "label": "Paired bootstrap confidence-interval analysis",
    "path": "generate_meeting_report.py",
    "query": {
        "engine": "sqlite",
        "sql": CI_SQL,
        "description": "Reads method-level point estimates and paired bootstrap confidence intervals computed from the 54 per-example eval_results.pt files.",
        "executed_at": generated_at,
        "language": "sql",
        "filters": [
            f"Prediction files under {PREDICTION_ROOT_REL}",
            "750 aligned examples: 395 positive and 355 negative",
            "Eight non-clean attacks receive equal weight",
            f"Accuracy: {ACCURACY_BOOTSTRAPS:,} stratified paired bootstrap resamples",
            f"AUROC: {AUROC_BOOTSTRAPS:,} stratified paired bootstrap resamples",
            f"Bootstrap seed: {BOOTSTRAP_SEED}",
        ],
        "metric_definitions": [
            "Accuracy uses the fixed 0.5 learned-model threshold in the evaluation code.",
            "Each bootstrap resample preserves class counts and uses identical sampled example indices across methods and attacks.",
            "Intervals are percentile 95% confidence intervals over the fixed trained checkpoints and evaluation sample; they do not include training-seed variability.",
        ],
        "tables_used": ["paired_bootstrap_results"],
    },
}

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "Accuracy-First Watermark Method Evaluation",
        "description": "Confidence-interval comparison of six methods across clean data and eight attacks.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "accuracy_coleaders",
                "description": "Equal-weight mean accuracy across eight attacks; both methods use a fixed 0.5 threshold.",
                "dataset": "headline_metrics",
                "sourceId": "paired_bootstrap_results",
                "metrics": [
                    {"label": "MetaSpiderMark accuracy", "field": "meta_accuracy", "format": "percent"},
                    {"label": "ASR accuracy", "field": "asr_accuracy", "format": "percent"},
                ],
            },
            {
                "id": "accuracy_difference",
                "description": "Paired MetaSpiderMark-minus-ASR accuracy difference and percentile 95% CI, in percentage points.",
                "dataset": "headline_metrics",
                "sourceId": "paired_bootstrap_results",
                "metrics": [
                    {"label": "Accuracy difference (pp)", "field": "accuracy_diff_pp", "format": "number", "signed": True},
                    {"label": "CI low", "field": "accuracy_diff_ci_low_pp", "format": "number", "signed": True},
                    {"label": "CI high", "field": "accuracy_diff_ci_high_pp", "format": "number", "signed": True},
                ],
            },
            {
                "id": "auc_difference",
                "description": "Paired MetaSpiderMark-minus-ATS AUROC difference and percentile 95% CI, in percentage points.",
                "dataset": "headline_metrics",
                "sourceId": "paired_bootstrap_results",
                "metrics": [
                    {"label": "AUROC difference (pp)", "field": "auc_diff_pp", "format": "number", "signed": True},
                    {"label": "CI low", "field": "auc_diff_ci_low_pp", "format": "number", "signed": True},
                    {"label": "CI high", "field": "auc_diff_ci_high_pp", "format": "number", "signed": True},
                ],
            },
            {
                "id": "uniform_floor",
                "description": "Highest minimum accuracy across the nine evaluated conditions.",
                "dataset": "headline_metrics",
                "sourceId": "paired_bootstrap_results",
                "metrics": [
                    {"label": "Best worst-case accuracy", "field": "uniform_worst_accuracy", "format": "percent"},
                ],
            },
        ],
        "charts": [
            {
                "id": "accuracy_ranking",
                "title": "Attack Mean Accuracy by Method",
                "subtitle": "Fixed threshold of 0.5; equal-weight mean across eight attacks; 95% CIs appear in tooltips and the table",
                "showDescription": True,
                "type": "horizontalBar",
                "intent": "comparison",
                "dataset": "method_summary",
                "sourceId": "paired_bootstrap_results",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "method", "type": "nominal", "label": "Method"},
                    "y": {"field": "avg_accuracy_attacks", "type": "quantitative", "label": "Mean accuracy", "format": "percent"},
                    "tooltip": [
                        {"field": "accuracy_ci_low", "type": "quantitative", "label": "95% CI low", "format": "percent"},
                        {"field": "accuracy_ci_high", "type": "quantitative", "label": "95% CI high", "format": "percent"},
                        {"field": "worst_accuracy", "type": "quantitative", "label": "Worst-case accuracy", "format": "percent"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "condition_accuracy",
                "title": "Accuracy Across All Evaluation Conditions",
                "subtitle": "Six methods across nine conditions at a fixed 0.5 threshold",
                "showDescription": True,
                "type": "heatmap",
                "intent": "comparison",
                "dataset": "condition_results",
                "sourceId": "controlled_six_results",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "condition", "type": "nominal", "label": "Evaluation condition"},
                    "y": {"field": "accuracy", "type": "quantitative", "label": "Accuracy", "format": "percent"},
                    "color": {"field": "method", "type": "nominal", "label": "Method"},
                    "tooltip": [
                        {"field": "auc", "type": "quantitative", "label": "AUROC", "format": "percent"},
                        {"field": "checkpoint_epoch", "type": "quantitative", "label": "Checkpoint epoch"},
                    ],
                },
                "layout": "full",
            },
            {
                "id": "auc_ranking",
                "title": "Attack Mean AUROC by Method",
                "subtitle": "Secondary metric; equal-weight mean across eight attacks; 95% CIs appear in tooltips and the table",
                "showDescription": True,
                "type": "horizontalBar",
                "intent": "comparison",
                "dataset": "method_auc_summary",
                "sourceId": "paired_bootstrap_results",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "method", "type": "nominal", "label": "Method"},
                    "y": {"field": "avg_auc_attacks", "type": "quantitative", "label": "Mean AUROC", "format": "percent"},
                    "tooltip": [
                        {"field": "auc_ci_low", "type": "quantitative", "label": "95% CI low", "format": "percent"},
                        {"field": "auc_ci_high", "type": "quantitative", "label": "95% CI high", "format": "percent"},
                        {"field": "worst_auc", "type": "quantitative", "label": "Worst-case AUROC", "format": "percent"},
                    ],
                },
                "layout": "full",
            },
        ],
        "tables": [
            {
                "id": "paired_comparison_table",
                "title": "Paired Confidence-Interval Comparisons",
                "subtitle": "MetaSpiderMark versus the closest comparator for each headline metric",
                "showDescription": True,
                "dataset": "paired_comparison",
                "sourceId": "paired_bootstrap_results",
                "defaultSort": {"field": "metric", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "metric", "label": "Metric", "type": "text"},
                    {"field": "candidate", "label": "Candidate", "type": "text"},
                    {"field": "comparator", "label": "Comparator", "type": "text"},
                    {"field": "candidate_value", "label": "Candidate value", "format": "percent"},
                    {"field": "comparator_value", "label": "Comparator value", "format": "percent"},
                    {"field": "difference_pp", "label": "Difference (pp)", "format": "number"},
                    {"field": "ci_low_pp", "label": "95% CI low (pp)", "format": "number"},
                    {"field": "ci_high_pp", "label": "95% CI high (pp)", "format": "number"},
                    {"field": "conclusion", "label": "Interpretation", "type": "text"},
                ],
            },
            {
                "id": "method_ranking_table",
                "title": "Accuracy-First Ranking with 95% Confidence Intervals",
                "subtitle": "Attack means across 750 paired examples and eight equally weighted attacks",
                "showDescription": True,
                "dataset": "method_summary",
                "sourceId": "paired_bootstrap_results",
                "defaultSort": {"field": "accuracy_rank", "direction": "asc"},
                "density": "spacious",
                "layout": "full",
                "columns": [
                    {"field": "accuracy_rank", "label": "Accuracy rank", "format": "number"},
                    {"field": "method", "label": "Method", "type": "text"},
                    {"field": "avg_accuracy_attacks", "label": "Mean accuracy", "format": "percent"},
                    {"field": "accuracy_ci_low", "label": "Accuracy CI low", "format": "percent"},
                    {"field": "accuracy_ci_high", "label": "Accuracy CI high", "format": "percent"},
                    {"field": "worst_accuracy", "label": "Worst accuracy", "format": "percent"},
                    {"field": "avg_auc_attacks", "label": "Mean AUROC", "format": "percent"},
                    {"field": "auc_ci_low", "label": "AUROC CI low", "format": "percent"},
                    {"field": "auc_ci_high", "label": "AUROC CI high", "format": "percent"},
                ],
            },
        ],
        "sources": [summary_source, bootstrap_source],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# Accuracy-First Watermark Method Evaluation"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "sourceId": "paired_bootstrap_results",
                "body": (
                    "## Executive Summary\n\n"
                    "- **Accuracy does not separate MetaSpiderMark from ASR.** Both reach **90.17% attack mean accuracy**. The paired difference is **0.00 percentage points**, with a **95% CI of -0.72 to +0.70 points**.\n"
                    "- **The accuracy-first conclusion is a tie.** The interval crosses zero, so neither method is demonstrably better on average accuracy for these fixed checkpoints and this evaluation sample.\n"
                    "- **AUROC provides a secondary tie-breaker.** MetaSpiderMark reaches **96.38% attack mean AUROC** versus **95.50% for ATS**; its paired lead is **+0.89 points (95% CI: +0.55 to +1.24)**.\n"
                    "- **Uniform has the strongest accuracy floor.** Its worst-condition accuracy is **78.00%**, higher than MetaSpiderMark at **75.33%** and ASR at **74.67%**."
                ),
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["accuracy_coleaders", "accuracy_difference", "auc_difference", "uniform_floor"]},
            {
                "id": "definitions",
                "type": "markdown",
                "body": (
                    "## Decision Rule and Confidence-Interval Scope\n\n"
                    "**Accuracy is the primary metric and uses the evaluation code's fixed threshold of 0.5.** Methods are compared using the equal-weight mean across eight non-clean attacks. "
                    "The paired bootstrap resamples the same 750 aligned examples across every method and attack while preserving the observed class counts. These intervals measure evaluation-sample uncertainty for the existing checkpoints; they do not include retraining variability."
                ),
            },
            {
                "id": "accuracy_finding",
                "type": "markdown",
                "sourceId": "paired_bootstrap_results",
                "body": (
                    "## MetaSpiderMark and ASR Are Accuracy Co-Leaders\n\n"
                    "**Both methods achieve 90.17% mean accuracy across the eight attacks.** Their paired difference interval, **-0.72 to +0.70 percentage points**, spans zero and is centered at no difference. "
                    "The current evidence therefore supports a tie, not an accuracy win for either method."
                ),
            },
            {"id": "accuracy_chart", "type": "chart", "chartId": "accuracy_ranking"},
            {
                "id": "accuracy_interpretation",
                "type": "markdown",
                "body": "**Implication:** If average accuracy is the only decision criterion, retain both MetaSpiderMark and ASR as co-leaders. Use the production attack mix, computational cost, or a predeclared secondary metric to choose between them.",
            },
            {
                "id": "condition_finding",
                "type": "markdown",
                "sourceId": "controlled_six_results",
                "body": (
                    "## The Tie Hides Different Condition-Level Strengths\n\n"
                    "**MetaSpiderMark leads under Strong JPEG and Down/up sampling, while ASR leads under Messaging-app combo, Blur, and Occlusion.** "
                    "BASS leads Random crop, ATS leads Geometric warp, and Uniform leads Training augmentation mix. A deployment-weighted accuracy score could therefore change the preferred method."
                ),
            },
            {"id": "condition_chart", "type": "chart", "chartId": "condition_accuracy"},
            {
                "id": "condition_interpretation",
                "type": "markdown",
                "body": "**Implication:** Equal weighting is suitable for a neutral benchmark, but the final deployment choice should be recalculated if real-world attack frequencies are known.",
            },
            {
                "id": "auc_finding",
                "type": "markdown",
                "sourceId": "paired_bootstrap_results",
                "body": (
                    "## AUROC Favors MetaSpiderMark as a Secondary Metric\n\n"
                    "**MetaSpiderMark's 96.38% attack mean AUROC exceeds ATS by 0.89 percentage points.** The paired **95% CI of +0.55 to +1.24 points** remains above zero. "
                    "This supports MetaSpiderMark as a practical tie-breaker when ranking quality matters after the accuracy tie."
                ),
            },
            {"id": "auc_chart", "type": "chart", "chartId": "auc_ranking"},
            {
                "id": "auc_interpretation",
                "type": "markdown",
                "body": "**Implication:** Do not describe MetaSpiderMark as the accuracy winner. It is the AUROC leader and can be selected only if AUROC is accepted as a secondary decision criterion.",
            },
            {
                "id": "ci_detail",
                "type": "markdown",
                "body": "## What the Confidence Intervals Establish\n\nThe paired intervals directly test the relevant method differences. Overlap between separate method intervals is not the decision test; the paired difference interval is. Accuracy crosses zero, while the AUROC difference remains positive.",
            },
            {"id": "paired_ci_table", "type": "table", "tableId": "paired_comparison_table"},
            {
                "id": "ranking_detail",
                "type": "markdown",
                "body": "## Full Accuracy-First Ranking\n\nThe table preserves exact point estimates and method-level bootstrap intervals for both metrics. Accuracy rank is the primary ordering; AUROC is supporting context.",
            },
            {"id": "ranking_table_block", "type": "table", "tableId": "method_ranking_table"},
            {
                "id": "recommendations",
                "type": "markdown",
                "body": (
                    "## Recommended Meeting Position\n\n"
                    "1. **Report MetaSpiderMark and ASR as tied on average accuracy.** Do not claim an accuracy winner from this evaluation.\n"
                    "2. **Use MetaSpiderMark as the practical default only if AUROC is an accepted secondary criterion.** Its paired AUROC lead is positive throughout the 95% interval.\n"
                    "3. **Use Uniform when the worst-case accuracy floor is more important than average accuracy.**\n"
                    "4. **Keep the inference claim scoped to the fixed trained checkpoints.** New training seeds would be required to generalize over retraining variability."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Questions to Resolve After the Meeting\n\n"
                    "- What is the production frequency of each attack, and should the benchmark remain equally weighted?\n"
                    "- Is average accuracy or worst-case accuracy the deployment objective?\n"
                    "- Should AUROC, latency, or model size be the formal tie-breaker between MetaSpiderMark and ASR?"
                ),
            },
            {
                "id": "caveats",
                "type": "markdown",
                "sourceId": "paired_bootstrap_results",
                "body": (
                    "## Caveats and Assumptions\n\n"
                    "- The 95% intervals quantify evaluation-sample uncertainty for one fixed checkpoint per method; they do not include training-seed or checkpoint-selection variability.\n"
                    "- The bootstrap uses 750 paired examples with 395 positive and 355 negative labels and preserves those class counts.\n"
                    "- The eight attacks receive equal weight. Rankings may change under a different production attack distribution.\n"
                    "- Accuracy uses a fixed learned-model threshold of 0.5. The saved `best_our_thr` values are not used for `our_acc`.\n"
                    "- Methods use different selected checkpoint epochs; this is a selected-checkpoint comparison, not a fixed-training-step comparison."
                ),
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": generated_at,
        "status": "ready",
        "datasets": {
            "headline_metrics": headline_metrics,
            "method_summary": summary,
            "method_auc_summary": auc_order,
            "condition_results": rows,
            "paired_comparison": paired_comparison,
        },
    },
    "sources": [summary_source, bootstrap_source],
    "package_info": {
        "originUrl": "artifact://accuracy-first-watermark-comparison",
        "sourceSnapshotModifiedAt": source_mtime.isoformat().replace("+00:00", "Z"),
    },
}

OUTPUT.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Wrote {OUTPUT}")
print(f"Wrote {CI_OUTPUT}")
print(
    "Accuracy difference Meta-ASR: "
    f"{accuracy_diff_point * 100:+.2f} pp "
    f"(95% CI {accuracy_diff_low * 100:+.2f}, {accuracy_diff_high * 100:+.2f})"
)
print(
    "AUROC difference Meta-ATS: "
    f"{auc_diff_point * 100:+.2f} pp "
    f"(95% CI {auc_diff_low * 100:+.2f}, {auc_diff_high * 100:+.2f})"
)
