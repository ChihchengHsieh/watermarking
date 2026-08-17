"""Build the canonical artifact input for the Stage 2 convergence report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_LABELS = {
    "uniform": "Uniform",
    "bandit_ucb": "Bandit UCB",
    "ats": "ATS",
    "bass": "BASS",
    "asr": "ASR",
    "metaspidermark_original": "MetaSpiderMark original",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("papers/meta_learning/benchmark_outputs/stage2_controlled_six"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports/stage2_convergence"))
    parser.add_argument("--window", type=int, default=20)
    return parser.parse_args()


def slope(values: pd.Series) -> float:
    y = values.to_numpy(dtype=float)
    return float(np.polyfit(np.arange(len(y), dtype=float), y, 1)[0])


def method_name(path: Path) -> str:
    run_id = path.parent.parent.name
    short = run_id.removeprefix("controlled_").split("_seed", 1)[0]
    return METHOD_LABELS.get(short, short)


def main() -> None:
    args = parse_args()
    paths = sorted(args.root.glob("controlled_*/downstream_shared87_best_acc/history.csv"))
    if len(paths) != 6:
        raise RuntimeError(f"Expected six corrected histories, found {len(paths)}")

    curve_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for path in paths:
        data = pd.read_csv(path).sort_values("epoch").reset_index(drop=True)
        if len(data) != 87 or int(data.iloc[-1]["epoch"]) != 87:
            raise RuntimeError(f"Incomplete 87-epoch history: {path}")
        method = method_name(path)
        data["accuracy_rolling5"] = data["acc_aug"].rolling(5, min_periods=1).mean()
        data["auroc_rolling5"] = data["auc_aug"].rolling(5, min_periods=1).mean()
        data["val_loss_rolling5"] = data["val_loss_aug"].rolling(5, min_periods=1).mean()
        for row in data.itertuples(index=False):
            curve_rows.append(
                {
                    "method": method,
                    "epoch": int(row.epoch),
                    "validation_accuracy": float(row.acc_aug),
                    "validation_auroc": float(row.auc_aug),
                    "validation_loss": float(row.val_loss_aug),
                    "training_loss": float(row.train_loss),
                    "accuracy_rolling5": float(row.accuracy_rolling5),
                    "auroc_rolling5": float(row.auroc_rolling5),
                    "val_loss_rolling5": float(row.val_loss_rolling5),
                }
            )

        late = data.tail(args.window)
        previous = data.iloc[-2 * args.window : -args.window]
        acc_slope = slope(late["acc_aug"])
        auc_slope = slope(late["auc_aug"])
        loss_slope = slope(late["val_loss_aug"])
        best_acc = data.sort_values(
            ["acc_aug", "auc_aug", "epoch"], ascending=[False, False, True]
        ).iloc[0]
        best_auc = data.loc[data["auc_aug"].idxmax()]
        ongoing = (
            int(best_auc["epoch"]) >= 83
            or acc_slope >= 0.00075
            or auc_slope >= 0.0005
            or loss_slope <= -0.0025
        )
        summaries.append(
            {
                "method": method,
                "assessment": "Still improving" if ongoing else "Near plateau; not proven",
                "assessment_rank": 1 if ongoing else 2,
                "best_accuracy": float(best_acc["acc_aug"]),
                "best_accuracy_epoch": int(best_acc["epoch"]),
                "best_auroc": float(best_auc["auc_aug"]),
                "best_auroc_epoch": int(best_auc["epoch"]),
                "last10_accuracy_mean": float(data.tail(10)["acc_aug"].mean()),
                "last10_accuracy_sd": float(data.tail(10)["acc_aug"].std()),
                "last20_accuracy_slope_pp_per_epoch": 100.0 * acc_slope,
                "last20_auroc_slope_pp_per_epoch": 100.0 * auc_slope,
                "last20_val_loss_slope_per_epoch": loss_slope,
                "late_accuracy_change_pp": 100.0
                * (late["acc_aug"].mean() - previous["acc_aug"].mean()),
                "late_auroc_change_pp": 100.0
                * (late["auc_aug"].mean() - previous["auc_aug"].mean()),
            }
        )

    summary = pd.DataFrame(summaries).sort_values(
        ["assessment_rank", "last20_auroc_slope_pp_per_epoch"], ascending=[True, False]
    )
    curves = pd.DataFrame(curve_rows)
    generated_at = datetime.now(timezone.utc).isoformat()
    source = {
        "id": "corrected-training-histories",
        "label": "Six corrected downstream training histories",
        "path": "papers/meta_learning/benchmark_outputs/stage2_controlled_six/controlled_*/downstream_shared87_best_acc/history.csv",
        "query": {
            "description": "Epoch-level metrics from the shared downstream replay through epoch 87.",
            "language": "python",
            "filters": [
                "Six controlled methods",
                "Corrected Accuracy-checkpointing replay",
                "Epochs 1 through 87",
                "Augmented validation metrics",
            ],
            "metric_definitions": [
                "Validation Accuracy: fraction correctly classified at probability threshold 0.5 on the augmented validation split.",
                "Validation AUROC: threshold-free ROC area on the same augmented validation split.",
                "Late slope: ordinary least-squares slope over epochs 68 through 87.",
                "Still improving: late Accuracy slope at least 0.075 percentage points/epoch, AUROC slope at least 0.05 points/epoch, validation-loss slope at most -0.0025/epoch, or best AUROC in the last five epochs.",
            ],
        },
    }
    title = "Stage 2 Downstream Convergence Assessment"
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": "Technical assessment of whether all six downstream methods converged by epoch 87.",
            "generatedAt": generated_at,
            "sources": [source],
            "charts": [
                {
                    "id": "accuracy-curve",
                    "title": "Augmented validation Accuracy",
                    "subtitle": "Five-epoch rolling mean through epoch 87; six methods share the same validation protocol",
                    "intent": "trend",
                    "question": "Did validation Accuracy flatten before epoch 87 for every method?",
                    "rationale": "A multi-series line chart reveals sustained late improvement and stochastic variation across the full training horizon.",
                    "type": "line",
                    "dataset": "training_curves",
                    "sourceId": "corrected-training-histories",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                        "y": {
                            "field": "accuracy_rolling5",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Validation Accuracy (5-epoch mean)",
                        },
                        "color": {"field": "method", "type": "nominal", "label": "Method"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
                {
                    "id": "auroc-curve",
                    "title": "Augmented validation AUROC",
                    "subtitle": "Five-epoch rolling mean through epoch 87; late upward movement remains visible",
                    "intent": "trend",
                    "question": "Did validation AUROC flatten before epoch 87 for every method?",
                    "rationale": "The secondary metric identifies continuing ranking-quality gains even when thresholded Accuracy is noisy.",
                    "type": "line",
                    "dataset": "training_curves",
                    "sourceId": "corrected-training-histories",
                    "encodings": {
                        "x": {"field": "epoch", "type": "quantitative", "label": "Epoch"},
                        "y": {
                            "field": "auroc_rolling5",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Validation AUROC (5-epoch mean)",
                        },
                        "color": {"field": "method", "type": "nominal", "label": "Method"},
                    },
                    "valueFormat": "percent",
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "diagnostic-table",
                    "title": "Late-window convergence diagnostics",
                    "subtitle": "Epochs 68–87; positive Accuracy/AUROC slopes and negative validation-loss slopes indicate continued improvement",
                    "dataset": "diagnostic_summary",
                    "sourceId": "corrected-training-histories",
                    "defaultSort": {"field": "assessment_rank", "direction": "asc"},
                    "density": "spacious",
                    "layout": "full",
                    "columns": [
                        {"field": "method", "label": "Method", "type": "text"},
                        {"field": "assessment", "label": "Assessment", "type": "text"},
                        {"field": "best_accuracy", "label": "Best Accuracy", "format": "percent"},
                        {"field": "best_accuracy_epoch", "label": "Best Accuracy epoch", "format": "number"},
                        {"field": "last20_accuracy_slope_pp_per_epoch", "label": "Accuracy slope, pp/epoch", "format": "number"},
                        {"field": "last20_auroc_slope_pp_per_epoch", "label": "AUROC slope, pp/epoch", "format": "number"},
                        {"field": "last20_val_loss_slope_per_epoch", "label": "Validation-loss slope/epoch", "format": "number"},
                    ],
                }
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical-summary",
                    "type": "markdown",
                    "sourceId": "corrected-training-histories",
                    "body": "## Technical summary\n\n**No—all six are not demonstrably converged at epoch 87.** Five methods retain material late-window improvement in Accuracy, AUROC, or validation loss. ASR is closest to a plateau, but its validation loss and AUROC are still improving. Epoch 87 is therefore an interim checkpoint, not a convergence endpoint.",
                },
                {
                    "id": "accuracy-finding",
                    "type": "markdown",
                    "sourceId": "corrected-training-histories",
                    "body": "## Accuracy remains noisy and several methods still trend upward\n\nATS, Bandit UCB, BASS, and Uniform show positive late-window Accuracy slopes. MetaSpiderMark's raw Accuracy slope is smaller, but its late average and secondary metrics continue to improve. The five-epoch mean separates persistent movement from single-epoch spikes.",
                },
                {"id": "accuracy-chart-block", "type": "chart", "chartId": "accuracy-curve"},
                {
                    "id": "auroc-finding",
                    "type": "markdown",
                    "sourceId": "corrected-training-histories",
                    "body": "## AUROC and validation loss provide stronger evidence against convergence\n\nEvery method has a positive AUROC slope and falling augmented-validation loss over epochs 68–87. Bandit UCB reaches its best observed AUROC at epoch 87 itself, the clearest sign that the training horizon ended before a stable plateau was established.",
                },
                {"id": "auroc-chart-block", "type": "chart", "chartId": "auroc-curve"},
                {
                    "id": "diagnostics-finding",
                    "type": "markdown",
                    "body": "## Five methods are still improving; ASR is only near a plateau\n\nThe classification is intentionally conservative: a model is not called converged while a primary or secondary validation signal retains material directional movement.",
                },
                {"id": "diagnostic-table-block", "type": "table", "tableId": "diagnostic-table"},
                {
                    "id": "scope-definitions",
                    "type": "markdown",
                    "body": "## Scope and metric definitions\n\nThe assessment covers one corrected shared downstream-training run for six methods, epochs 1–87. Convergence is assessed on the augmented validation split because it owns checkpoint selection. Accuracy uses the fixed 0.5 probability threshold; AUROC is threshold-free. Five-epoch rolling curves are descriptive only; numerical slopes use the raw final 20 epochs.",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "corrected-training-histories",
                    "body": "## Diagnostic method\n\nFor each method, ordinary least-squares slopes were fitted to raw Accuracy, AUROC, and augmented-validation loss over epochs 68–87. The final ten-epoch mean and standard deviation, best metric epochs, and the change between consecutive 20-epoch windows were also checked. Evidence of continued improvement in any checkpoint-relevant validation signal prevents a definitive convergence call.",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": "## Limitations and uncertainty\n\nThis is one training seed, and validation augmentation introduces epoch-level noise. A best epoch occurring early does not prove convergence because it may be a noisy spike. Conversely, a small nonzero slope does not guarantee meaningful future gains. The analysis diagnoses the observed trajectory; it does not estimate training-seed variability.",
                },
                {
                    "id": "next-steps",
                    "type": "markdown",
                    "body": "## Recommended next step\n\nContinue the corrected shared run to epoch 120. Preserve `best_acc.pth` throughout and use Accuracy as the primary selection metric with AUROC only as the declared tie-breaker. Reassess convergence using the last 20 epochs and require no material rolling improvement for at least 15 epochs before stopping early.",
                },
                {
                    "id": "further-questions",
                    "type": "markdown",
                    "body": "## Further questions\n\n- Do the corrected trajectories flatten between epochs 88 and 120?\n- Does the Accuracy-selected checkpoint change after epoch 87?\n- Are the late improvements stable under a second downstream-training seed?",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "training_curves": curves.to_dict(orient="records"),
                "diagnostic_summary": summary.to_dict(orient="records"),
            },
        },
        "sources": [source],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "convergence_diagnostics.csv", index=False)
    (args.output_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print(f"[DONE] {args.output_dir / 'artifact.json'}")


if __name__ == "__main__":
    main()
