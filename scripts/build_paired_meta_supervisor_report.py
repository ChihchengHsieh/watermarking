"""Build a supervisor-ready record of the paired downstream evaluation.

The report deliberately separates measured results from plausible explanations.
It uses only the Python standard library so it can be regenerated in the local
PyTorch environment without installing reporting dependencies.
"""

from __future__ import annotations

import csv
import html
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "eval_results/stage2_downstream_paired_meta/combined_attack_eval_summary.csv"
OUT_DIR = ROOT / "papers/meta_learning/reports/paired_evaluation_supervisor_20260811"

DISPLAY = {
    "uniform": "Uniform",
    "bandit_ucb": "Bandit-UCB",
    "ats": "ATS",
    "bass": "BASS",
    "asr": "ASR",
    "metaspidermark_epoch110": "MetaSpiderMark 110",
    "metaspidermark_epoch116": "MetaSpiderMark 116",
}
ATTACK_DISPLAY = {
    "clean": "Clean",
    "jpeg_strong": "Strong JPEG",
    "msg_app_combo": "Messaging app",
    "down_up": "Down/up sampling",
    "blur": "Blur",
    "random_crop": "Random crop",
    "occlusion": "Occlusion",
    "geom_warp": "Geometric warp",
    "train_aug_mix": "Train augmentation mix",
}


def load_rows() -> list[dict[str, object]]:
    with SOURCE.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["our_acc"] = float(row["our_acc"])
        row["our_auc"] = float(row["our_auc"])
        row["checkpoint_epoch"] = int(float(row["checkpoint_epoch"]))
    expected = len(DISPLAY) * len(ATTACK_DISPLAY)
    keys = {(str(r["checkpoint_label"]), str(r["attack"])) for r in rows}
    protocols = {str(r["evaluation_protocol"]) for r in rows}
    if len(rows) != expected or len(keys) != expected:
        raise ValueError(f"Expected {expected} unique model/attack rows, found {len(rows)} rows and {len(keys)} keys")
    if protocols != {"shared_downstream_attack_eval_v1"}:
        raise ValueError(f"Unexpected evaluation protocols: {sorted(protocols)}")
    return rows


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["checkpoint_label"])].append(row)
    result = []
    for model, values in grouped.items():
        result.append(
            {
                "model": model,
                "epoch": values[0]["checkpoint_epoch"],
                "mean_acc": mean(float(v["our_acc"]) for v in values),
                "mean_auc": mean(float(v["our_auc"]) for v in values),
                "worst_acc": min(float(v["our_acc"]) for v in values),
                "worst_auc": min(float(v["our_auc"]) for v in values),
            }
        )
    return sorted(result, key=lambda item: float(item["mean_auc"]), reverse=True)


def svg_mean_auc(summary: list[dict[str, object]]) -> str:
    width, left, top, row_h = 860, 190, 30, 42
    plot_w = width - left - 80
    low, high = 0.95, 0.98
    height = top + row_h * len(summary) + 55
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Mean AUROC by model">']
    for tick in (0.95, 0.96, 0.97, 0.98):
        x = left + (tick - low) / (high - low) * plot_w
        parts.append(f'<line x1="{x:.1f}" y1="15" x2="{x:.1f}" y2="{height-35}" class="grid"/>')
        parts.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle">{tick:.2f}</text>')
    for i, item in enumerate(summary):
        y = top + i * row_h
        auc = float(item["mean_auc"])
        bar_w = max(0, (auc - low) / (high - low) * plot_w)
        color = "#e8792e" if str(item["model"]).startswith("metaspidermark") else "#2774ae"
        parts.append(f'<text x="{left-12}" y="{y+20}" text-anchor="end">{html.escape(DISPLAY[str(item["model"])])}</text>')
        parts.append(f'<rect x="{left}" y="{y+5}" width="{bar_w:.1f}" height="22" rx="4" fill="{color}"/>')
        parts.append(f'<text x="{left+bar_w+7:.1f}" y="{y+21}" class="value">{auc:.4f}</text>')
    parts.append('<text x="430" y="18" text-anchor="middle" class="axis-title">Zoomed axis: 0.95–0.98; higher is better</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_attack_delta(rows: list[dict[str, object]]) -> str:
    by_key = {(str(r["checkpoint_label"]), str(r["attack"])): r for r in rows}
    attacks = list(ATTACK_DISPLAY)
    width, left, top, row_h = 900, 205, 28, 38
    plot_w, limit = 600, 0.016
    zero = left + plot_w / 2
    height = top + row_h * len(attacks) + 48
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="MetaSpiderMark 116 minus Uniform AUROC by attack">']
    parts.append(f'<line x1="{zero}" y1="18" x2="{zero}" y2="{height-30}" class="zero"/>')
    parts.append(f'<text x="{left}" y="{height-9}" text-anchor="middle">−0.016 (Uniform better)</text>')
    parts.append(f'<text x="{left+plot_w}" y="{height-9}" text-anchor="middle">+0.016 (Meta better)</text>')
    for i, attack in enumerate(attacks):
        y = top + i * row_h
        delta = float(by_key[("metaspidermark_epoch116", attack)]["our_auc"]) - float(by_key[("uniform", attack)]["our_auc"])
        dx = delta / limit * (plot_w / 2)
        x = zero if dx >= 0 else zero + dx
        color = "#e8792e" if delta >= 0 else "#2774ae"
        parts.append(f'<text x="{left-12}" y="{y+18}" text-anchor="end">{html.escape(ATTACK_DISPLAY[attack])}</text>')
        parts.append(f'<rect x="{x:.1f}" y="{y+5}" width="{abs(dx):.1f}" height="20" rx="3" fill="{color}"/>')
        anchor = "start" if delta >= 0 else "end"
        tx = zero + dx + (7 if delta >= 0 else -7)
        parts.append(f'<text x="{tx:.1f}" y="{y+20}" text-anchor="{anchor}" class="value">{delta:+.4f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_markdown(summary: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    by_model = {str(item["model"]): item for item in summary}
    uniform, meta = by_model["uniform"], by_model["metaspidermark_epoch116"]
    acc_gap = (float(uniform["mean_acc"]) - float(meta["mean_acc"])) * 100
    auc_gap = float(uniform["mean_auc"]) - float(meta["mean_auc"])
    lines = [
        "# Paired downstream evaluation: supervisor briefing",
        "",
        f"Recorded: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}",
        "",
        "## Technical summary",
        "",
        f"In the completed paired evaluation, **Uniform ranks first on mean AUROC ({float(uniform['mean_auc']):.6f}) and mean accuracy ({float(uniform['mean_acc'])*100:.2f}%)**. MetaSpiderMark epoch 116 reaches {float(meta['mean_auc']):.6f} mean AUROC and {float(meta['mean_acc'])*100:.2f}% mean accuracy, a gap of {auc_gap:.6f} AUROC and {acc_gap:.2f} accuracy percentage points in Uniform's favour. However, **MetaSpiderMark epoch 116 has the best worst-attack AUROC ({float(meta['worst_auc']):.6f})**, slightly above Uniform ({float(uniform['worst_auc']):.6f}).",
        "",
        "This ranking is a valid description of the finished systems under the shared paired evaluator. It is **not yet a controlled causal comparison of scheduler quality**, because the models were meta-trained with different episode sizes and random seeds.",
        "",
        "## Aggregate results",
        "",
        "| Rank | Model | Selected epoch | Mean accuracy | Mean AUROC | Worst accuracy | Worst AUROC |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for rank, item in enumerate(summary, 1):
        lines.append(f"| {rank} | {DISPLAY[str(item['model'])]} | {item['epoch']} | {float(item['mean_acc'])*100:.2f}% | {float(item['mean_auc']):.6f} | {float(item['worst_acc'])*100:.2f}% | {float(item['worst_auc']):.6f} |")
    lines += [
        "",
        "## What the paired rerun established",
        "",
        "- All seven checkpoints were evaluated under `shared_downstream_attack_eval_v1` across the same nine attacks.",
        "- Each model/attack result contains 750 predictions: 150 validation images repeated five times.",
        "- Labels align across checkpoints, so evaluation-stream randomness is no longer the main explanation for the ranking.",
        "- MetaSpiderMark 116 beats Uniform on mean accuracy in two attacks, ties once, and loses six; for AUROC it wins only on the training-augmentation mixture.",
        "",
        "## Why meta-training parameters are a plausible confound",
        "",
        "| Parameter | Scheduler baselines | Original MetaSpiderMark | Why it can matter |",
        "|---|---:|---:|---|",
        "| Support examples per task | 16 | 8 | More support examples can reduce noise in the inner-loop adaptation gradient. |",
        "| Query examples per task | 16 | 8 | More query examples can reduce variance in the outer-loop meta-gradient. |",
        "| Meta-batch size | 3 | 3 | Matched. |",
        "| Meta-training steps | 2,000 | 2,000 | Matched in optimizer-step count. |",
        "| Nominal sample slots | 192,000 | 96,000 | The baselines saw twice the support-plus-query exposure: `2000 × 3 × (support + query)`. |",
        "| Seed | 0 | 19,980,802 | Initialization, split/order, episodes and stochastic attacks can differ. |",
        "",
        "The 16/16 configuration therefore gave every scheduler baseline twice the nominal meta-training sample exposure of MetaSpiderMark while keeping the same 2,000 optimizer steps. That can make adaptation and meta-gradients more stable and may improve the final initialization. The different seed adds another source of variation. **These mechanisms are plausible, but the present experiment does not prove that either one caused Uniform's advantage.**",
        "",
        "The mismatch also cannot explain why Uniform beat Bandit-UCB, ATS, BASS and ASR, because all five scheduler baselines used the same 16/16 setting and seed. Possible explanations for that internal ranking include a small, balanced seven-task pool that already suits uniform sampling; noisy or non-stationary scheduler feedback; and 120 epochs of downstream training reducing the value of a more specialized meta-initialization.",
        "",
        "## Additional limitation",
        "",
        "The baseline downstream checkpoints were chosen by best augmented-validation AUROC on the same 15% image split later used for attack evaluation, while the historical MetaSpiderMark checkpoints came from a different selection history. This can introduce selection optimism and means the comparison is not fully symmetric even though the final attack tensors are paired.",
        "",
        "## Recommended interpretation for the meeting",
        "",
        "> Under the paired downstream evaluator, Uniform currently has the best average performance, while MetaSpiderMark retains the strongest worst-attack AUROC. The paired rerun rules out evaluation randomness as the main explanation. However, the meta-training budgets were not matched: the scheduler baselines used 16 support and 16 query examples versus 8 and 8 for MetaSpiderMark, giving them twice the nominal sample exposure, and they also used a different seed. Therefore this is a ranking of the trained systems, not yet a clean causal comparison of scheduling methods.",
        "",
        "## Next experiment",
        "",
        "Retrain only the five scheduler baselines with the original MetaSpiderMark settings—support 8, query 8, meta-batch 3, 2,000 steps, seed 19,980,802—then downstream-train all six methods with identical settings and evaluate them in one paired run. The original MetaSpiderMark meta-checkpoint does not need to be retrained. The prepared resumable command is:",
        "",
        "```powershell",
        "powershell -ExecutionPolicy Bypass -File scripts\\run_stage2_controlled_six.ps1",
        "```",
        "",
        "## Scope, metric definitions and provenance",
        "",
        "Accuracy is measured at the verifier's fixed threshold; AUROC is threshold-free ranking performance. Means are unweighted across nine attacks; worst values are minima across those attacks. The primary source is `eval_results/stage2_downstream_paired_meta/combined_attack_eval_summary.csv`. Original MetaSpiderMark settings are recorded in `[2] verifier_pretraining_meta_nvidia.ipynb`; scheduler-baseline defaults are in `scripts/run_stage2_scheduler_training.py`.",
        "",
        "## Further questions",
        "",
        "- Does Uniform still lead after the 8/8, same-seed controlled rerun?",
        "- Are rankings stable across more than one meta-training seed?",
        "- Would a held-out checkpoint-selection split change the downstream ranking?",
    ]
    return "\n".join(lines) + "\n"


def markdown_table_to_html(summary: list[dict[str, object]]) -> str:
    body = "".join(
        f"<tr><td>{rank}</td><td>{html.escape(DISPLAY[str(item['model'])])}</td><td>{item['epoch']}</td><td>{float(item['mean_acc'])*100:.2f}%</td><td>{float(item['mean_auc']):.6f}</td><td>{float(item['worst_acc'])*100:.2f}%</td><td>{float(item['worst_auc']):.6f}</td></tr>"
        for rank, item in enumerate(summary, 1)
    )
    return f"<table><thead><tr><th>Rank</th><th>Model</th><th>Epoch</th><th>Mean accuracy</th><th>Mean AUROC</th><th>Worst accuracy</th><th>Worst AUROC</th></tr></thead><tbody>{body}</tbody></table>"


def build_html(summary: list[dict[str, object]], rows: list[dict[str, object]]) -> str:
    markdown = build_markdown(summary, rows)
    by_model = {str(item["model"]): item for item in summary}
    u, m = by_model["uniform"], by_model["metaspidermark_epoch116"]
    css = """
    :root{--ink:#17212b;--muted:#5d6b78;--line:#d9e1e8;--panel:#f7f9fb;--blue:#2774ae;--orange:#e8792e}*{box-sizing:border-box}body{margin:0;background:#eef2f5;color:var(--ink);font:16px/1.58 Inter,Segoe UI,Arial,sans-serif}.page{max-width:1080px;margin:32px auto;background:white;padding:54px 64px;box-shadow:0 8px 30px #2638491c}h1{font-size:36px;line-height:1.15;margin:0 0 8px}h2{margin-top:42px;border-top:1px solid var(--line);padding-top:28px}h3{margin-top:28px}.subtitle,.note{color:var(--muted)}.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:26px 0}.metric{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px}.metric b{display:block;font-size:28px}.metric span{color:var(--muted);font-size:13px}.callout{border-left:5px solid var(--orange);background:#fff7f0;padding:16px 20px;margin:24px 0}.chart{border:1px solid var(--line);border-radius:10px;padding:14px;margin:18px 0 28px;overflow:auto}svg{width:100%;min-width:720px;font:13px Segoe UI,Arial}.grid{stroke:#dce3e8;stroke-width:1}.zero{stroke:#586875;stroke-width:1.5;stroke-dasharray:5 4}.value{font-weight:600}.axis-title{font-weight:600;fill:#5d6b78}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right}th{background:var(--panel)}th:nth-child(2),td:nth-child(2){text-align:left}.params td:first-child,.params th:first-child,.params td:last-child,.params th:last-child{text-align:left}code{background:#f2f5f7;padding:2px 5px;border-radius:4px}pre{background:#17212b;color:#f4f7fa;padding:16px;border-radius:8px;overflow:auto}.talk{font-size:18px;background:#edf6fc;border-left:5px solid var(--blue);padding:20px 24px}.footer{margin-top:46px;color:var(--muted);font-size:13px}@media(max-width:760px){.page{margin:0;padding:28px 22px}.metrics{grid-template-columns:1fr}}
    """
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Paired downstream evaluation — supervisor briefing</title><style>{css}</style></head><body><main class="page">
    <h1>Paired downstream evaluation</h1><p class="subtitle">Supervisor briefing · recorded {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}</p>
    <h2>Technical summary</h2><p><strong>Uniform leads the completed paired evaluation on average</strong>, while <strong>MetaSpiderMark 116 retains the strongest worst-attack AUROC</strong>. This is a valid ranking of the finished systems, but not yet a controlled causal comparison of scheduler quality because their meta-training episode sizes and seeds differ.</p>
    <div class="metrics"><div class="metric"><b>{float(u['mean_auc']):.4f}</b><span>Uniform mean AUROC · rank 1</span></div><div class="metric"><b>{float(u['mean_acc'])*100:.2f}%</b><span>Uniform mean accuracy · +{(float(u['mean_acc'])-float(m['mean_acc']))*100:.2f} pp vs Meta 116</span></div><div class="metric"><b>{float(m['worst_auc']):.4f}</b><span>MetaSpiderMark 116 worst AUROC · best robustness floor</span></div></div>
    <h2>Aggregate results</h2>{markdown_table_to_html(summary)}
    <h3>Mean AUROC</h3><div class="chart">{svg_mean_auc(summary)}</div>
    <h2>Where Uniform's AUROC advantage comes from</h2><p>The signed bars show MetaSpiderMark 116 minus Uniform. MetaSpiderMark is slightly better only on the training-augmentation mixture; Uniform is better on the other eight attacks.</p><div class="chart">{svg_attack_delta(rows)}</div>
    <h2>What the paired rerun established</h2><ul><li>Seven checkpoints, nine attacks, and one shared evaluation protocol.</li><li>Each model/attack result contains 750 predictions: 150 validation images repeated five times.</li><li>Labels align across checkpoints, so evaluation-stream randomness is no longer the main explanation.</li><li>MetaSpiderMark 116 has the best worst-attack AUROC ({float(m['worst_auc']):.6f}) versus Uniform ({float(u['worst_auc']):.6f}).</li></ul>
    <h2>Why meta-training parameters may matter</h2><table class="params"><thead><tr><th>Parameter</th><th>Scheduler baselines</th><th>Original MetaSpiderMark</th><th>Interpretation</th></tr></thead><tbody><tr><td>Support per task</td><td>16</td><td>8</td><td>Larger support can stabilize the inner-loop gradient.</td></tr><tr><td>Query per task</td><td>16</td><td>8</td><td>Larger query can reduce outer-gradient variance.</td></tr><tr><td>Meta-batch</td><td>3</td><td>3</td><td>Matched.</td></tr><tr><td>Steps</td><td>2,000</td><td>2,000</td><td>Matched optimizer-step count.</td></tr><tr><td>Nominal sample slots</td><td>192,000</td><td>96,000</td><td>Baselines received 2× exposure.</td></tr><tr><td>Seed</td><td>0</td><td>19,980,802</td><td>Initialization, order, episodes and attacks can vary.</td></tr></tbody></table>
    <div class="callout"><strong>Important:</strong> this is a plausible confound, not a proven cause. The 16/16 setting also cannot explain why Uniform beat the four adaptive baselines, because all five used the same meta-training budget. That result may reflect the small balanced task pool, noisy scheduler feedback, seed sensitivity, or downstream training washing out meta-initialization differences.</div>
    <h2>Additional limitation</h2><p>The baseline checkpoints were selected by best augmented-validation AUROC on the same 15% image split later used for attack evaluation, whereas MetaSpiderMark came from a different checkpoint-selection history. This asymmetry may add selection optimism.</p>
    <h2>Suggested meeting explanation</h2><p class="talk">Under the paired downstream evaluator, Uniform currently has the best average performance, while MetaSpiderMark retains the strongest worst-attack AUROC. The paired rerun rules out evaluation randomness as the main explanation. However, the meta-training budgets were not matched: the scheduler baselines used 16 support and 16 query examples versus 8 and 8 for MetaSpiderMark, giving them twice the nominal sample exposure, and they used a different seed. Therefore this is a ranking of the trained systems, not yet a clean causal comparison of scheduling methods.</p>
    <h2>Recommended next experiment</h2><p>Retrain only the five scheduler baselines with support 8, query 8, meta-batch 3, 2,000 steps and seed 19,980,802. Then downstream-train all six methods identically and evaluate them together. The original MetaSpiderMark meta-checkpoint does not need retraining.</p><pre>powershell -ExecutionPolicy Bypass -File scripts\\run_stage2_controlled_six.ps1</pre>
    <h2>Scope and provenance</h2><p>Accuracy uses the verifier's fixed threshold; AUROC is threshold-free. Means are unweighted across nine attacks; worst values are minima. Primary data: <code>eval_results/stage2_downstream_paired_meta/combined_attack_eval_summary.csv</code>. Configuration evidence: <code>[2] verifier_pretraining_meta_nvidia.ipynb</code> and <code>scripts/run_stage2_scheduler_training.py</code>.</p>
    <p class="footer">The companion REPORT.md contains the same conclusions in a diff-friendly format. This HTML is self-contained and can be opened directly in a browser.</p>
    <!-- Embedded source narrative for auditability:\n{html.escape(markdown)}\n--></main></body></html>"""


def main() -> None:
    rows = load_rows()
    summary = aggregate(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "REPORT.md").write_text(build_markdown(summary, rows), encoding="utf-8")
    (OUT_DIR / "report.html").write_text(build_html(summary, rows), encoding="utf-8")
    print(OUT_DIR / "REPORT.md")
    print(OUT_DIR / "report.html")


if __name__ == "__main__":
    main()
