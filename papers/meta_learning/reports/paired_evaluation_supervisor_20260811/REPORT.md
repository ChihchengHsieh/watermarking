# Paired downstream evaluation: supervisor briefing

Recorded: 2026-08-11 22:57 E. Australia Standard Time

## Technical summary

In the completed paired evaluation, **Uniform ranks first on mean AUROC (0.973623) and mean accuracy (92.86%)**. MetaSpiderMark epoch 116 reaches 0.967304 mean AUROC and 91.67% mean accuracy, a gap of 0.006318 AUROC and 1.19 accuracy percentage points in Uniform's favour. However, **MetaSpiderMark epoch 116 has the best worst-attack AUROC (0.881975)**, slightly above Uniform (0.880906).

This ranking is a valid description of the finished systems under the shared paired evaluator. It is **not yet a controlled causal comparison of scheduler quality**, because the models were meta-trained with different episode sizes and random seeds.

## Aggregate results

| Rank | Model | Selected epoch | Mean accuracy | Mean AUROC | Worst accuracy | Worst AUROC |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Uniform | 5 | 92.86% | 0.973623 | 77.60% | 0.880906 |
| 2 | Bandit-UCB | 68 | 90.62% | 0.970584 | 75.73% | 0.878945 |
| 3 | MetaSpiderMark 110 | 110 | 91.11% | 0.967604 | 79.20% | 0.878902 |
| 4 | MetaSpiderMark 116 | 116 | 91.67% | 0.967304 | 79.07% | 0.881975 |
| 5 | ASR | 35 | 91.53% | 0.964851 | 76.93% | 0.867684 |
| 6 | ATS | 49 | 89.59% | 0.963681 | 73.73% | 0.871086 |
| 7 | BASS | 68 | 90.04% | 0.961718 | 77.07% | 0.868882 |

## What the paired rerun established

- All seven checkpoints were evaluated under `shared_downstream_attack_eval_v1` across the same nine attacks.
- Each model/attack result contains 750 predictions: 150 validation images repeated five times.
- Labels align across checkpoints, so evaluation-stream randomness is no longer the main explanation for the ranking.
- MetaSpiderMark 116 beats Uniform on mean accuracy in two attacks, ties once, and loses six; for AUROC it wins only on the training-augmentation mixture.

## Why meta-training parameters are a plausible confound

| Parameter | Scheduler baselines | Original MetaSpiderMark | Why it can matter |
|---|---:|---:|---|
| Support examples per task | 16 | 8 | More support examples can reduce noise in the inner-loop adaptation gradient. |
| Query examples per task | 16 | 8 | More query examples can reduce variance in the outer-loop meta-gradient. |
| Meta-batch size | 3 | 3 | Matched. |
| Meta-training steps | 2,000 | 2,000 | Matched in optimizer-step count. |
| Nominal sample slots | 192,000 | 96,000 | The baselines saw twice the support-plus-query exposure: `2000 × 3 × (support + query)`. |
| Seed | 0 | 19,980,802 | Initialization, split/order, episodes and stochastic attacks can differ. |

The 16/16 configuration therefore gave every scheduler baseline twice the nominal meta-training sample exposure of MetaSpiderMark while keeping the same 2,000 optimizer steps. That can make adaptation and meta-gradients more stable and may improve the final initialization. The different seed adds another source of variation. **These mechanisms are plausible, but the present experiment does not prove that either one caused Uniform's advantage.**

The mismatch also cannot explain why Uniform beat Bandit-UCB, ATS, BASS and ASR, because all five scheduler baselines used the same 16/16 setting and seed. Possible explanations for that internal ranking include a small, balanced seven-task pool that already suits uniform sampling; noisy or non-stationary scheduler feedback; and 120 epochs of downstream training reducing the value of a more specialized meta-initialization.

## Additional limitation

The baseline downstream checkpoints were chosen by best augmented-validation AUROC on the same 15% image split later used for attack evaluation, while the historical MetaSpiderMark checkpoints came from a different selection history. This can introduce selection optimism and means the comparison is not fully symmetric even though the final attack tensors are paired.

## Recommended interpretation for the meeting

> Under the paired downstream evaluator, Uniform currently has the best average performance, while MetaSpiderMark retains the strongest worst-attack AUROC. The paired rerun rules out evaluation randomness as the main explanation. However, the meta-training budgets were not matched: the scheduler baselines used 16 support and 16 query examples versus 8 and 8 for MetaSpiderMark, giving them twice the nominal sample exposure, and they also used a different seed. Therefore this is a ranking of the trained systems, not yet a clean causal comparison of scheduling methods.

## Next experiment

Retrain only the five scheduler baselines with the original MetaSpiderMark settings—support 8, query 8, meta-batch 3, 2,000 steps, seed 19,980,802—then downstream-train all six methods with identical settings and evaluate them in one paired run. The original MetaSpiderMark meta-checkpoint does not need to be retrained. The prepared resumable command is:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_stage2_controlled_six.ps1
```

## Scope, metric definitions and provenance

Accuracy is measured at the verifier's fixed threshold; AUROC is threshold-free ranking performance. Means are unweighted across nine attacks; worst values are minima across those attacks. The primary source is `eval_results/stage2_downstream_paired_meta/combined_attack_eval_summary.csv`. Original MetaSpiderMark settings are recorded in `[2] verifier_pretraining_meta_nvidia.ipynb`; scheduler-baseline defaults are in `scripts/run_stage2_scheduler_training.py`.

## Further questions

- Does Uniform still lead after the 8/8, same-seed controlled rerun?
- Are rankings stable across more than one meta-training seed?
- Would a held-out checkpoint-selection split change the downstream ranking?
