# Downstream Meta vs No-Meta Verifier Comparison

This compares the no-meta verifier results shown in the table, stored at:

`eval_results/our_improved_final/attack_eval_summary.csv`

against the downstream meta-learning checkpoint selected by validation accuracy:

`verifier_dataset_stablediff_octoweb_downstream_from_nvidia_meta_iter2000_300_epoch116.pth`

Meta-learning evaluation source:

`eval_results/downstream_meta_checkpoint_sweep/epoch116/attack_eval_summary.csv`

## Main Comparison

| Attack | No-meta Acc | No-meta AUROC | Meta Acc | Meta AUROC | Delta Acc | Delta AUROC |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.9467 | 0.9904 | 0.9800 | 0.9995 | +0.0333 | +0.0091 |
| jpeg_strong | 0.8720 | 0.9409 | 0.9347 | 0.9867 | +0.0627 | +0.0457 |
| msg_app_combo | 0.6813 | 0.8468 | 0.9173 | 0.9558 | +0.2360 | +0.1090 |
| down_up | 0.9133 | 0.9713 | 0.9533 | 0.9939 | +0.0400 | +0.0226 |
| blur | 0.8160 | 0.8757 | 0.9040 | 0.9678 | +0.0880 | +0.0920 |
| random_crop | 0.8787 | 0.9503 | 0.9027 | 0.9691 | +0.0240 | +0.0188 |
| occlusion | 0.9573 | 0.9878 | 0.9800 | 0.9993 | +0.0227 | +0.0114 |
| geom_warp | 0.8560 | 0.9381 | 0.8893 | 0.9626 | +0.0333 | +0.0245 |
| train_aug_mix | 0.7280 | 0.8620 | 0.8027 | 0.8830 | +0.0747 | +0.0211 |

## Average Over Attacks

| Method | Mean Acc | Mean AUROC |
|---|---:|---:|
| No-meta | 0.8499 | 0.9293 |
| Meta epoch116 | 0.9182 | 0.9686 |
| Delta | +0.0683 | +0.0393 |

## Takeaway

Using downstream meta initialization plus fine-tuning improves accuracy on every listed attack. The biggest gain is on `msg_app_combo`, where accuracy rises from `0.6813` to `0.9173`.

For the paper table, use `epoch116` as the meta-learning checkpoint unless a later full sweep with `epoch3` or `final` beats it.
