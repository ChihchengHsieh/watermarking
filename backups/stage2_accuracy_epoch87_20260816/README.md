# Stage 2 Accuracy-First Epoch-87 Snapshot

This snapshot preserves the accepted epoch-87 baseline before extending the shared downstream run to epoch 100.

Contents:

- `training/<run_id>/best_acc.pth`: Accuracy-selected checkpoint used by the full COCO evaluation.
- `training/<run_id>/latest.pth`: epoch-87 model and optimizer state for an exact training resume.
- `training/<run_id>/history.csv`: validation and training history through epoch 87.
- `training/shared_state.pth`: shared data-loader and RNG state aligned at epoch 87.
- `evaluation/`: the full 1,000-image COCO evaluation and bootstrap confidence intervals for the epoch-87 checkpoints.
- `reports/`: convergence and patience diagnostics produced before the extension.
- `snapshot_manifest.csv`: file sizes, timestamps, and SHA-256 hashes.

The epoch-100 evaluation must use a different output directory. If epoch 100 does not improve the result, use the checkpoints and evaluation preserved here.
