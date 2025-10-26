## Robustness Evaluation (Ours vs PSNR)

We compare our learned watermark detector (**Ours**) against a PSNR-based baseline (**PSNR**) under multiple real-world attack scenarios.  
Both models are evaluated using **Accuracy** (threshold = 0.5) and **AUROC** (threshold-independent separability).  
For PSNR, we apply a **threshold of –4**, which converts PSNR values into probabilities before computing metrics.

### Attack Types Explained

| Attack | Description |
|:-------|:-------------|
| **clean** | No augmentation — the original, unmodified image. |
| **jpeg_strong** | Strong JPEG compression (low quality factor, simulating heavy online recompression). |
| **down_up** | Simple rescaling (downscale + upscale) without compression, to test robustness to interpolation. |
| **blur** | Gaussian blur applied to simulate optical defocus or motion blur. |
| **random_crop** | Random resized cropping, mimicking reframing or partial viewing of the image. |
| **occlusion** | Part of the image masked with a black rectangle to test spatial redundancy of the watermark. |
| **geom_warp** | Small geometric warps (rotation, affine transform) to test robustness to viewpoint distortion. |
| **train_aug_mix** | A mixture of augmentations used during training (mild rotations, color jitter, JPEG, etc.), to test generalization. |

---

### Quantitative Results

| Attack        | Accuracy (Ours) | AUROC (Ours) | Accuracy (PSNR) | AUROC (PSNR) |
|---------------|----------------:|-------------:|-----------------:|--------------:|
| clean         | 0.9333          | 0.9918       | 0.9533           | 0.9557        |
| jpeg_strong   | 0.7467          | 0.8297       | 0.5800           | 0.6013        |
| down_up       | 0.8133          | 0.8645       | 0.5533           | 0.5759        |
| blur          | 0.6213          | 0.7027       | 0.4733           | 0.5000        |
| random_crop   | 0.8080          | 0.8912       | 0.5600           | 0.5823        |
| occlusion     | 0.9227          | 0.9774       | 0.9200           | 0.9241        |
| geom_warp     | 0.7973          | 0.8822       | 0.5333           | 0.5570        |
| train_aug_mix | 0.7280          | 0.7926       | 0.5533           | 0.5759        |

---

### Analysis

1. **Overall robustness**  
   Our model consistently outperforms PSNR across nearly all attacks.  
   The PSNR detector collapses under pixel-level distortions such as compression, blur, and rescaling, while our model still maintains meaningful separation (AUROC ≥ 0.7–0.9).

2. **Compression and messaging attacks**  
   - `jpeg_strong`: our model achieves 0.75 Accuracy vs. 0.58 for PSNR.  
   This result confirm that simple PSNR-based similarity measures fail to capture watermark integrity once strong compression artifacts appear.

3. **Spatial and geometric perturbations**  
   Our model remains stable under `random_crop` (0.89 AUROC) and `geom_warp` (0.88 AUROC), while PSNR drops near 0.55–0.58 AUROC.  
   This indicates that the learned watermark representation generalizes spatially rather than relying on raw pixel alignment.

4. **Occlusion resilience**  
   Both models perform well under `occlusion` (≈0.92 Accuracy), showing that the watermark signal is spatially redundant—removing part of the image does not destroy detectability.

5. **Visual degradations**  
   Under `blur`, PSNR fails entirely (AUROC = 0.5), while our model still achieves 0.70 AUROC.  
   This demonstrates that our watermark embedding resides in frequency or structural features that are partially invariant to smoothing.

6. **Clean and mild augmentations**  
   On clean images, both models perform nearly perfectly.  
   However, when realistic transformations are introduced (`train_aug_mix`), our model still holds up (0.79 AUROC) whereas PSNR quickly loses reliability (0.58 AUROC).

---

### Conclusion

- The **PSNR baseline**, using a sigmoid threshold of –4, can detect extreme pixel changes but fails to recognize the presence of the watermark once the image undergoes typical real-world transformations.  
- The **learned detector (Ours)** consistently preserves high separability even under aggressive corruptions, confirming that it captures watermark-specific features rather than superficial image fidelity.
- This robustness is critical for real-world deployment — where images are frequently recompressed, reframed, or blurred — and demonstrates the superiority of learned watermark detectors over static PSNR-based metrics.
