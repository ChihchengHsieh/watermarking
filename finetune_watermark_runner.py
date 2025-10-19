#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
finetune_watermark_runner.py (no _resolve_defaults version)

Overview
--------
This script provides a **drop-in, instrumented wrapper** around a diffusion-model
fine-tuning and watermark-detection pipeline. It prints progress messages and
wall-clock timings for each major block so you can quickly see **where a run
stops or slows down**. It also exposes a simple CLI for configuring common
hyperparameters.

Watermarking method (brief, practical explainer)
-----------------------------------------------
The pipeline assumes a **spectral watermark** is embedded in generated images
and is detectable by analysing the **average sample latent/image** at a specific
diffusion time **t_A** (sometimes called the “watermark capture time”).
Concretely:

1) **Fine-tune** your UNet (optionally with EMA).
2) Use a sampler (DDIM/DPMSolver/etc.) to draw several samples and compute the
   **per-pixel average** `x̄(t_A)` (optionally converted to `uint8` for display).
3) Transform `x̄(t_A)` into the frequency domain (e.g., FFT) and compute a
   **spectral correlation** score against a known watermark pattern/channel.
4) The correlation magnitude at **t_A** is the detection statistic
   (higher is stronger evidence of the watermark).
5) For robustness, verify on multiple **independent batches** of samples and
   report the fraction exceeding a threshold.

Aside from the detection statistic at **t_A**, the script can also sample final
images for visual inspection (quality eyeballing) and—optionally—compute image
quality metrics like IS/FID/Precision/Recall.

Note: This file focuses on **instrumentation** and CLI. It expects the core
project code to provide the following functions/objects:
    - _prepare_models_and_opt(train_scope, lr, use_ema)
    - _run_finetune(work_unet, opt, ema_unet, ema_decay, max_steps, gamma)
    - _detect_at_tA(sampler_unet, sample_steps, gamma)
    - sample_paper(sampler_unet, scheduler, n, steps, capture_tA, gamma)
    - _verify_batches(sampler_unet, sample_steps, gamma, batches, batch_size, threshold)
    - _build_summary_figure(max_steps, xavg, ch, lo, hi, grid_np, correlations, corr_threshold, corr_tA, extra)
and variables/types:
    - A scheduler object compatible with your sampler (passed via CLI or imported)
    - A scalar/time-step `tA` (capture step) used by `sample_paper` (passed via CLI)

If you already have `scheduler` and `tA` as globals in your project, you can
either (a) import them here, or (b) pass via CLI. The script will gracefully
skip final-sample generation if they’re not supplied.

Usage
-----
Example (with defaults):
    python finetune_watermark_runner.py --max-steps 500

Example (providing watermark threshold, disabling metrics):
    python finetune_watermark_runner.py --max-steps 1000 --contour-threshold 0.08 --no-metrics

Example (providing capture tA and enabling sample grid):
    python finetune_watermark_runner.py --max-steps 500 --tA 250 --enable-samples

This will print per-block timings and write a summary of timings to stdout.
"""

import argparse
import importlib
import math
import sys
import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid  # make sure torchvision is installed


# -----------------------------
# Placeholders for project API
# -----------------------------
# IMPORTANT: Replace these imports with your project's modules OR provide
#            --scheduler-module/--scheduler-name and --tA via CLI.
#
# Example if your project exposes these in a module named `wm_core`:
# from wm_core import (
#     _prepare_models_and_opt,
#     _run_finetune,
#     _detect_at_tA,
#     _verify_batches,
#     _build_summary_figure,
#     sample_paper,
# )
#
# For safety, we keep them as **undeclared** here on purpose, so that any missing
# dependency will raise a clear NameError pointing you to wire them up.

# ---------- Full fine-tuning loop ----------
def _run_finetune(
    work_unet, opt, ema_unet, ema_decay, max_steps: int, gamma: float
):
    """
    Fine-tune the UNet with biased batches (Algorithm 1) for 'max_steps'.
    This is the main training loop used before we evaluate watermark detectability.
    """
    work_unet.train()
    pbar = tqdm(range(max_steps), desc=f"Fine-tuning (Alg.1, f1=0) {max_steps} steps")
    data_iter = iter(train_loader)
    loss_traj = []

    for _ in pbar:
        # Get next batch (recycle loader if needed)
        try:
            x0, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            x0, _ = next(data_iter)

        x0 = x0.to(device)
        B = x0.size(0)
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)

        # Build biased input/target for current timestep
        x_t_prime, eps_dblprime = _build_biased_batch(x0, t, gamma)

        # Forward + loss
        noise_pred = work_unet(x_t_prime, t, return_dict=False)[0]
        loss =  torch.nn.functional.mse_loss(noise_pred, eps_dblprime)

        # Backprop and update
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(work_unet.parameters(), 1.0)
        opt.step()

        # EMA update (optional)
        if ema_unet is not None:
            with torch.no_grad():
                for pe, p in zip(ema_unet.parameters(), work_unet.parameters()):
                    pe.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)

        loss_val = float(loss.detach().cpu())
        loss_traj.append(loss_val)
        pbar.set_postfix(loss=loss_val)

    # Return EMA model if available (for sampling), else the fine-tuned one
    sampler_unet = ema_unet if (ema_unet is not None and channels == 3) else work_unet
    sampler_unet.eval()
    return sampler_unet, loss_traj

# ---------- Small config resolver ----------
def _resolve_defaults(
    lr, gamma, use_ema, verify_batches, verify_batch_size, sample_steps,
    n_final_samples, contour_threshold  # kept name for backward-compat; now used as corr_threshold
):
    lr = cfg.lr if lr is None else lr
    gamma = cfg.gamma if gamma is None else gamma
    use_ema = (cfg.use_ema and channels == 3) if use_ema is None else use_ema
    verify_batches = cfg.verify_batches if verify_batches is None else verify_batches
    verify_batch_size = cfg.verify_batch_size if verify_batch_size is None else verify_batch_size
    sample_steps = cfg.sample_steps if sample_steps is None else sample_steps
    n_final_samples = cfg.n_final_samples if n_final_samples is None else n_final_samples
    corr_threshold = cfg.contour_threshold if contour_threshold is None else contour_threshold  # reinterpret as correlation threshold
    return (lr, gamma, use_ema, verify_batches, verify_batch_size, sample_steps,
            n_final_samples, corr_threshold)

# ---------- Model/opt/EMA prep ----------
def _prepare_models_and_opt(train_scope: str, lr: float, use_ema: bool):
    """Deep-copy baseline, unfreeze the requested scope, create optimizer and optional EMA."""
    work_unet = copy.deepcopy(baseline_unet).to(device)
    set_trainable(work_unet, mode=train_scope)

    # Optimizer over trainable params
    for p in work_unet.parameters():
        if p.requires_grad:
            p.requires_grad = True
    opt = torch.optim.AdamW(work_unet.parameters(), lr=lr, weight_decay=0.0)

    # Optional EMA for stable sampling
    ema_unet = None
    if use_ema:
        ema_unet = copy.deepcopy(work_unet).eval()
        for p in ema_unet.parameters():
            p.requires_grad = False
    ema_decay = 0.999
    return work_unet, opt, ema_unet, ema_decay

# ---------- One training step: build biased inputs/targets ----------
def _build_biased_batch(x0: torch.Tensor, t: torch.Tensor, gamma: float,
                        lam_clip: float = 0.02):
    """
    Algorithm 1 with Fourier-domain watermark injection:
      - For t <= tA: apply a small multiplicative magnitude modulation in FFT
        on the watermarked channel inside a mid-band annulus (ANNULUS) using PN.
      - For t > tA: construct a spectrally watermarked x'_{tA} once, then
        simulate forward to the desired t using ᾱ_t / ᾱ_{tA}.
    Target shift remains: ε'' = γ ε' + (1-γ) K x_A (for t <= tA).
    """
    import numpy as np

    B, C, H, W = x0.shape
    assert C > WTMK_CH, "WTMK_CH out of range for channels"

    # Base forward noising (simulate q(x_t | x0))
    eps_prime = torch.randn_like(x0)
    x_sim = scheduler.add_noise(x0, eps_prime, t)

    x_t_prime    = x_sim.clone()
    eps_dblprime = eps_prime.clone()

    # ---- helper: spectral magnitude modulation on one HxW array ----
    def _spectral_mod_one(x_hw: np.ndarray, lam: float) -> np.ndarray:
        Xi = np.fft.fftshift(np.fft.fft2(x_hw))
        mag, ph = np.abs(Xi), np.angle(Xi)
        # bounded multiplicative tweak (keep positivity)
        gain = np.ones_like(mag, dtype=np.float32)
        gain[ANNULUS] = (1.0 + np.clip(lam, -lam_clip, lam_clip) * PN[ANNULUS])
        mag2 = mag * gain
        Xi2 = np.fft.ifftshift(mag2 * np.exp(1j * ph))
        x_mod = np.fft.ifft2(Xi2).real.astype(np.float32)
        return x_mod

    # ---------- EMBEDDING stage: t <= tA ----------
    embed = (t <= tA)
    if embed.any():
        idx = embed.nonzero(as_tuple=False).squeeze(1)
        sqrt_ab_t  = (a_bar.index_select(0, t[idx]).view(-1,1,1,1)).sqrt()
        sqrt_1m_t  = (1 - a_bar.index_select(0, t[idx]).view(-1,1,1,1)).sqrt()
        f2_t       = f2.index_select(0, t[idx]).view(-1,1,1,1)

        # Baseline xt without spatial bias
        base_xt = sqrt_ab_t * x0[idx] + gamma * sqrt_1m_t * eps_prime[idx]
        x_mod = base_xt.clone()

        # Per-sample modulation strength following f2(t)
        lam_vec = ((1.0 - gamma) * f2_t.view(-1)).clamp(min=0.0).cpu().numpy()

        ch = WTMK_CH
        x_np = x_mod[:, ch].detach().cpu().numpy().astype(np.float32)  # [Be,H,W]
        for i in range(x_np.shape[0]):
            x_np[i] = _spectral_mod_one(x_np[i], lam=float(lam_vec[i]))
        x_mod[:, ch] = torch.from_numpy(x_np).to(x_mod.device)

        x_t_prime[idx] = x_mod
        # Target shift (teach UNet the watermark direction)
        eps_dblprime[idx] = gamma * eps_prime[idx] + (1.0 - gamma) * (K * xA)

    # ---------- SIMULATION stage: t > tA ----------
    sim = ~embed
    if sim.any():
        idx = sim.nonzero(as_tuple=False).squeeze(1)
        B2 = idx.numel()
        tA_tensor = torch.full((B2,), tA, device=x0.device, dtype=torch.long)

        sqrt_ab_tA  = (a_bar.index_select(0, tA_tensor).view(-1,1,1,1)).sqrt()
        sqrt_1m_tA  = (1 - a_bar.index_select(0, tA_tensor).view(-1,1,1,1)).sqrt()
        f2_tA       = f2.index_select(0, tA_tensor).view(-1,1,1,1)

        # Build x'_{tA} and apply spectral modulation at tA
        eps_prime_tA = torch.randn_like(x0[idx])
        x_tA_base = sqrt_ab_tA * x0[idx] + gamma * sqrt_1m_tA * eps_prime_tA
        x_tA_mod  = x_tA_base.clone()

        lam_vecA = ((1.0 - gamma) * f2_tA.view(-1)).clamp(min=0.0).cpu().numpy()
        ch = WTMK_CH
        x_np = x_tA_mod[:, ch].detach().cpu().numpy().astype(np.float32)
        for i in range(x_np.shape[0]):
            x_np[i] = _spectral_mod_one(x_np[i], lam=float(lam_vecA[i]))
        x_tA_mod[:, ch] = torch.from_numpy(x_np).to(x_tA_mod.device)

        # Simulate distributionally from tA -> t
        ratio = (a_bar.index_select(0, t[idx]).view(-1,1,1,1) / a_bar[tA]).clamp(min=1e-12)
        x_t_prime[idx] = ratio.sqrt() * x_tA_mod + (1.0 - ratio).sqrt() * eps_prime[idx]

        # No target shift for t>tA
        eps_dblprime[idx] = eps_prime[idx]

    return x_t_prime, eps_dblprime

# ---------- Detection: average at t_A & correlation (↑ better) ----------
@torch.no_grad()
def _detect_at_tA(sampler_unet, sample_steps: int, gamma: float):
    """
    Return:
      xavg: [1,C,H,W] float in [0,1]
      xavg_u8: [C,H,W] uint8
      ch: watermark channel as np.float32
      (lo, hi): percentiles for heatmap
      dist_mask_dummy: kept for API compatibility (unused here)
      corr_tA: spectral correlation score (↑ better)
    """
    xmid = _sample_mid_after(sampler_unet, scheduler, n=cfg.verify_batch_size,
                             steps=sample_steps, capture_tA=tA, gamma=gamma)
    xavg = xmid.mean(dim=0, keepdim=True)  # [1,C,H,W] in [0,1]
    xavg_u8 = to_u8(xavg)[0]
    ch = xavg[0, WTMK_CH].cpu().numpy().astype(np.float32)
    lo, hi = np.percentile(ch, [5, 95]); 
    if hi <= lo: hi = lo + 1e-3

    # Compute spectral correlation (expects float [C,H,W] in [0,1])
    corr_tA = float(spectral_correlation_score(xavg[0], PN, ANNULUS))
    return xavg, xavg_u8, ch, (lo, hi), None, corr_tA

# ---------- Verification: multiple averages under paper-exact sampling ----------
@torch.no_grad()
def _verify_batches(sampler_unet, sample_steps: int, gamma: float,
                    batches: int, batch_size: int, threshold: float):
    """
    Repeat averaging at t_A and compute spectral correlation across 'batches' trials.
    Success := correlation >= threshold  (since ↑ is better).
    """
    corrs = []
    for _ in range(batches):
        _, xmid_ = sample_paper(sampler_unet, scheduler, n=batch_size,
                                steps=sample_steps, capture_tA=tA, gamma=gamma)
        xavg_ = xmid_.mean(dim=0, keepdim=True)
        corr = float(spectral_correlation_score(xavg_[0], PN, ANNULUS))
        corrs.append(corr)
    arr = np.array(corrs, dtype=np.float32)
    success = float((arr >= threshold).mean())
    return arr, success

# ---------- Visualization ----------
def _build_summary_figure(max_steps: int, xavg, ch, lo, hi,
                          grid_np, correlations, corr_threshold, corr_tA, _unused):
    """
    Compose the one-page report figure.
    Note: Plot shows spectral correlation (↑ better). Threshold line = minimum acceptable correlation.
    """
    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax5 = fig.add_subplot(gs[1, 2])

    # (1) Avg RGB @ t_A
    ax1.imshow(xavg[0].permute(1, 2, 0).cpu().numpy())
    ax1.set_title(f"Avg @ t_A (RGB) — steps={max_steps}")
    ax1.axis("off")

    # (2) Watermark channel heatmap
    im2 = ax2.imshow(ch, cmap="magma", vmin=lo, vmax=hi)
    ax2.set_title(f"Watermark channel (ch={WTMK_CH})")
    ax2.axis("off")
    cbar = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # (3) Simple magnitude spectrum (for intuition, optional)
    F = np.fft.fftshift(np.fft.fft2(ch))
    ax3.imshow(np.log1p(np.abs(F)), cmap="inferno")
    ax3.set_title("Log magnitude spectrum")
    ax3.axis("off")

    # (4) Final sample grid
    ax4.imshow(grid_np)
    ax4.set_title("Watermarked samples (paper-exact)")
    ax4.axis("off")

    # (5) Spectral correlation across verification batches
    xs = np.arange(len(correlations))
    ax5.plot(xs, correlations, marker='o')
    ax5.axhline(corr_threshold, linestyle='--', label='threshold')
    ax5.set_xlabel("Batch")
    ax5.set_ylabel("Spectral correlation (↑ better)")
    ax5.set_title(f"Verification @ t_A\nsuccess={(correlations >= corr_threshold).mean():.2f}")
    ax5.legend(loc="best", fontsize=8)

    fig.text(0.01, 0.02, f"corr@tA={corr_tA:.4f}", fontsize=10)
    fig.tight_layout()
    return fig

# ---------- Optional: quality metrics (IS/FID/Precision/Recall) ----------
def _compute_quality_metrics_if_requested(
    compute_metrics: bool, sampler_unet, sample_steps: int, gamma: float,
    metrics_num_gen: int, metrics_num_real: int, metrics_splits: int
):
    metrics = {"IS": None, "FID": None, "Precision": None, "Recall": None}
    if not compute_metrics:
        return metrics

    try:
        real_u8 = _collect_real_u8(train_loader, metrics_num_real)     # [Nr,C,H,W] uint8 CPU
        gen_u8  = _generate_u8(sampler_unet, scheduler, metrics_num_gen,
                               steps=sample_steps, gamma=gamma)         # [Nf,C,H,W] uint8 CPU
    except Exception as e:
        print(f"[warn] collecting real/gen failed: {e}")
        return metrics

    # Inception Score
    try:
        from torchmetrics.image.inception import InceptionScore
        # 'splits' vs 'num_splits' varies by version; prefer 'num_splits'
        is_metric = InceptionScore(num_splits=metrics_splits, normalize=False).to(device)
        for i in range(0, gen_u8.shape[0], 128):
            is_metric.update(gen_u8[i:i+128].to(device))
        IS_mean, IS_std = is_metric.compute()
        metrics["IS"] = (float(IS_mean), float(IS_std))
    except Exception as e:
        print(f"[warn] IS failed: {e}")

    # FID
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(normalize=False).to(device)
        for i in range(0, real_u8.shape[0], 128):
            fid.update(real_u8[i:i+128].to(device), real=True)
        for i in range(0, gen_u8.shape[0], 128):
            fid.update(gen_u8[i:i+128].to(device), real=False)
        metrics["FID"] = float(fid.compute().item())
    except Exception as e:
        print(f"[warn] FID failed: {e}")

    # Precision/Recall
    try:
        feats_real = _inception_feats_u8(real_u8, device=device, batch=128)
        feats_fake = _inception_feats_u8(gen_u8,  device=device, batch=128)
        P, R = _precision_recall_knn(feats_real, feats_fake, k=3, chunk=1024, device=device)
        metrics["Precision"] = float(P)
        metrics["Recall"]    = float(R)
    except Exception as e:
        print(f"[warn] Precision/Recall failed: {e}")

    return metrics

# -----------------------------
# Instrumented core function
# -----------------------------
def finetune_and_report(
    max_steps: int,
    *,
    lr: Optional[float] = None,
    gamma: Optional[float] = None,
    use_ema: Optional[bool] = None,
    verify_batches: Optional[int] = None,
    verify_batch_size: Optional[int] = None,
    sample_steps: Optional[int] = None,
    n_final_samples: Optional[int] = None,
    contour_threshold: Optional[float] = None,   # interpreted as correlation threshold
    return_models: bool = False,
    show: bool = True,
    train_scope: str = "all",          # {"all","last_conv","last_block","last_two_blocks"}
    compute_metrics: bool = True,
    metrics_num_gen: int = 2048,
    metrics_num_real: int = 2048,
    metrics_splits: int = 10,
    # New: optionally pass scheduler/tA to avoid relying on globals.
    scheduler: Optional[Any] = None,
    tA: Optional[int] = None,
) -> Tuple[Dict[str, Any], Any]:
    """
    Instrumented version: prints progress + wall-clock time for each major block.
    If any block errors, the function prints the step name and re-raises the exception.
    Returns (result_dict, figure).
    """
    def _now() -> float:
        return time.perf_counter()

    def _print_step_header(step_name: str):
        print(f"\n[STEP] {step_name} ...", flush=True)

    def _print_step_done(step_name: str, t_start: float, t0_global: float):
        dt = _now() - t_start
        dt_total = _now() - t0_global
        print(f"[DONE] {step_name} in {dt:.2f}s (cumulative {dt_total:.2f}s)", flush=True)

    t0_global = _now()
    print(f"=== finetune_and_report start (max_steps={max_steps}) ===", flush=True)

    timings: Dict[str, float] = {}

    # 1) Prepare models/opt/EMA
    step = "1) Prepare models/optimizer/EMA"
    _print_step_header(step)
    t = _now()
    try:
        work_unet, opt, ema_unet, ema_decay = _prepare_models_and_opt(train_scope, lr, use_ema)
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 2) Fine-tune
    step = "2) Run fine-tuning"
    _print_step_header(step)
    t = _now()
    try:
        sampler_unet, loss_traj = _run_finetune(work_unet, opt, ema_unet, ema_decay, max_steps, gamma)
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 3) Detect watermark at t_A (avg) -> spectral correlation
    step = "3) Detect watermark at t_A (spectral correlation)"
    _print_step_header(step)
    t = _now()
    try:
        xavg, xavg_u8, ch, (lo, hi), _mask_unused, corr_tA = _detect_at_tA(
            sampler_unet, sample_steps=sample_steps, gamma=gamma
        )
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 4) Final samples (quality eyeballing)
    step = "4) Generate final samples"
    _print_step_header(step)
    t = _now()
    try:
        grid_np = None
        imgs_wtmk = None
        if (scheduler is not None) and (tA is not None) and (n_final_samples and n_final_samples > 0):
            imgs_wtmk, _ = sample_paper(
                sampler_unet, scheduler, n=n_final_samples,
                steps=sample_steps, capture_tA=tA, gamma=gamma
            )
            grid = make_grid(imgs_wtmk, nrow=int(math.sqrt(n_final_samples)), padding=2)
            grid_np = (grid.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        else:
            print("[WARN] Skipping final-sample grid: scheduler/tA not provided or n_final_samples <= 0.", flush=True)
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 5) Verification batches (Spectral correlation ONLY)
    step = "5) Verify batches (spectral correlation)"
    _print_step_header(step)
    t = _now()
    try:
        correlations, success = _verify_batches(
            sampler_unet, sample_steps=sample_steps, gamma=gamma,
            batches=verify_batches, batch_size=verify_batch_size,
            threshold=contour_threshold
        )
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 6) Figure
    step = "6) Build summary figure"
    _print_step_header(step)
    t = _now()
    try:
        fig = _build_summary_figure(
            max_steps, xavg, ch, lo, hi, grid_np, correlations,
            contour_threshold, corr_tA, None
        )
        if show:
            plt.show()
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 7) Optional quality metrics (IS/FID/Precision/Recall)
    step = "7) Compute quality metrics (optional)"
    _print_step_header(step)
    t = _now()
    try:
        metrics = _compute_quality_metrics_if_requested(
            compute_metrics, sampler_unet, sample_steps, gamma,
            metrics_num_gen, metrics_num_real, metrics_splits
        )
        metrics = metrics or {}
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)
    timings[step] = _now() - t

    # 8) Results dict (+ optional models)
    step = "8) Assemble results"
    _print_step_header(step)
    t = _now()
    try:
        result: Dict[str, Any] = {
            "max_steps": max_steps,
            "loss_traj": loss_traj,
            "xavg_tA": xavg,                         # [1,C,H,W] in [0,1]
            "samples": imgs_wtmk if 'imgs_wtmk' in locals() else None,
            "spectral_corr_tA": corr_tA,             # ↑ better
            "verify_correlations": correlations,     # np.array of correlations
            "verify_success": float(success),        # fraction >= threshold
            **metrics,                               # IS/FID/Precision/Recall (may be None)
            "timings": timings,                      # include raw per-step timings
            "total_time_sec": _now() - t0_global,    # total runtime
        }
    except Exception as e:
        print(f"[ERROR] while {step}: {e}\n{traceback.format_exc()}", flush=True)
        raise
    _print_step_done(step, t, t0_global)

    total = _now() - t0_global
    print(f"\n=== finetune_and_report done in {total:.2f}s ===", flush=True)
    return result, fig


# -----------------------------
# Command-line interface (CLI)
# -----------------------------
def _maybe_import_scheduler(module_name: Optional[str], obj_name: Optional[str]) -> Optional[Any]:
    """Optionally import a scheduler object by module and attribute name."""
    if not module_name or not obj_name:
        return None
    mod = importlib.import_module(module_name)
    return getattr(mod, obj_name)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Instrumented diffusion fine-tune & watermark detection runner.")
    # Core training steps
    p.add_argument("--max-steps", type=int, required=True, help="Number of fine-tuning steps.")
    # Hyperparameters (defaults handled here; set to None to defer to project-side defaults)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (None -> project default).")
    p.add_argument("--gamma", type=float, default=None, help="Gamma / guidance parameter (None -> project default).")
    # EMA flags
    ema_group = p.add_mutually_exclusive_group()
    ema_group.add_argument("--use-ema", dest="use_ema", action="store_true", help="Force-enable EMA.")
    ema_group.add_argument("--no-ema", dest="use_ema", action="store_false", help="Force-disable EMA.")
    p.set_defaults(use_ema=None)  # If neither flag set, pass None (project decides)
    # Verification / sampling / thresholds
    p.add_argument("--verify-batches", type=int, default=10, help="Number of verification batches.")
    p.add_argument("--verify-batch-size", type=int, default=16, help="Verification batch size.")
    p.add_argument("--sample-steps", type=int, default=50, help="Sampler steps for generation/detection.")
    p.add_argument("--n-final-samples", type=int, default=16, help="Number of final samples to draw for a grid.")
    p.add_argument("--contour-threshold", type=float, default=0.05, help="Correlation threshold for success.")
    # Training scope
    p.add_argument("--train-scope", type=str, default="all",
                   choices=["all", "last_conv", "last_block", "last_two_blocks"],
                   help="Which layers to train.")
    # Quality metrics controls
    p.add_argument("--metrics-num-gen", type=int, default=2048, help="Generated samples for metrics.")
    p.add_argument("--metrics-num-real", type=int, default=2048, help="Real samples for metrics.")
    p.add_argument("--metrics-splits", type=int, default=10, help="Splits for metrics.")
    p.add_argument("--no-metrics", action="store_true", help="Disable IS/FID/Precision/Recall computation.")
    p.add_argument("--no-show", action="store_true", help="Disable plt.show().")
    # Optional scheduler/tA wiring
    p.add_argument("--scheduler-module", type=str, default=None,
                   help="Module path to import a scheduler from, e.g., 'myproj.schedulers'.")
    p.add_argument("--scheduler-name", type=str, default=None,
                   help="Attribute name of the scheduler object in the module, e.g., 'my_scheduler'.")
    p.add_argument("--tA", type=int, default=None, help="Capture step for watermark detection in sampling.")
    return p.parse_args()


def main():
    args = parse_args()

    # Optionally import scheduler object
    scheduler = _maybe_import_scheduler(args.scheduler_module, args.scheduler_name)

    # Run
    try:
        result, fig = finetune_and_report(
            args.max_steps,
            lr=args.lr,
            gamma=args.gamma,
            use_ema=args.use_ema,
            verify_batches=args.verify_batches,
            verify_batch_size=args.verify_batch_size,
            sample_steps=args.sample_steps,
            n_final_samples=args.n_final_samples,
            contour_threshold=args.contour_threshold,
            return_models=False,
            show=not args.no_show,
            train_scope=args.train_scope,
            compute_metrics=not args.no_metrics,
            metrics_num_gen=args.metrics_num_gen,
            metrics_num_real=args.metrics_num_real,
            metrics_splits=args.metrics_splits,
            scheduler=scheduler,
            tA=args.tA,
        )
        # Brief timing report
        print("\n=== Timing summary ===")
        timings = result.get("timings", {})
        for k, v in timings.items():
            print(f"{k:45s} {v:7.2f}s")
        print(f"{'TOTAL':45s} {result.get('total_time_sec', float('nan')):7.2f}s")
    except NameError as ne:
        print("\n[CONFIG ERROR] A required project function was not found.\n"
              "Please ensure your project provides:\n"
              "  _prepare_models_and_opt, _run_finetune, _detect_at_tA,\n"
              "  _verify_batches, _build_summary_figure, sample_paper\n"
              "and optionally pass --scheduler-module/--scheduler-name and --tA.\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(2)
    except Exception as e:
        print("\n[RUN ERROR] The script failed during execution:", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
