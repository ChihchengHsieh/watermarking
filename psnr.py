import matplotlib.pyplot as plt
import math
from sklearn.metrics import accuracy_score, roc_auc_score

try:
    from scipy.stats import ncx2
except Exception as e:
    ncx2 = None
    _scipy_import_error = e

# requires: numpy, torch, matplotlib (for plotting), scipy
import numpy as np
import torch

try:
    from scipy.stats import ncx2
except Exception as e:
    ncx2 = None
    _scipy_import_error = e


from torchvision import transforms


def transform_img(image, target_size=512):
    tform = transforms.Compose(
        [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float32),
        ]
    )
    image = tform(image)
    return 2.0 * image - 1.0


def _compute_psnr_masked(pred, target, mask, eps=1e-8):
    """
    Fallback PSNR on masked region: 10*log10(MAX^2 / MSE_masked).
    Assumes inputs in [0,1] or any bounded range; uses dynamic MAX from target.
    """
    # restrict to mask
    diff = (pred - target)[mask]
    mse = (diff.float() ** 2).mean().clamp_min(eps)
    # dynamic range from target over mask
    t = target[mask].float()
    max_val = (t.max() - t.min()).clamp_min(eps)
    # if target is constant, fall back to 1.0 range
    if max_val.item() == 0.0:
        max_val = torch.tensor(1.0, device=t.device)
    psnr = 10.0 * torch.log10((max_val**2) / mse)
    return psnr.item()


def eval_watermark(
    latent: torch.Tensor,
    watermarking_mask: torch.Tensor,
    gt_patch: torch.Tensor,
    w_measurement: str,
) -> float:
    """
    Evaluate watermark quality for a SINGLE latent against gt_patch on watermarking_mask.

    Args:
        latent:              Tensor [..., H, W] or [..., C, H, W] (last two dims are H,W).
        watermarking_mask:   Bool/byte mask broadcastable to latent/gt_patch spatial dims.
        gt_patch:            Ground-truth pattern tensor, same shape as latent (or broadcastable).
        w_measurement:       e.g. "complex+l1", "seed+l1", "complex+psnr", "seed+psnr".

    Returns:
        metric: float scalar.
    """
    # choose domain
    if "complex" in w_measurement:
        # FFT over spatial dims (H, W); shift to center
        latent_dom = torch.fft.fftshift(torch.fft.fft2(latent), dim=(-1, -2))
        target_dom = gt_patch
    elif "seed" in w_measurement:
        # real domain
        latent_dom = latent
        target_dom = gt_patch
    else:
        raise NotImplementedError(
            f"w_measurement domain not recognized: {w_measurement}"
        )

    # ensure mask is boolean and broadcastable to spatial dims
    wm_mask = watermarking_mask.bool()

    # metric
    if "l1" in w_measurement:
        metric = torch.abs(latent_dom[wm_mask] - target_dom[wm_mask]).mean().item()
    elif "psnr" in w_measurement:
        # try user's compute_psnr first (signature: (pred, target, mask))
        if "compute_psnr" in globals() and callable(globals()["compute_psnr"]):
            metric = globals()["compute_psnr"](latent_dom, target_dom, wm_mask)
        else:
            metric = _compute_psnr_masked(latent_dom, target_dom, wm_mask)
    else:
        raise NotImplementedError(
            f"w_measurement metric not recognized: {w_measurement}"
        )
    return metric


def _ensure_mask_shape(mask, ref_tensor):
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask)
    mask = mask.bool()
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # 1,1,H,W
    if mask.shape != ref_tensor.shape:
        mask = mask.expand(ref_tensor.shape)
    return mask


def detect_watermark_ncx2_from_latent(
    latent_tensor,  # torch tensor (B,C,H,W)
    gt_patch,  # key (either in freq domain or spatial)
    watermarking_mask,  # boolean mask selecting frequency bins M (H,W or B,C,H,W)
    k_in_freq_domain=True,
    alpha=0.01,
    eps=1e-12,
    verbose=True,
):

    l1_metric = eval_watermark(
        latent_tensor, watermarking_mask, gt_patch, w_measurement="l1_complex"
    )
    psnr_metric = eval_watermark(
        latent_tensor, watermarking_mask, gt_patch, w_measurement="psnr_complex"
    )

    if ncx2 is None:
        raise RuntimeError(
            "scipy required for noncentral chi2. Import error: %s"
            % (_scipy_import_error,)
        )

    # Ensure batch dim
    if latent_tensor.ndim == 3:
        latent_tensor = latent_tensor.unsqueeze(0)
    B, C, H, W = latent_tensor.shape
    if B != 1 and verbose:
        print("Warning: only processing first sample of batch")
        latent_tensor = latent_tensor[0:1]

    device = latent_tensor.device

    # compute y_fft in standard convention (fft2 + fftshift)
    # ensure we do FFT on float32 inputs so output complex dtype becomes complex64
    y_input = latent_tensor
    if y_input.dtype == torch.float16 or y_input.dtype == torch.bfloat16:
        y_input = y_input.to(torch.float32)
    y_fft = torch.fft.fftshift(
        torch.fft.fft2(y_input), dim=(-1, -2)
    )  # complex dtype (likely complex64)

    # prepare key in Fourier domain
    if k_in_freq_domain:
        k_fft = gt_patch
        # If user passed a float tensor rather than complex, convert to fft
        if not torch.is_complex(k_fft):
            if k_fft.ndim == 3:
                k_fft = k_fft.unsqueeze(0)
            k_tmp = k_fft
            if k_tmp.dtype == torch.float16 or k_tmp.dtype == torch.bfloat16:
                k_tmp = k_tmp.to(torch.float32)
            k_fft = torch.fft.fftshift(torch.fft.fft2(k_tmp), dim=(-1, -2))
    else:
        k_tmp = gt_patch
        if k_tmp.ndim == 3:
            k_tmp = k_tmp.unsqueeze(0)
        if k_tmp.dtype == torch.float16 or k_tmp.dtype == torch.bfloat16:
            k_tmp = k_tmp.to(torch.float32)
        k_fft = torch.fft.fftshift(torch.fft.fft2(k_tmp), dim=(-1, -2))

    # ensure shapes align (replicate channels if necessary)
    if k_fft.shape != y_fft.shape:
        if (
            k_fft.shape[0] == 1
            and y_fft.shape[0] == 1
            and k_fft.shape[1] == 1
            and y_fft.shape[1] > 1
        ):
            k_fft = k_fft.repeat(1, y_fft.shape[1], 1, 1)
        else:
            raise ValueError(
                f"Shape mismatch y_fft {y_fft.shape} vs k_fft {k_fft.shape}"
            )

    # Convert complex-half types to complex64 before moving to CPU / numpy
    if (
        y_fft.is_complex()
        and y_fft.dtype != torch.complex64
        and y_fft.dtype != torch.complex128
    ):
        # prefer complex64
        y_fft = y_fft.to(torch.complex64)
    if (
        k_fft.is_complex()
        and k_fft.dtype != torch.complex64
        and k_fft.dtype != torch.complex128
    ):
        k_fft = k_fft.to(torch.complex64)

    # prepare mask on CPU for indexing
    mask = _ensure_mask_shape(watermarking_mask, y_fft)  # boolean on same shape
    mask_cpu = mask.detach().cpu()

    # Move fft tensors to CPU (complex64) for numpy/scipy interaction
    y_cpu = y_fft.detach().cpu()
    k_cpu = k_fft.detach().cpu()

    # select masked elements and flatten
    y_sel = y_cpu[mask_cpu].numpy().astype(np.complex128).ravel()
    k_sel = k_cpu[mask_cpu].numpy().astype(np.complex128).ravel()

    M = y_sel.size
    if M == 0:
        raise ValueError("Empty watermark mask (|M|=0)")

    # compute sigma2, eta, lam in numpy (scalars)
    sigma2 = float(np.mean(np.abs(y_sel) ** 2))
    sigma2 = max(sigma2, eps)
    eta = (1.0 / sigma2) * float(np.sum(np.abs(k_sel - y_sel) ** 2))
    lam = (1.0 / sigma2) * float(np.sum(np.abs(k_sel) ** 2))

    # p-value and threshold via non-central chi2
    p_value = float(ncx2.cdf(eta, df=M, nc=lam))
    try:
        threshold_eta = float(ncx2.ppf(alpha, df=M, nc=lam))
    except Exception:
        threshold_eta = None

    detected = p_value <= alpha

    if verbose:
        print(
            f"ncx2 test: M={M}, sigma2={sigma2:.4e}, eta={eta:.4f}, lam={lam:.4f}, p={p_value:.4e}, alpha={alpha}, detected={detected}"
        )

    return dict(
        p_value=p_value,
        eta=eta,
        threshold_eta=threshold_eta,
        detected=bool(detected),
        sigma2=sigma2,
        M=int(M),
        lam=lam,
        alpha=float(alpha),
        l1_metric=float(l1_metric),
        psnr_metric=float(psnr_metric),
    )


# ---------------------------
# Full integrated verify function
# ---------------------------
def verify_with_ncx2(
    image,
    pipe,
    text_embeddings,
    watermarking_mask,
    gt_patch,
    num_inference_steps,
    alpha=0.01,
    k_in_freq_domain=True,
    show_visual=True,
    device=None,
):
    """
    Integrates the paper's ncx2 test into your verify routine.
    Parameters:
      - image: PIL or np array as before
      - pipe: your diffusion pipeline with get_image_latents and forward_diffusion
      - text_embeddings: embeddings used in forward_diffusion call (if needed)
      - watermarking_mask: boolean mask in frequency domain (H,W) or (1,C,H,W)
      - gt_patch: your key (if k_in_freq_domain True, gt_patch must already be in FFT domain or same shape)
      - num_inference_steps: used when calling forward_diffusion (your pipeline)
      - alpha: significance level (paper uses 0.01)
      - k_in_freq_domain: whether gt_patch is already Fourier (True by default)
    Returns:
      - results dict with detection & metrics
    """
    if device is None:
        device = (
            next(pipe.unet.parameters()).device
            if hasattr(pipe, "unet")
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # show image
    if show_visual:
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.axis("off")
        plt.show()

    # to tensor and move to dtype/device expected by pipeline
    tsr_img = (
        transform_img(image)
        .unsqueeze(0)
        .to(next(pipe.unet.parameters()).dtype)
        .to(device)
    )
    print(tsr_img.shape, tsr_img.dtype)  # torch.Size([1, 3, 512, 512]) torch.float16
    # encode to image latents (user's helper)
    image_latents = pipe.get_image_latents(tsr_img, sample=False)
    print(
        image_latents.shape, image_latents.dtype
    )  # torch.Size([1, 4, 64, 64]) torch.float16

    reversed_latents = pipe.forward_diffusion(
        latents=image_latents,
        text_embeddings=text_embeddings,
        guidance_scale=1,
        num_inference_steps=num_inference_steps,
    )

    # Visualise in Fourier domain (keep as complex)
    vis_latent_fft = torch.fft.fftshift(torch.fft.fft2(reversed_latents), dim=(-1, -2))

    if show_visual:
        # display magnitude for first channel
        mag = vis_latent_fft[0, 0].abs().detach().cpu().numpy()
        plt.figure(figsize=(4, 4))
        plt.title("FFT magnitude (reversed_latents)")
        plt.imshow(np.log1p(mag), cmap="magma")
        plt.axis("off")
        plt.show()

    # compute l1/psnr metrics if you still want them (assuming eval_watermark accepts complex tensors)
    l1_metric = None
    psnr_metric = None
    try:
        l1_metric = eval_watermark(
            reversed_latents, watermarking_mask, gt_patch, w_measurement="l1_complex"
        )
        psnr_metric = eval_watermark(
            reversed_latents, watermarking_mask, gt_patch, w_measurement="psnr_complex"
        )
    except Exception as e:
        # not fatal — just print
        print("Warning: eval_watermark failed:", e)

    # run statistical test (paper)
    stats = detect_watermark_ncx2_from_latent(
        latent_tensor=reversed_latents,
        gt_patch=gt_patch,
        watermarking_mask=watermarking_mask,
        k_in_freq_domain=k_in_freq_domain,
        alpha=alpha,
        verbose=True,
    )

    # nicely return all info
    results = dict(
        l1_metric=l1_metric,
        psnr_metric=psnr_metric,
        ncx2_stats=stats,
    )
    return results


def psnr_to_prob_sigmoid(psnr, threshold=-4.0, scale=1.0):
    """Simple sigmoid mapping. threshold -> p=0.5, smaller scale -> steeper curve."""
    return 1.0 / (1.0 + math.exp(-(psnr - threshold) / scale))


def detector(x, gt_patch, watermarking_mask):
    result = detect_watermark_ncx2_from_latent(
        latent_tensor=x,
        gt_patch=gt_patch,
        watermarking_mask=watermarking_mask,
        k_in_freq_domain=True,
        alpha=0.01,
        verbose=True,
    )

    # detected = result["detected"]
    # detected = bool(result['psnr_metric'] > -4.0)
    detected = psnr_to_prob_sigmoid(result["psnr_metric"], threshold=-4.0, scale=1.0)
    # turn into probability-like output
    return detected


def batch_detector(x_batch, gt_patch, watermarking_mask):
    results = []
    for i in range(x_batch.shape[0]):
        res = detect_watermark_ncx2_from_latent(
            latent_tensor=x_batch[i : i + 1],
            gt_patch=gt_patch,
            watermarking_mask=watermarking_mask,
            k_in_freq_domain=True,
            alpha=0.01,
            verbose=False,
        )
        detected = psnr_to_prob_sigmoid(res["psnr_metric"], threshold=-4.0, scale=1.0)
        results.append(float(detected))
        # results.append(float(res["detected"]))
    return torch.tensor(results, device=x_batch.device)


from tqdm import tqdm


def eval_watermark_detector(detector, loader, device):
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="eval", leave=False):
            X = X.to(device)
            y = y.to(device)
            out = detector(X)
            pred = (out > 0.5).cpu().numpy().astype(int).tolist()
            probs.extend(pred)
            preds.extend(pred)
            trues.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = float("nan")
    return acc, auc


def eval_watermark_detector_pnsr(detector, loader, device):
    trues, preds, probs = [], [], []
    with torch.no_grad():
        for X, y in tqdm(loader, desc="eval", leave=False):
            X = X.to(device)
            y = y.to(device)
            out = detector(X)
            pred = (out > 0.5).cpu().numpy().astype(int).tolist()
            # probs.extend(out.cpu().numpy().tolist())
            probs.extend(pred)
            preds.extend(pred)
            trues.extend(y.cpu().numpy().tolist())
    acc = accuracy_score(trues, preds)
    try:
        auc = roc_auc_score(trues, probs)
    except Exception:
        auc = float("nan")
    return acc, auc
