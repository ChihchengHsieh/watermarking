import torch.nn.functional as F

import torch


def compute_psnr(x, y, mask, max_val=1.0):
    x_real = x.real if torch.is_complex(x) else x
    y_real = y.real if torch.is_complex(y) else y
    mse = F.mse_loss(x_real[mask], y_real[mask])
    psnr = 10 * torch.log10(max_val**2 / mse)
    return psnr.item()


def eval_watermark(
    reversed_latents_no_w,
    reversed_latents_w,
    watermarking_mask,
    gt_patch,
    w_measurement: str,
):
    """
    Evaluate the watermarking performance based on the specified measurement method.

    Parameters:
        - reversed_latents_no_w: The latents without watermarking.
        - reversed_latents_w: The latents with watermarking.
        - watermarking_mask: The mask indicating the watermarking area.
        - gt_patch: The watermark pattern.
        - w_measurement: The method of watermark measurement (e.g., "l1", "complex").

    Returns:
        - no_w_metric: The measurement result for the latents without watermarking.
        - w_metric: The measurement result for the latents with watermarking.
    """

    if "complex" in w_measurement:
        """
        Measurement in FFT domain: It goes into the FFT to obtain the pattern.
        """

        reversed_latents_no_w_fft = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents_no_w), dim=(-1, -2)
        )

        reversed_latents_w_fft = torch.fft.fftshift(
            torch.fft.fft2(reversed_latents_w), dim=(-1, -2)
        )

        target_patch = gt_patch

    elif "seed" in w_measurement:
        """
        Measurement in real domain: It stays in the real domain to obtain the pattern.
        """

        reversed_latents_no_w_fft = reversed_latents_no_w

        reversed_latents_w_fft = reversed_latents_w

        target_patch = gt_patch

    else:

        raise NotImplementedError(f"w_measurement: {w_measurement}")

    if "l1" in w_measurement:
        """
        L1 measurement: compare the pattern in the real/FFT domain on the *mask*.
        """

        no_w_metric = (
            torch.abs(
                reversed_latents_no_w_fft[watermarking_mask]
                - target_patch[watermarking_mask]
            )
            .mean()
            .item()
        )

        w_metric = (
            torch.abs(
                reversed_latents_w_fft[watermarking_mask]
                - target_patch[watermarking_mask]
            )
            .mean()
            .item()
        )

        ### Implement PSNR

    elif "psnr" in w_measurement:

        def compute_psnr(x, y, mask, max_val=1.0):
            mse = F.mse_loss(x[mask], y[mask])
            psnr = 10 * torch.log10(max_val**2 / mse)
            return psnr.item()

        no_w_metric = compute_psnr(
            reversed_latents_no_w_fft, target_patch, watermarking_mask
        )
        w_metric = compute_psnr(reversed_latents_w_fft, target_patch, watermarking_mask)

    else:
        raise NotImplementedError(f"w_measurement: {w_measurement}")

    return no_w_metric, w_metric
