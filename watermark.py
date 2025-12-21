from torchvision import transforms  
import torch.nn.functional as F
import torch
import numpy as np
import math
import copy


def inject_watermark(init_latents_w, watermarking_mask, gt_patch, w_injection: str):
    """
    Inject the watermark into the latents.
    Parameters:
        - init_latents_w: The initial latents with watermarking.
        - watermarking_mask: The mask indicating where to inject the watermark. (in the Fourier domain)
        - gt_patch: The watermark pattern to be injected.
        - w_injection: The method of watermark injection (e.g., "complex", "seed").
    Returns:
        - init_latents_w: The latents with the watermark injected.
    """

    # Perform FFT on the latents
    init_latents_w_fft = torch.fft.fftshift(torch.fft.fft2(init_latents_w), dim=(-1, -2))

    # Inject the watermark into the latents in FFT domain
    if w_injection == 'complex':
        init_latents_w_fft[watermarking_mask] = gt_patch[watermarking_mask].clone()
    elif w_injection == 'seed':
        init_latents_w[watermarking_mask] = gt_patch[watermarking_mask].clone()
        return init_latents_w
    else:
        NotImplementedError(f'w_injection: {w_injection}')

    # Perform inverse FFT to get the latents with watermark
    init_latents_w = torch.fft.ifft2(torch.fft.ifftshift(init_latents_w_fft, dim=(-1, -2))).real

    return init_latents_w

def _draw_segment(canvas, x0, y0, x1, y1, thickness_px):
    """
    Mutates canvas[H,W] in-place (bool OR).
    (x0,y0)->(x1,y1) are in pixel coords: x = column index, y = row index.
    thickness_px is radius in pixels.
    """
    H, W = canvas.shape
    device = canvas.device
    dtype = torch.float32

    ys = torch.arange(H, device=device, dtype=dtype).view(H, 1)
    xs = torch.arange(W, device=device, dtype=dtype).view(1, W)

    ABx = x1 - x0
    ABy = y1 - y0
    AB_len2 = ABx * ABx + ABy * ABy + 1e-12

    APx = xs - x0
    APy = ys - y0

    t = (APx * ABx + APy * ABy) / AB_len2
    t = t.clamp(0.0, 1.0)

    projx = x0 + t * ABx
    projy = y0 + t * ABy

    dx = xs - projx
    dy = ys - projy
    dist = torch.sqrt(dx * dx + dy * dy + 1e-12)

    canvas |= dist <= thickness_px


def generate_regular_spiderweb(
    H: int,
    W: int,
    *,
    n_spokes: int = 8,
    n_levels: int = 10,
    thickness_px: float = 2.0,
    max_r_frac: float = 0.45,  # how large the web is relative to min(H,W)
    device=None,
    dtype=torch.float32,
):
    """
    Returns [1,1,H,W] float {0,1}:
    - hub at image centre
    - n_spokes straight radial spokes (evenly spaced 360/n_spokes apart)
    - n_levels straight connectors between adjacent spokes at equal normalized radii
    This matches the clean geometric web you asked for.
    """
    device = device or "cpu"

    canvas = torch.zeros((H, W), dtype=torch.bool, device=device)

    # hub in the middle of the canvas
    hub_x = (W - 1) / 2.0
    hub_y = (H - 1) / 2.0

    # evenly spaced spokes
    angles = torch.linspace(
        0,
        2 * math.pi,
        steps=n_spokes + 1,
        device=device,
        dtype=dtype,
    )[
        :-1
    ]  # shape [n_spokes]

    # use same length for every spoke so it's not skewed
    max_r = min(H, W) * max_r_frac  # pixel radius of outer ring

    dir_x = torch.cos(angles)  # [n_spokes]
    dir_y = torch.sin(angles)  # [n_spokes]

    # radial checkpoints along each spoke:
    # start at 0.10 (not 0.0) so centre isn't a solid blob
    ts = torch.linspace(
        0.10,
        1.00,
        steps=n_levels,
        device=device,
        dtype=dtype,
    )  # [n_levels]

    # spoke_points_x[i, k] = x coord of level k on spoke i
    # spoke_points_y[i, k] = y coord of level k on spoke i
    spoke_points_x = hub_x + ts.view(1, -1) * (dir_x.view(-1, 1) * max_r)
    spoke_points_y = hub_y + ts.view(1, -1) * (dir_y.view(-1, 1) * max_r)
    # shapes: [n_spokes, n_levels]

    # 1. draw each spoke from hub to its outermost checkpoint
    for i in range(n_spokes):
        x0, y0 = hub_x, hub_y
        x1 = spoke_points_x[i, -1].item()
        y1 = spoke_points_y[i, -1].item()
        _draw_segment(canvas, x0, y0, x1, y1, thickness_px)

    # 2. draw connectors at each level between adjacent spokes
    # this creates the polygonal "rings"
    for k in range(n_levels):
        for i in range(n_spokes):
            j = (i + 1) % n_spokes  # neighbor spoke, wraps around
            x0 = spoke_points_x[i, k].item()
            y0 = spoke_points_y[i, k].item()
            x1 = spoke_points_x[j, k].item()
            y1 = spoke_points_y[j, k].item()
            _draw_segment(canvas, x0, y0, x1, y1, thickness_px)

    # convert bool -> float32 and shape -> [1,1,H,W]
    return canvas.float().unsqueeze(0).unsqueeze(0)

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


def circle_mask(size=64, r=10, x_offset=0, y_offset=0):
    """
    Create a circular mask of given size and radius.
    Parameters:
        - size: The size of the mask (width and height).
        - r: The radius of the circle.
        - x_offset: The x-coordinate offset for the center of the circle.
        - y_offset: The y-coordinate offset for the center of the circle.
    Returns:
        - mask: A boolean mask where True values represent the circular area.

    Reference: https://stackoverflow.com/questions/69687798/generating-a-soft-circluar-mask-using-numpy-python-3
    """
    x0 = y0 = size // 2
    x0 += x_offset
    y0 += y_offset
    y, x = np.ogrid[:size, :size]
    y = y[::-1]
    return ((x - x0) ** 2 + (y - y0) ** 2) <= r**2

def get_watermarking_mask(
    init_latents_w,
    w_mask_shape: str,
    w_channel: int,
    w_radius: float,
    device: str,
):
    """
    Generate a watermarking mask based on the specified parameters.
    Parameters:
        - init_latents_w: The initial latents with watermarking.
        - w_mask_shape: The shape of the watermarking mask (e.g., "circle", "square").
        - w_channel: The channel to apply the watermarking mask to (-1 for all channels).
        - w_radius: The radius for the watermarking mask.
        - device: The device to use for computation (e.g., "cuda" or "cpu").
    Returns:
        - watermarking_mask: The generated watermarking mask.
    """
    watermarking_mask = torch.zeros(init_latents_w.shape, dtype=torch.bool).to(device)

    if w_mask_shape == "circle":
        np_mask = circle_mask(init_latents_w.shape[-1], r=w_radius)
        torch_mask = torch.tensor(np_mask).to(device)

        if w_channel == -1:
            # all channels
            watermarking_mask[:, :] = torch_mask
        else:
            watermarking_mask[:, w_channel] = torch_mask
    elif w_mask_shape == "square":
        anchor_p = init_latents_w.shape[-1] // 2
        if w_channel == -1:
            # all channels
            watermarking_mask[
                :,
                :,
                anchor_p - w_radius : anchor_p + w_radius,
                anchor_p - w_radius : anchor_p + w_radius,
            ] = True
        else:
            watermarking_mask[
                :,
                w_channel,
                anchor_p - w_radius : anchor_p + w_radius,
                anchor_p - w_radius : anchor_p + w_radius,
            ] = True
    elif w_mask_shape == "no":
        pass
    else:
        raise NotImplementedError(f"w_mask_shape: {w_mask_shape}")

    return watermarking_mask

def _ensure_bchw(x, device=None, dtype=None):
    if not torch.is_tensor(x):
        raise TypeError("expected torch.Tensor")
    if x.ndim == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 3:
        x = x.unsqueeze(0)
    if device is not None:
        x = x.to(device)
    if dtype is not None:
        x = x.to(dtype)
    return x

def generate_logpolar_grid(
    H_lp, W_lp, n_spokes=16, n_rings=12, device="cpu", dtype=torch.float16
):
    """
    Create a synthetic log-polar canvas that contains both concentric rings and radial spokes.
    Output shape: (1,1,H_lp,W_lp) - single-channel; you can expand to match channels.
    """
    # normalized coords u in [0,1] for radius index, v in [0,1) for theta
    u = torch.linspace(0.0, 1.0, H_lp, device=device, dtype=dtype).view(H_lp, 1)
    v = torch.linspace(0.0, 1.0, W_lp, device=device, dtype=dtype).view(1, W_lp)

    # ring pattern: sinusoidal in log-radius (higher frequency near center if desired)
    rings = 0.5 * (1.0 + torch.sign(torch.sin(u * n_rings * math.pi)))  # coarse rings
    # alternative smoother rings (uncomment if you prefer smooth):
    # rings = 0.5 * (1.0 + torch.sin(u * n_rings * math.pi))

    # spoke pattern: sinusoidal in angle
    spokes = 0.5 * (1.0 + torch.sin(v * n_spokes * 2.0 * math.pi))

    # combine them multiplicatively (or additively) to produce grid intersections
    lp = rings * spokes  # shape (H_lp, W_lp)
    lp = lp.unsqueeze(0).unsqueeze(0)  # (1,1,H_lp,W_lp)
    return lp.to(device=device, dtype=dtype)

def logpolar_to_cartesian(lp_img, out_H, out_W, r_min=1.0, eps=1e-6):
    B, C, H_lp, W_lp = lp_img.shape
    device = lp_img.device
    dtype = lp_img.dtype

    cx = (out_W - 1) / 2.0
    cy = (out_H - 1) / 2.0
    R_max = math.hypot(cx, cy)

    log_r_min = math.log(r_min + eps)
    log_r_max = math.log(R_max + eps)

    xs = torch.linspace(0.0, out_W - 1, out_W, device=device, dtype=dtype)
    ys = torch.linspace(0.0, out_H - 1, out_H, device=device, dtype=dtype)
    y_grid, x_grid = torch.meshgrid(ys, xs, indexing="ij")  # (out_H, out_W)

    dx = x_grid - cx
    dy = y_grid - cy
    r = torch.sqrt(dx * dx + dy * dy)
    theta = torch.atan2(dy, dx)
    theta = torch.where(theta < 0.0, theta + 2.0 * math.pi, theta)

    r_clamped = torch.clamp(r, min=1.0, max=R_max)
    log_r = torch.log(r_clamped + eps)
    u = (log_r - log_r_min) / (log_r_max - log_r_min)
    u = u.clamp(0.0, 1.0) * (H_lp - 1)

    v = (theta / (2.0 * math.pi)) * (W_lp - 1)

    x_src = (v / (W_lp - 1)) * 2.0 - 1.0
    y_src = (u / (H_lp - 1)) * 2.0 - 1.0

    grid = torch.stack((x_src, y_src), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
    cart = F.grid_sample(
        lp_img, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    return cart


def get_watermarking_pattern(
    pipe,
    w_seed: int,
    w_pattern: str,
    w_radius: int,
    device: torch.device,
    shape: tuple | None = None,
    strength: float = 0.9,
    n_spokes=8,
    n_levels=4,
    thickness_px=1.0,  # line thickness in pixels
    max_r_frac=0.3,
):
    def set_random_seed(s):
        torch.manual_seed(int(s) & 0xFFFFFFFF)

    set_random_seed(w_seed)

    # prepare initial tensor
    if shape is not None:
        gt_init = torch.randn(*shape, device=device)
    else:
        if hasattr(pipe, "get_random_latents"):
            gt_init = pipe.get_random_latents()
        else:
            if hasattr(pipe, "unet") and hasattr(pipe.unet.config, "sample_size"):
                sample_size = pipe.unet.config.sample_size
                if isinstance(sample_size, int):
                    H = W = sample_size
                else:
                    H, W = (
                        sample_size
                        if len(sample_size) >= 2
                        else (sample_size[0], sample_size[0])
                    )
                in_ch = pipe.unet.config.in_channels
                gt_init = torch.randn(1, in_ch, H, W, device=device)
            else:
                gt_init = torch.randn(1, 3, 256, 256, device=device)

    gt_init = _ensure_bchw(gt_init, device=device, dtype=torch.float32)

    # handle patterns that require FFT on float32: cast to float32 then cast back
    def safe_fft2_shift(x):
        orig_dtype = x.dtype
        x_f = x.to(torch.float32)
        fft = torch.fft.fft2(x_f)
        fft_s = torch.fft.fftshift(fft, dim=(-1, -2))
        # return cast back to original dtype (complex->real not appropriate); here we keep real part if needed
        # but most usage expects complex spectrum; if you only use magnitude/phase you should adapt accordingly.
        return fft_s.to(orig_dtype) if orig_dtype.is_floating_point else fft_s

    if "logpolar_grid" in w_pattern:
        # create synthetic LP canvas (single-channel) and expand to channels
        B, C, H, W = gt_init.shape
        H_lp = H
        W_lp = W
        # tune these counts to get desired spokes/rings
        n_spokes = max(8, int(w_radius))  # e.g. control by w_radius
        n_rings = max(6, int(w_radius // 2))
        lp_canvas = generate_logpolar_grid(
            H_lp,
            W_lp,
            n_spokes=n_spokes,
            n_rings=n_rings,
            device=device,
            dtype=gt_init.dtype,
        )

        fft_s = safe_fft2_shift(gt_init)
        gt_temp = fft_s.real if torch.is_complex(fft_s) else fft_s
        # gt_temp = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))

        # expand channel-wise and add small noise so it's less "perfect"
        lp_canvas = lp_canvas.expand(B, C, -1, -1)
        # map back to cartesian
        gt_patch = logpolar_to_cartesian(lp_canvas, out_H=H, out_W=W)
        # normalise to mean ~0, std ~1 (optional) or to model range
        # here we scale to similar stats as gt_init
        gt_patch = (gt_patch - gt_patch.mean()) / (gt_patch.std(unbiased=False) + 1e-8)
        gt_patch = (gt_patch * strength) + (
            gt_temp * (1 - strength)
        )  # scale/downweight

        return gt_patch.to(dtype=torch.complex32)

    # if "spiderweb" in w_pattern:
    #     # base tensor
    #     B, C, H, W = gt_init.shape

    #     # pick complexity from w_radius (so you can tune from CLI)
    #     n_spokes = max(8, int(w_radius))          # more spokes = denser web
    #     n_rings  = max(6, int(w_radius // 2))     # more rings = more layers

    #     # 1) generate an organic spiderweb mask in polar-ish coords
    #     web_mask = generate_octoweb_grid_draw(128, 128, n_rings=8, line_thickness=0.015) # [1,1,H,W] ~0..1

    #     # 2) expand to all channels
    #     web_mask = web_mask.expand(B, C, H, W)

    #     # 3) optional: slightly blur / soften the mask so it's not razor pixel-y
    #     # a tiny box blur by avg-pooling; keep it light so pattern is still visible
    #     web_mask_soft = torch.nn.functional.avg_pool2d(
    #         web_mask,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #     )

    #     # 4) normalise the mask so it's roughly zero-mean / unit-var,
    #     #    then blend with some base frequency content
    #     fft_s = safe_fft2_shift(gt_init)
    #     base_freq = fft_s.real if torch.is_complex(fft_s) else fft_s

    #     wm = web_mask_soft
    #     wm = (wm - wm.mean()) / (wm.std(unbiased=False) + 1e-8)

    #     gt_patch = (wm * strength) + (base_freq * (1.0 - strength))

    #     # return complex32 for downstream consistency (like you do elsewhere)
    #     return gt_patch.to(dtype=torch.complex32)

    if "octoweb" in w_pattern:
        # base latent shape
        B, C, H, W = gt_init.shape

        # generate the clean web mask in image space
        # tie density to w_radius, so you can control levels from CLI
        # n_levels = max(4, int(w_radius))  # how many "rings" outward
        web_mask = generate_regular_spiderweb(
            H,
            W,
            n_spokes=n_spokes,
            n_levels=n_levels,
            thickness_px=thickness_px,  # line thickness in pixels
            max_r_frac=max_r_frac,  # relative size of web
            device=device,
            dtype=gt_init.dtype,
        )  # [1,1,H,W] in {0,1}

        # repeat across channels to match gt_init channels
        web_mask = web_mask.expand(B, C, H, W)  # [B,C,H,W]

        # normalise spiderweb mask so it blends like a watermark carrier
        wm = (web_mask - web_mask.mean()) / (web_mask.std(unbiased=False) + 1e-8)

        # get "base_freq" version of your latent like your other branches
        # (safe float32 FFT, shift to freq-ish layout, then take real part)
        fft_s = torch.fft.fftshift(
            torch.fft.fft2(gt_init.to(torch.float32)),
            dim=(-1, -2),
        )
        base_freq = fft_s.real  # [B,C,H,W]

        # blend web pattern with base frequency content
        gt_patch = wm * strength + base_freq * (1.0 - strength)

        # keep return dtype consistent with others (complex32)
        return gt_patch.to(dtype=torch.complex32)

    # keep previous behaviors but fix dtype issues and linspace earlier (we already fixed linspace above)
    if "seed_ring" in w_pattern:
        gt_patch = gt_init
        gt_patch_tmp = copy.deepcopy(gt_patch)
        H = gt_init.shape[-1]
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask, device=device, dtype=torch.bool)
            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    elif "seed_zeros" in w_pattern:
        gt_patch = gt_init * 0

    elif "seed_rand" in w_pattern:
        gt_patch = gt_init

    elif "rand" in w_pattern:
        # perform safe fft on float32 to avoid half dtype error
        fft_s = safe_fft2_shift(gt_init)
        # use magnitude or real-part as desired - here take real part
        gt_patch = fft_s.real if torch.is_complex(fft_s) else fft_s
        gt_patch = gt_patch * 0  # your original wanted some behaviour; adapt as needed
        gt_patch[:] = gt_patch[0]

    elif "zeros" in w_pattern:
        fft_s = safe_fft2_shift(gt_init)
        gt_patch = (fft_s.real if torch.is_complex(fft_s) else fft_s) * 0

    elif "const" in w_pattern:
        w_pattern_const = globals().get("w_pattern_const", 0.0)
        fft_s = safe_fft2_shift(gt_init)
        gt_patch = (fft_s.real if torch.is_complex(fft_s) else fft_s) * 0
        gt_patch = gt_patch + float(w_pattern_const)

    elif "ring" in w_pattern:
        gt_patch = torch.fft.fftshift(torch.fft.fft2(gt_init), dim=(-1, -2))
        gt_patch_tmp = copy.deepcopy(gt_patch)
        for i in range(w_radius, 0, -1):
            tmp_mask = circle_mask(gt_init.shape[-1], r=i)
            tmp_mask = torch.tensor(tmp_mask).to(device)

            for j in range(gt_patch.shape[1]):
                gt_patch[:, j, tmp_mask] = gt_patch_tmp[0, j, 0, i].item()

    else:
        gt_patch = gt_init

    return gt_patch.to(dtype=torch.complex32)