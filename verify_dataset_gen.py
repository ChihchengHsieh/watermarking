import torch
import random
from pathlib import Path
from tqdm.auto import tqdm
from PIL import Image
from watermark import inject_watermark, get_watermarking_mask, get_watermarking_pattern

def ensure_pil(img):
    # accepts PIL, numpy (H,W,C), or tensor (C,H,W) or (B,C,H,W)
    if isinstance(img, Image.Image):
        return img
    if isinstance(img, torch.Tensor):
        if img.ndim == 4:
            img = img[0]
        arr = img.detach().cpu().permute(1, 2, 0).numpy()
        # arr expected in [0,1] or [0,255]
        if arr.dtype == "float32" or arr.dtype == "float64":
            arr = (arr * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
    # numpy array
    import numpy as np

    if isinstance(img, (np.ndarray,)):
        arr = img
        if arr.dtype in (np.float32, np.float64):
            arr = (arr * 255).clip(0, 255).astype("uint8")
        return Image.fromarray(arr)
    raise RuntimeError("Unsupported image type")


# a robust getter for a random latent without writing new helpers
def obtain_random_latents_from_pipe(pipe, default_device, default_dtype):
    # 1) try no-arg get_random_latents()
    try:
        lat = pipe.get_random_latents()
        if isinstance(lat, torch.Tensor):
            return lat
        # sometimes returns wrapper
        if hasattr(lat, "latents"):
            return lat.latents
        if isinstance(lat, (tuple, list)) and isinstance(lat[0], torch.Tensor):
            return lat[0]
    except Exception:
        pass

    # 2) try prepare_latents (typical diffusers)
    try:
        if hasattr(pipe, "prepare_latents"):
            # try typical signature
            try:
                # infer shape
                ch = getattr(pipe.unet.config, "in_channels", 4)
                sample_size = getattr(pipe.unet.config, "sample_size", 64)
                if isinstance(sample_size, int):
                    H = W = sample_size
                else:
                    H = sample_size[0]
                    W = sample_size[1] if len(sample_size) > 1 else sample_size[0]
                lat = pipe.prepare_latents(
                    batch_size=1, num_images_per_prompt=1, height=H, width=W
                )
                if isinstance(lat, torch.Tensor):
                    return lat
                if hasattr(lat, "latents"):
                    return lat.latents
            except TypeError:
                # try bare call
                try:
                    lat = pipe.prepare_latents()
                    if isinstance(lat, torch.Tensor):
                        return lat
                    if hasattr(lat, "latents"):
                        return lat.latents
                except Exception:
                    pass
    except Exception:
        pass

    # 3) fallback random tensor based on unet config
    try:
        sample_size = getattr(pipe.unet.config, "sample_size", 64)
        in_ch = getattr(pipe.unet.config, "in_channels", 4)
        if isinstance(sample_size, int):
            H = W = sample_size
        else:
            H = sample_size[0]
            W = sample_size[1] if len(sample_size) > 1 else sample_size[0]
        lat = torch.randn(1, in_ch, H, W, device=default_device, dtype=default_dtype)
        return lat
    except Exception:
        return torch.randn(1, 4, 64, 64, device=default_device, dtype=default_dtype)


def get_gen_dataset(gen_dataset):
    if gen_dataset == "stablediff":
        from datasets import load_dataset

        hf_dataset_name = "Gustavosta/Stable-Diffusion-Prompts"
        split = "train"
        ds = load_dataset(hf_dataset_name, split=split)
        prompt_key = "Prompt"
    elif gen_dataset == "coco":
        import json

        with open("fid_outputs/coco/meta_data.json") as f:
            ds = json.load(f)
            ds = ds["annotations"]
            prompt_key = "caption"
    else:
        raise RuntimeError(f"Unknown dataset: {gen_dataset}")

    if prompt_key is None:
        raise RuntimeError(
            f"Prompt column not found in dataset columns: {ds.column_names}"
        )

    return ds, prompt_key


def generate_verifier_training_set(
    gen_dataset,
    pipe,
    w_mask_shape,
    w_channel,
    w_radius,
    w_seed,
    w_pattern,
    w_strength,
    w_injection,
    gen_seed,
    num_inference_steps,
    guidance_scale,
    device,
    n_per_class=500,
):
    ds, prompt_key = get_gen_dataset(gen_dataset)

    out_dir = Path(f"verifier_dataset_{gen_dataset}_{w_pattern}")
    out_clean_dir = out_dir / "clean"
    out_w_dir = out_dir / "watermarked"
    out_clean_dir.mkdir(parents=True, exist_ok=True)
    out_w_dir.mkdir(parents=True, exist_ok=True)

    # deterministic selection
    rng = random.Random(gen_seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    chosen = indices[:n_per_class]
    

    # save chosen prompts
    chosen_prompts = [ds[i][prompt_key] for i in chosen]
    with open(out_dir / "chosen_prompts.txt", "w", encoding="utf-8") as f:
        for prompt in chosen_prompts:
            f.write(prompt + "\n")

    # return # for generating the prompts only

    # infer device/dtype from pipe.unet if possible
    try:
        unet_param = next(pipe.unet.parameters())
        default_device = unet_param.device
        default_dtype = unet_param.dtype
    except Exception:
        default_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_dtype = torch.float32

    # main loop
    for i, ds_idx in tqdm(
        list(enumerate(chosen)), desc="generating", total=len(chosen)
    ):
        prompt = ds[ds_idx][prompt_key]

        # 1) base latent (no args)
        init_latents = obtain_random_latents_from_pipe(
            pipe, default_device, default_dtype
        )
        init_latents = init_latents.to(default_device)
        try:
            init_latents = init_latents.to(default_dtype)
        except Exception:
            pass

        # 2) mask and pattern using your notebook helpers
        watermarking_mask = get_watermarking_mask(
            init_latents,
            w_mask_shape=w_mask_shape,
            w_channel=w_channel,
            w_radius=w_radius,
            device=default_device,
        )

        gt_patch = get_watermarking_pattern(
            pipe,
            w_seed=w_seed + i,
            w_pattern=w_pattern,
            w_radius=w_radius,
            device=device,
            strength=w_strength,
            shape=None,
        )

        # 3) dtype-safe injection attempts (use your inject_watermark)
        lat_clean = init_latents.clone()
        lat_w = None

        # detect FFT dtype of the latents (what inject might expect for complex injection)
        try:
            fft_example = torch.fft.fft2(lat_clean)
            fft_dtype = fft_example.dtype
            is_fft_complex = torch.is_complex(fft_example)
        except Exception:
            fft_dtype = None
            is_fft_complex = False

        # Try direct injection, otherwise try casting gt_patch to a compatible dtype
        try:
            lat_w = inject_watermark(
                lat_clean.clone(),
                watermarking_mask,
                gt_patch.clone(),
                w_injection=w_injection,
            )
        except Exception as e:
            # common dtype mismatch: complex vs real half/float
            tried = False
            if fft_dtype is not None and is_fft_complex:
                # create complex gt_patch in the same complex dtype, using real part = gt_patch, imag = 0
                try:
                    # map real -> float32 then to complex if needed
                    real_dtype = (
                        torch.float32 if fft_dtype == torch.complex64 else torch.float64
                    )
                    gp_real = gt_patch.clone().to(device=default_device).to(real_dtype)
                    gp_complex = gp_real.to(
                        fft_dtype
                    )  # cast to complex dtype (zero imag)
                    lat_w = inject_watermark(
                        lat_clean.clone(),
                        watermarking_mask,
                        gp_complex,
                        w_injection=w_injection,
                    )
                    tried = True
                except Exception:
                    tried = False
            if not tried:
                # last fallback: try injecting with seed-mode (real-space) if available
                try:
                    lat_w = inject_watermark(
                        lat_clean.clone(),
                        watermarking_mask,
                        gt_patch.clone().to(default_dtype),
                        w_injection="seed",
                    )
                    tried = True
                except Exception:
                    tried = False
            if not tried:
                raise RuntimeError(
                    f"inject_watermark failed (original error: {e}). Tried complex-cast and 'seed' fallback but both failed."
                )

        if lat_w is None:
            raise RuntimeError("Failed to produce watermarked latents (lat_w is None)")

        # 4) produce images with pipe (prefer passing latents)
        try:
            out_clean = pipe(
                prompt=prompt,
                latents=init_latents,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            im_clean = (
                out_clean.images if hasattr(out_clean, "images") else out_clean[0]
            )
        except Exception:
            # fallback to prompt-only generation
            out_clean = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            im_clean = (
                out_clean.images if hasattr(out_clean, "images") else out_clean[0]
            )

        try:
            out_w = pipe(
                prompt=prompt,
                latents=lat_w,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            im_w = out_w.images if hasattr(out_w, "images") else out_w[0]
        except Exception:
            out_w = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
            )
            im_w = out_w.images if hasattr(out_w, "images") else out_w[0]

        pil_clean = ensure_pil(im_clean[0])
        pil_w = ensure_pil(im_w[0])

        fname = f"{i:04d}"
        pil_clean.save(out_clean_dir / f"{fname}.png")
        pil_w.save(out_w_dir / f"{fname}.png")

    print("Saved dataset to:", out_dir)
