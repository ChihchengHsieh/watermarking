import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import io
import os
from PIL import ImageFilter


# ---------------- helper to collect files and labels ----------------
def discover_dataset_files(data_dir: str):
    """
    Discover files in data_dir. Support:
      - subfolders 'watermarked' and 'clean' (or any two subfolders)
      - .pt files with keys
    Returns lists: file_paths, labels
    """
    p = Path(data_dir)
    if not p.exists():
        raise RuntimeError(f"{data_dir} not found")

    # case: two subfolders inside (binary classes)
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    if len(subdirs) >= 2:
        # choose the first two directories as classes
        classes = sorted(subdirs)[:2]
        file_paths = []
        labels = []
        for label, cdir in enumerate(classes):
            exts = list(cdir.glob("*"))
            for f in exts:
                if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".pt", ".pth"]:
                    file_paths.append(str(f))
                    labels.append(label)
        return file_paths, labels

    # case: many .pt files with 'fft' or 'image' and 'label'
    pts = list(p.glob("*.pt"))
    if len(pts) > 0:
        file_paths = []
        labels = []
        for f in pts:
            try:
                d = torch.load(f)
                if isinstance(d, dict) and "label" in d:
                    file_paths.append(str(f))
                    labels.append(int(d["label"]))
            except Exception:
                continue
        if len(file_paths) > 0:
            return file_paths, labels

    # fallback: collect images in folder and try to infer labels by filename (contains 'water' or 'wm')
    imgs = [
        str(f)
        for f in p.glob("*")
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]
    ]
    if len(imgs) > 0:
        file_paths = []
        labels = []
        for f in imgs:
            fname = os.path.basename(f).lower()
            lbl = 0
            if "water" in fname or "wm" in fname or "marked" in fname or "1_" in fname:
                lbl = 1
            file_paths.append(f)
            labels.append(lbl)
        return file_paths, labels

    raise RuntimeError(
        "Unable to discover dataset files. Please structure dataset as subfolders or .pt files with label key."
    )


def jpeg_compress_pil(img: Image.Image, quality: int = 85):
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class RandomJPEG:
    def __init__(self, p=0.5, q_range=(60, 95)):
        self.p = p
        self.q_range = q_range

    def __call__(self, img):
        if random.random() < self.p:
            q = random.randint(self.q_range[0], self.q_range[1])
            return jpeg_compress_pil(img, q)
        return img


class RandomGaussianNoise:
    def __init__(self, p=0.5, std=0.01):
        self.p = p
        self.std = std

    def __call__(self, img):
        if random.random() < self.p:
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(0, self.std, arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
            img2 = Image.fromarray((arr * 255).astype(np.uint8))
            return img2
        return img


def make_train_image_augmentations(IMAGE_SIZE):
    # Compose PIL-based augmentations (randomly applied)
    aug_list = []
    # random rotation small
    aug_list.append(
        transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5)
    )
    # random resized crop (sometimes)
    aug_list.append(
        transforms.RandomApply(
            [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0))], p=0.6
        )
    )
    # horizontal flip
    aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
    # color jitter
    aug_list.append(
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.6)
    )
    # JPEG
    aug_list.append(RandomJPEG(p=0.3, q_range=(60, 95)))
    # RandAugment (if torchvision supports it) - wrapped
    try:
        from torchvision.transforms import RandAugment

        aug_list.append(transforms.RandomApply([RandAugment()], p=0.25))
    except Exception:
        pass

    # Gaussian blur sometimes
    aug_list.append(
        transforms.RandomApply(
            [
                lambda img: img.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.8))
                )
            ],
            p=0.25,
        )
    )
    # Add gaussian pixel noise sometimes
    aug_list.append(RandomGaussianNoise(p=0.25, std=0.02))
    # brightness jitter more finely (RandomApply)
    aug_list.append(
        transforms.RandomApply([transforms.ColorJitter(brightness=(0.8, 1.2))], p=0.5)
    )

    # final: ensure image is resized to IMAGE_SIZE (if not already)
    final = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *aug_list])
    return final

def get_test_aug(image_size):
    return transforms.Compose(
        [
            transforms.Resize(image_size),
        ]
    )

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

# ----------------- assume helpers are available from your notebook -----------------
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

import math
def psnr_to_prob_sigmoid(psnr, threshold=-4.0, scale=1.0):
    """Simple sigmoid mapping. threshold -> p=0.5, smaller scale -> steeper curve."""
    return 1.0 / (1.0 + math.exp(-(psnr - threshold) / scale))


# ---------------- Data loader that does augment -> pipe -> forward_diffusion -> FFT ----------------
class WatermarkOnTheFlyDataset(Dataset):
    """
    Loads image files and labels, applies augmentations, then runs:
      tsr_img -> pipe.get_image_latents(sample=False) -> pipe.forward_diffusion(...) -> FFT
    Returns: (fft_channels_tensor (float32, shape (2*C, H, W)), label)
    """

    def __init__(
        self,
        file_paths,
        labels,
        pipe,
        text_embeddings,
        num_inference_steps,
        watermarking_mask,
        gt_patch,
        guidance_scale=1.0,
        device="cpu",
        image_aug=None,
        image_aug_prob=0.5,
        image_size=512,
        include_psnr=False,
        include_mask_patch=False,
        psnr_return_prob=True,
    ):
        assert len(file_paths) == len(labels)
        self.file_paths = file_paths
        self.labels = labels
        self.pipe = pipe
        self.text_embeddings = text_embeddings
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.device = device
        self.image_aug = image_aug
        self.image_aug_prob = image_aug_prob
        self.test_aug = get_test_aug(image_size)
        self.watermarking_mask = watermarking_mask 
        self.gt_patch = gt_patch
        self.include_psnr = include_psnr
        self.include_mask_patch = include_mask_patch
        self.return_reversed_latents = False  # default behavior
        self.psnr_return_prob = psnr_return_prob

    def __len__(self):
        return len(self.file_paths)
    
    def _load_pil(self, fp):
        # accept PIL.Image, numpy array, or path
        if isinstance(fp, Image.Image):
            return fp.convert("RGB")
        if isinstance(fp, torch.Tensor):
            # convert tensor (C,H,W) to PIL
            arr = (fp.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return Image.fromarray(arr)
        p = Path(fp)
        img = Image.open(p).convert("RGB")
        return img
    
    def set_return_reversed_latents(self, return_reversed_latents: bool):
        self.return_reversed_latents = return_reversed_latents

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = int(self.labels[idx])

        pil_img = self._load_pil(path)

        # --- AUGMENT IMAGE FIRST ---

        if self.image_aug and (random.random() <= self.image_aug_prob):
            img_aug = self.image_aug(pil_img)
        else:
            img_aug = self.test_aug(pil_img)

        # convert to tensor and to device/dtype for pipe
        tsr_img = transform_img(img_aug).unsqueeze(0)  # (1,C,H,W)
        # move to correct dtype & device for the unet/vae as the user did earlier:
        target_dtype = next(self.pipe.unet.parameters()).dtype
        tsr_img = tsr_img.to(dtype=target_dtype, device=self.device)

        # --- encode to image latents ---
        with torch.no_grad():
            image_latents = self.pipe.get_image_latents(
                tsr_img, sample=False
            )  # user's helper expects (C,H,W) or (B,C,H,W)
            # ensure batch dim
            if image_latents.ndim == 3:
                image_latents = image_latents.unsqueeze(0)

            # --- forward/inversion -> x_T (depending on forward_diffusion implementation)
            reversed_latents = self.pipe.forward_diffusion(
                latents=image_latents,
                text_embeddings=self.text_embeddings,
                guidance_scale=1,
                num_inference_steps=self.num_inference_steps,
            )  # expect tensor shape (B,C,H,W)

            if self.return_reversed_latents:
                return reversed_latents[0], torch.tensor(label, dtype=torch.long)

            psnr_metric = eval_watermark(
                reversed_latents,  # single sample
                self.watermarking_mask,
                self.gt_patch,
                w_measurement="psnr_complex",
            )

            if self.psnr_return_prob:
                psnr_metric = psnr_to_prob_sigmoid(psnr_metric, threshold=-4.0, scale=1.0)

            # Keep complex-safe: cast to float32 after splitting real/imag
            # Compute FFT (complex)
            vis_latent_fft = torch.fft.fftshift(
                torch.fft.fft2(reversed_latents), dim=(-1, -2)
            )  # (B,C,H,W) complex
            # we assume batch==1
            fft_b = vis_latent_fft[0]
            # convert to float channels (real, imag) as float32
            real = fft_b.real.to(dtype=torch.float32)
            imag = fft_b.imag.to(dtype=torch.float32)
            fft_ch = torch.cat([real, imag], dim=0)  # (2*C, H, W)

        if self.include_mask_patch:
            # concatenate watermarking mask and gt_patch as float32 channels
            mask_ch = self.watermarking_mask.to(dtype=torch.float32).squeeze(0)  # (1,H,W)
            gt_patch_ch = self.gt_patch.to(dtype=torch.float32).squeeze(0)  # (1,H,W)
            fft_ch = torch.cat([fft_ch, mask_ch, gt_patch_ch], dim=0)  # (2*C+2, H, W)

            if self.include_psnr:
                return fft_ch, torch.tensor(psnr_metric, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            return fft_ch, torch.tensor(label, dtype=torch.long)

        if self.include_psnr:
            return fft_ch, torch.tensor(psnr_metric, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        
        return fft_ch, torch.tensor(label, dtype=torch.long)
