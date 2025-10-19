import io
import random
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
import torchvision.transforms.functional as TF

def set_random_seed(seed: Optional[int]):
    """Set python, numpy and torch seeds for reproducibility across transforms."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class Attacker:
    """
    Attacker applies deterministic distortions to a single PIL Image.

    Example:
        atk = Attacker(r_degree=15, jpeg_ratio=80, crop_scale=0.6, crop_ratio=0.9, ...)
        img_adv = atk.attack(img, seed=42)
    """

    def __init__(
        self,
        r_degree: Optional[float] = 15,
        jpeg_ratio: Optional[int] = None,
        crop_scale: Optional[float] = 0.6,
        crop_ratio: Optional[float] = 0.9,
        gaussian_blur_r: Optional[float] = 0.2,
        gaussian_std: Optional[float] = 0.03,
        brightness_factor: Optional[float] = None,
        rand_aug: int = 0,
        run_name: str = "attack",
    ):
        self.r_degree = r_degree
        self.jpeg_ratio = jpeg_ratio
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.gaussian_blur_r = gaussian_blur_r
        self.gaussian_std = gaussian_std
        self.brightness_factor = brightness_factor
        self.rand_aug = rand_aug
        self.run_name = run_name

        # optional RandAugment
        if rand_aug and hasattr(transforms, "RandAugment"):
            self.randaugment = transforms.RandAugment(num_ops=rand_aug)
        else:
            self.randaugment = None

    def _apply_jpeg_bytes(self, img: Image.Image, quality: int) -> Image.Image:
        """Compress image to JPEG via in-memory buffer and reload (no disk)."""
        buffer = io.BytesIO()
        q = int(max(1, min(95, quality)))
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert(img.mode)

    def _random_resized_crop(self, img: Image.Image, scale: float, ratio: float, seed: int) -> Image.Image:
        """
        Deterministic RandomResizedCrop using get_params with the provided seed.
        Returns an image of the same size as input (resized back to original).
        """
        set_random_seed(seed)
        target_size = img.size  # (W, H)
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, scale=(scale, scale), ratio=(ratio, ratio))
        return TF.resized_crop(img, i, j, h, w, target_size)

    def _add_gaussian_noise(self, img: Image.Image, std: float, seed: Optional[int]) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.normal(0.0, std * 255.0, arr.shape).astype(np.float32)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(out)

    def attack(self, img: Image.Image, seed: Optional[int] = None) -> Image.Image:
        """
        Apply distortions to a single PIL Image. Returns a new PIL Image.
        Deterministic for a given seed.
        """
        # Normalize mode
        img = img.convert("RGB") if img.mode != "RGB" else img.copy()

        # Rotation: uniform angle in [-r_degree, r_degree]
        if self.r_degree is not None:
            if seed is not None:
                set_random_seed(seed)
            angle = random.uniform(-abs(self.r_degree), abs(self.r_degree))
            img = TF.rotate(img, angle)

        # JPEG compression via in-memory bytes
        if self.jpeg_ratio is not None:
            img = self._apply_jpeg_bytes(img, self.jpeg_ratio)

        # RandomResizedCrop (deterministic via seed)
        if self.crop_scale is not None and self.crop_ratio is not None:
            s = seed if seed is not None else random.randint(0, 2 ** 31 - 1)
            img = self._random_resized_crop(img, self.crop_scale, self.crop_ratio, s)

        # Optional RandAugment
        if self.randaugment is not None:
            if seed is not None:
                set_random_seed(seed)
            img = self.randaugment(img)

        # Gaussian blur
        if self.gaussian_blur_r is not None and self.gaussian_blur_r > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.gaussian_blur_r))

        # Additive Gaussian noise
        if self.gaussian_std is not None and self.gaussian_std > 0:
            img = self._add_gaussian_noise(img, self.gaussian_std, seed)

        # Brightness jitter (deterministic if seed provided)
        if self.brightness_factor is not None:
            if seed is not None:
                set_random_seed(seed)
            if isinstance(self.brightness_factor, (tuple, list)):
                factor = random.uniform(self.brightness_factor[0], self.brightness_factor[1])
            else:
                factor = float(self.brightness_factor)
            img = TF.adjust_brightness(img, factor)

        return img
