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



from torchvision import transforms
from ds import RandomGaussianNoise, RandomJPEG
from PIL import ImageFilter


# 1. No attack (identity transform / just resize to model input size)
def make_clean_aug(IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
    )


# 2. Strong JPEG
def make_jpeg_aug(IMAGE_SIZE, q_low=40, q_high=70):
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            RandomJPEG(p=1.0, q_range=(q_low, q_high)),  # always compress
        ]
    )


# 3. Blur attack
def make_blur_aug(IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.Lambda(
                lambda img: img.filter(
                    ImageFilter.GaussianBlur(radius=random.uniform(1.5, 3.0))
                )
            ),
        ]
    )


# 4. Random crop / rotate style geometric distortion
def make_geom_aug(IMAGE_SIZE):
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        ]
    )


def make_down_up_attack(IMAGE_SIZE, downscale_frac=0.5):
    small = max(1, int(IMAGE_SIZE * downscale_frac))
    return transforms.Compose(
        [
            transforms.Resize(
                (small, small), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    )


def make_msg_app_combo(IMAGE_SIZE):
    # downscale -> strong jpeg -> upsample (very realistic)
    small = max(1, int(IMAGE_SIZE * 0.5))
    return transforms.Compose(
        [
            transforms.Resize(
                (small, small), interpolation=transforms.InterpolationMode.BILINEAR
            ),
            RandomJPEG(p=1.0, q_range=(40, 70)),
            transforms.Resize(
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
        ]
    )

def make_random_crop_attack(IMAGE_SIZE, scale=(0.5, 0.9)):
    # heavy random crop + resize (brutal for local watermarks)
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=scale, ratio=(0.75, 1.33)),
        ]
    )


def make_occlusion_block(IMAGE_SIZE, box_frac=0.25):
    class Block(object):
        def __init__(self, frac):
            self.frac = frac

        def __call__(self, img):
            w, h = img.size
            bw, bh = int(w * self.frac), int(h * self.frac)
            x0 = random.randint(0, max(0, w - bw))
            y0 = random.randint(0, max(0, h - bh))
            img = img.copy()
            import PIL.ImageDraw as ImageDraw

            draw = ImageDraw.Draw(img)
            draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(0, 0, 0))
            return img

    return transforms.Compose(
        [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), Block(box_frac)]
    )


class RandomChoiceApply:
    """
    With probability p, apply exactly ONE transform from a list (uniformly).
    Otherwise, return image unchanged.
    """
    def __init__(self, transforms_list, p=0.3):
        self.transforms_list = transforms_list
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        t = random.choice(self.transforms_list)
        return t(img)
    

def make_train_image_augmentations(IMAGE_SIZE):
    aug_list = []

    # ----- your existing "light/mid" augs -----
    aug_list.append(transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5))

    aug_list.append(transforms.RandomApply(
        [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0))],
        p=0.6
    ))

    aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    aug_list.append(transforms.RandomApply(
        [transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)],
        p=0.6
    ))

    # mild JPEG
    aug_list.append(RandomJPEG(p=0.3, q_range=(60, 95)))

    # optional RandAugment
    try:
        from torchvision.transforms import RandAugment
        aug_list.append(transforms.RandomApply([RandAugment()], p=0.25))
    except Exception:
        pass

    # mild blur
    aug_list.append(transforms.RandomApply(
        [transforms.Lambda(lambda img: img.filter(
            ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.8))
        ))],
        p=0.25
    ))

    aug_list.append(RandomGaussianNoise(p=0.25, std=0.02))

    aug_list.append(transforms.RandomApply(
        [transforms.ColorJitter(brightness=(0.8, 1.2))],
        p=0.5
    ))

    # ----- NEW: heavy attack bank (covers your evaluation attacks) -----
    heavy_attacks = [
        make_jpeg_aug(IMAGE_SIZE),
        make_blur_aug(IMAGE_SIZE),
        make_down_up_attack(IMAGE_SIZE, downscale_frac=0.5),
        make_msg_app_combo(IMAGE_SIZE),
        make_random_crop_attack(IMAGE_SIZE, scale=(0.5, 0.9)),
        make_occlusion_block(IMAGE_SIZE, box_frac=0.25),
    ]

    # apply ONE heavy attack with some probability
    aug_list.append(RandomChoiceApply(heavy_attacks, p=0.30))

    # final resize safety
    final = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *aug_list])
    return final