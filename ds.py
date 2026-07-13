from __future__ import annotations

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

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from attack import make_blur_aug, make_clean_aug, make_down_up_attack, make_jpeg_aug, make_msg_app_combo, make_occlusion_block, make_random_crop_attack

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



# def make_train_image_augmentations_deprecated(IMAGE_SIZE):
#     # Compose PIL-based augmentations (randomly applied)
#     aug_list = []
#     # random rotation small
#     aug_list.append(
#         transforms.RandomApply([transforms.RandomRotation(degrees=15)], p=0.5)
#     )
#     # random resized crop (sometimes)
#     aug_list.append(
#         transforms.RandomApply(
#             [transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0))], p=0.6
#         )
#     )
#     # horizontal flip
#     aug_list.append(transforms.RandomHorizontalFlip(p=0.5))
#     # color jitter
#     aug_list.append(
#         transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.1, 0.05)], p=0.6)
#     )
#     # JPEG
#     aug_list.append(RandomJPEG(p=0.3, q_range=(60, 95)))
#     # RandAugment (if torchvision supports it) - wrapped
#     try:
#         from torchvision.transforms import RandAugment

#         aug_list.append(transforms.RandomApply([RandAugment()], p=0.25))
#     except Exception:
#         pass

#     # Gaussian blur sometimes
#     aug_list.append(
#         transforms.RandomApply(
#             [
#                 lambda img: img.filter(
#                     ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.8))
#                 )
#             ],
#             p=0.25,
#         )
#     )
#     # Add gaussian pixel noise sometimes
#     aug_list.append(RandomGaussianNoise(p=0.25, std=0.02))
#     # brightness jitter more finely (RandomApply)
#     aug_list.append(
#         transforms.RandomApply([transforms.ColorJitter(brightness=(0.8, 1.2))], p=0.5)
#     )

#     # final: ensure image is resized to IMAGE_SIZE (if not already)
#     final = transforms.Compose([transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), *aug_list])
#     return final


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
        include_psnr_l1=False,
        include_mask_patch=False,
        psnr_return_prob=False,
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
        self.include_psnr_l1 = include_psnr_l1
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
        # move to corr`ect dtype & device for the unet/vae as the user did earlier:
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

            l1_metric = eval_watermark(
                reversed_latents,  # single sample
                self.watermarking_mask,
                self.gt_patch,
                w_measurement="l1_complex",
            )

            if self.psnr_return_prob:
                psnr_metric = psnr_to_prob_sigmoid(
                    psnr_metric, threshold=-4.0, scale=1.0
                )

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
            mask_ch = self.watermarking_mask.to(dtype=torch.float32).squeeze(
                0
            )  # (1,H,W)
            gt_patch_ch = self.gt_patch.to(dtype=torch.float32).squeeze(0)  # (1,H,W)
            fft_ch = torch.cat([fft_ch, mask_ch, gt_patch_ch], dim=0)  # (2*C+2, H, W)

            if self.include_psnr_l1:
                return (
                    fft_ch,
                    torch.tensor(psnr_metric, dtype=torch.float32),
                    torch.tensor(l1_metric, dtype=torch.float32),
                    torch.tensor(label, dtype=torch.long),
                )
            return fft_ch, torch.tensor(label, dtype=torch.long)

        if self.include_psnr_l1:
            return (
                fft_ch,
                torch.tensor(psnr_metric, dtype=torch.float32),
                torch.tensor(l1_metric, dtype=torch.float32),
                torch.tensor(label, dtype=torch.long),
            )

        return fft_ch, torch.tensor(label, dtype=torch.long)






import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset



@dataclass(frozen=True)
class AttackTask:
    """One meta-learning task = one augmentation / attack family."""
    name: str
    image_aug: Optional[Callable] = None  # PIL -> PIL (or compatible)
    image_aug_prob: float = 1.0           # set 1.0 so the task is deterministic


def build_meta_attack_tasks(
    image_size: int,
    task_names: Optional[Union[str, Sequence[str]]] = None,
) -> List[AttackTask]:
    """
    Build the candidate task pool for meta-training.

    task_names:
      - None or "default": clean + downup50 + crop + jpeg
      - "all": every registered attack task
      - sequence of names: explicit task pool in that order
    """
    registry = {
        "clean": AttackTask(
            name="clean",
            image_aug=make_clean_aug(image_size),
            image_aug_prob=1.0,
        ),
        "downup50": AttackTask(
            name="downup50",
            image_aug=make_down_up_attack(image_size, downscale_frac=0.5),
            image_aug_prob=1.0,
        ),
        "crop": AttackTask(
            name="crop",
            image_aug=make_random_crop_attack(image_size, scale=(0.5, 0.9)),
            image_aug_prob=1.0,
        ),
        "jpeg": AttackTask(
            name="jpeg",
            image_aug=make_jpeg_aug(image_size),
            image_aug_prob=1.0,
        ),
        "blur": AttackTask(
            name="blur",
            image_aug=make_blur_aug(image_size),
            image_aug_prob=1.0,
        ),
        "msg_app": AttackTask(
            name="msg_app",
            image_aug=make_msg_app_combo(image_size),
            image_aug_prob=1.0,
        ),
        "occlusion": AttackTask(
            name="occlusion",
            image_aug=make_occlusion_block(image_size, box_frac=0.25),
            image_aug_prob=1.0,
        ),
    }

    default_names = ["clean", "downup50", "crop", "jpeg"]
    if task_names is None or task_names == "default":
        names = default_names
    elif task_names == "all":
        names = list(registry.keys())
    else:
        names = list(task_names)

    unknown = [name for name in names if name not in registry]
    if unknown:
        raise ValueError(
            f"Unknown meta attack task(s): {unknown}. "
            f"Available tasks: {list(registry.keys())}"
        )
    return [registry[name] for name in names]


def _to_sample_dict(sample: Tuple) -> Dict[str, Any]:
    """
    Convert outputs from WatermarkOnTheFlyDataset into a unified dict.

    Your base dataset returns one of:
      (x, y) or (x, psnr, l1, y)

    Returns:
      {"x": x_tensor, "y": y_tensor, "extra": {...}}
    """
    if not isinstance(sample, (tuple, list)):
        raise TypeError(f"Expected tuple/list from base dataset, got {type(sample)}")

    if len(sample) == 2:
        x, y = sample
        extra = {}
    elif len(sample) == 4:
        x, psnr, l1, y = sample
        extra = {"psnr": psnr, "l1": l1}
    else:
        raise ValueError(f"Unexpected base dataset return length: {len(sample)}")

    if not torch.is_tensor(x):
        raise TypeError(f"x must be a Tensor, got {type(x)}")
    if not torch.is_tensor(y):
        raise TypeError(f"y must be a Tensor, got {type(y)}")

    return {"x": x, "y": y, "extra": extra}


def _collate_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stack list of {"x","y","extra"} into one batch dict.
    """
    xs = torch.stack([s["x"] for s in samples], dim=0)               # [B, ...]
    ys = torch.stack([s["y"] for s in samples], dim=0).long()        # [B]

    # extras: stack tensors if present
    extra: Dict[str, Any] = {}
    keys = set()
    for s in samples:
        keys |= set(s["extra"].keys())

    for k in keys:
        vals = [s["extra"].get(k, None) for s in samples]
        if all(v is None for v in vals):
            continue
        if torch.is_tensor(vals[0]):
            extra[k] = torch.stack(vals, dim=0)
        else:
            extra[k] = vals

    return {"x": xs, "y": ys, "extra": extra}


import random
import torch
from torch.utils.data import Dataset


class WatermarkMetaTaskDataset(Dataset):
    def __init__(
        self,
        ds,                      # base WatermarkOnTheFlyDataset
        tasks,                   # List[AttackTask]
        n_support: int,
        n_query: int,
        tasks_per_epoch: int = 200,
        seed: int = 0,
        task_sampling: str = "uniform",
        residual_agent=None,
        residual_base_sampling: str = "uniform",
        residual_config: Optional[Dict[str, Any]] = None,
    ):
        self.ds = ds
        self.tasks = list(tasks)
        if len(self.tasks) == 0:
            raise ValueError("WatermarkMetaTaskDataset requires at least one task.")
        self.n_support = int(n_support)
        self.n_query = int(n_query)
        self.tasks_per_epoch = int(tasks_per_epoch)
        self.task_sampling = task_sampling
        self.rng = random.Random(seed)
        self.residual_base_sampling = residual_base_sampling
        self.residual_agent = residual_agent
        self._last_residual_info = None
        self._scheduler_controller_modes = {
            "hard_task",
            "progress",
            "bandit_ucb",
            "bandit_thompson",
            "ats",
            "bass",
            "asr",
            "derts_proxy",
            "gcp_proxy",
        }

        if self.task_sampling in ("residual", "llm_residual") and self.residual_agent is None:
            from residual_agent import (
                LLMResidualTaskController,
                LLMTaskControllerConfig,
                ResidualTaskController,
                ResidualTaskControllerConfig,
            )

            if self.task_sampling == "llm_residual":
                cfg = LLMTaskControllerConfig(seed=seed, **(residual_config or {}))
                self.residual_agent = LLMResidualTaskController(
                    num_tasks=len(self.tasks),
                    task_names=[task.name for task in self.tasks],
                    config=cfg,
                )
            else:
                cfg = ResidualTaskControllerConfig(
                    seed=seed,
                    **(residual_config or {}),
                )
                self.residual_agent = ResidualTaskController(
                    num_tasks=len(self.tasks),
                    task_names=[task.name for task in self.tasks],
                    config=cfg,
                )
        elif self.task_sampling in self._scheduler_controller_modes and self.residual_agent is None:
            from scheduler_baselines import (
                BaselineSchedulerConfig,
                BaselineTaskSchedulerController,
            )

            scheduler_config = dict(residual_config or {})
            scheduler_config.pop("mode", None)
            cfg = BaselineSchedulerConfig(
                mode=self.task_sampling,
                seed=seed,
                **scheduler_config,
            )
            self.residual_agent = BaselineTaskSchedulerController(
                num_tasks=len(self.tasks),
                task_names=[task.name for task in self.tasks],
                config=cfg,
            )

    def __len__(self):
        return self.tasks_per_epoch

    def _choose_task_id(self, idx: int) -> int:
        if self.task_sampling == "cycle":
            return idx % len(self.tasks)
        if self.task_sampling == "uniform":
            return self.rng.randrange(len(self.tasks))
        if self.task_sampling in ("residual", "llm_residual"):
            if self.residual_agent is None:
                raise RuntimeError(
                    f"task_sampling={self.task_sampling!r} requires residual_agent."
                )

            base_mode = self.residual_base_sampling
            cycle_idx = idx % len(self.tasks)
            base_weights = self.residual_agent.base_weights(base_mode, cycle_idx)
            task_id, info = self.residual_agent.sample(base_weights)
            self._last_residual_info = info
            return task_id
        if self.task_sampling in self._scheduler_controller_modes:
            if self.residual_agent is None:
                raise RuntimeError(
                    f"task_sampling={self.task_sampling!r} requires a scheduler controller."
                )
            task_id, info = self.residual_agent.sample()
            self._last_residual_info = info
            return task_id
        raise ValueError(
            "Unknown task_sampling="
            f"{self.task_sampling!r}; expected uniform, cycle, hard_task, progress, "
            "bandit_ucb, bandit_thompson, ats, bass, asr, derts_proxy, "
            "gcp_proxy, residual, or llm_residual."
        )

    def update_task_feedback(
        self,
        task_id: int,
        *,
        loss: Optional[float] = None,
        val_gain: Optional[float] = None,
        fail_rate: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Optional[float]:
        """
        Feed downstream meta-training feedback to the residual controller.

        Existing training code can call this after query/outer loss is known:
            meta_ds.update_task_feedback(task_id, loss=float(outer_loss))

        No-op unless task_sampling uses a residual controller or a residual_agent was supplied.
        """
        if self.residual_agent is None:
            return None
        return self.residual_agent.update(
            task_id,
            loss=loss,
            val_gain=val_gain,
            fail_rate=fail_rate,
            reward=reward,
        )

    def update_task_feedback_from_batch(
        self,
        task_batch: Dict[str, Any],
        *,
        loss: Optional[float] = None,
        val_gain: Optional[float] = None,
        fail_rate: Optional[float] = None,
        reward: Optional[float] = None,
    ) -> Optional[float]:
        task_id = task_batch.get("task_id")
        if torch.is_tensor(task_id):
            task_id = int(task_id.view(-1)[0].item())
        elif isinstance(task_id, (list, tuple)):
            task_id = int(task_id[0])
        else:
            task_id = int(task_id)
        return self.update_task_feedback(
            task_id,
            loss=loss,
            val_gain=val_gain,
            fail_rate=fail_rate,
            reward=reward,
        )

    def residual_snapshot(self) -> Optional[Dict[str, Any]]:
        if self.residual_agent is None:
            return None
        return self.residual_agent.snapshot()

    def update_residual_global_context(self, **kwargs) -> None:
        if self.residual_agent is None:
            return None
        update_fn = getattr(self.residual_agent, "update_global_context", None)
        if update_fn is None:
            return None
        return update_fn(**kwargs)

    def _sample_indices(self, k: int):
        n = len(self.ds)
        if k <= n:
            return self.rng.sample(range(n), k)
        return [self.rng.randrange(n) for _ in range(k)]

    def _pack(self, samples):
        xs, ys = [], []
        extra = {}

        for s in samples:
            if len(s) == 2:
                x, y = s
            elif len(s) == 4:
                x, psnr, l1, y = s
                extra.setdefault("psnr", []).append(psnr)
                extra.setdefault("l1", []).append(l1)
            else:
                raise ValueError(f"Unexpected sample format: len={len(s)}")

            xs.append(x)
            ys.append(y)

        out = {
            "x": torch.stack(xs, 0),
            "y": torch.stack(ys, 0).long().view(-1),
        }
        if extra:
            out["extra"] = {k: torch.stack(v, 0).view(-1) for k, v in extra.items()}
        return out

    def __getitem__(self, idx):
        # choose task
        task_id = self._choose_task_id(idx)
        task = self.tasks[task_id]

        # save old aug settings
        old_aug = getattr(self.ds, "image_aug", None)
        old_prob = getattr(self.ds, "image_aug_prob", None)

        # IMPORTANT: set task-specific aug
        self.ds.image_aug = task.image_aug
        self.ds.image_aug_prob = float(task.image_aug_prob)

        try:
            total = self.n_support + self.n_query
            inds = self._sample_indices(total)
            s_inds = inds[: self.n_support]
            q_inds = inds[self.n_support :]

            support_samples = [self.ds[i] for i in s_inds]
            query_samples   = [self.ds[i] for i in q_inds]

            out = {
                "support": self._pack(support_samples),
                "query": self._pack(query_samples),
                "task_name": task.name,
                "task_id": task_id,
            }
            if self.task_sampling in ("residual", "llm_residual") or self.task_sampling in self._scheduler_controller_modes:
                out["scheduler_info"] = self._last_residual_info
            return out
        finally:
            # restore (CRITICAL)
            self.ds.image_aug = old_aug
            self.ds.image_aug_prob = old_prob
