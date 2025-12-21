#!/usr/bin/env python3

import os
import argparse
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn.functional as F
import open_clip
import csv

# verifier_dataset_coco_octoweb: Mean CLIP similarity:  0.3273
# verifier_dataset_coco_ring: Mean CLIP similarity:  0.3296
# clean_coco: Mean CLIP similarity:  0.3302

# verifier_dataset_stablediff_octoweb: Mean CLIP similarity:  0.3599
# verifier_dataset_stablediff_ring: Mean CLIP similarity:  0.3614
# clean_stablediff: Mean CLIP similarity:  0.3615

class ImagePromptDataset(Dataset):
    """
    Dataset that aligns images in a folder with prompts from chosen_prompts.txt.

    Assumes:
      - image_dir has N images
      - prompts_file has N lines, one prompt per image, in the same order as generation
      - Images are sorted by filename and matched index-wise to prompts
        (i.e. prompts[i] corresponds to sorted_images[i]).
    """

    def __init__(self, image_dir: str, prompts_file: str, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Collect images
        self.image_files: List[str] = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
        )
        if not self.image_files:
            raise RuntimeError(f"No images found in {image_dir}")

        # Load prompts
        with open(prompts_file, "r", encoding="utf-8") as f:
            self.prompts: List[str] = [line.strip() for line in f if line.strip()]

        if len(self.image_files) != len(self.prompts):
            raise ValueError(
                f"Number of images ({len(self.image_files)}) and prompts "
                f"({len(self.prompts)}) do not match.\n"
                f"image_dir: {image_dir}\n"
                f"prompts_file: {prompts_file}"
            )

        print(f"[INFO] Found {len(self.image_files)} images and {len(self.prompts)} prompts.")
        print("[INFO] Using sorted filenames as the order; "
              "assume generation used the same order as chosen_prompts.txt.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx: int):
        fname = self.image_files[idx]
        img_path = os.path.join(self.image_dir, fname)
        img = Image.open(img_path).convert("RGB")
        prompt = self.prompts[idx]

        if self.transform is not None:
            img_t = self.transform(img)
        else:
            img_t = img  # should not happen for CLIP

        return img_t, prompt, fname


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP (open_clip) image–text similarity given "
                    "an image folder and chosen_prompts.txt."
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default="./verifier_dataset_stablediff_octoweb/clean",
        # required=True,
        help="Directory containing generated/watermarked images",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./verifier_dataset_stablediff_octoweb/chosen_prompts.txt",
        # required=True,
        help="Path to chosen_prompts.txt (one prompt per line)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for CLIP encoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-g-14",
        help="open_clip model name (e.g. 'ViT-B-32', 'ViT-L-14')",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default="laion2b_s12b_b42k",
        help="open_clip pretrained tag (e.g. 'laion2b_s34b_b79k', 'laion2b_s32b_b82k')",
    )
    parser.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="Optional path to save per-image CLIP scores as CSV",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"Image dir not found: {args.image_dir}")
    if not os.path.isfile(args.prompts_file):
        raise FileNotFoundError(f"Prompts file not found: {args.prompts_file}")

    print("====================================================")
    print("Calculating open_clip image–text similarity...")
    print(f"  Images:       {args.image_dir}")
    print(f"  Prompts file: {args.prompts_file}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Device:       {args.device}")
    print(f"  Model:        {args.model_name}")
    print(f"  Pretrained:   {args.pretrained}")
    print("====================================================")

    device = args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"

    # Load CLIP model + preprocess + tokenizer
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name,
        pretrained=args.pretrained,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(args.model_name)
    model.eval()

    dataset = ImagePromptDataset(args.image_dir, args.prompts_file, transform=preprocess)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    all_sims = []
    all_fnames = []

    with torch.no_grad():
        for img_batch, prompts, fnames in loader:
            img_batch = img_batch.to(device)  # (B,C,H,W)

            # Encode images
            image_features = model.encode_image(img_batch)       # (B,D)
            image_features = F.normalize(image_features, dim=-1)

            # Encode text
            text_tokens = tokenizer(list(prompts)).to(device)    # (B,L)
            text_features = model.encode_text(text_tokens)       # (B,D)
            text_features = F.normalize(text_features, dim=-1)

            # Diagonal cosine similarity: image i vs prompt i
            sims = (image_features * text_features).sum(dim=-1)  # (B,)

            all_sims.append(sims.cpu())
            all_fnames.extend(list(fnames))

    sims = torch.cat(all_sims, dim=0)

    mean_sim = sims.mean().item()
    std_sim = sims.std().item()
    min_sim = sims.min().item()
    max_sim = sims.max().item()

    print("\n==================== RESULT ====================")
    print(f"Num samples:           {len(sims)}")
    print(f"Mean CLIP similarity:  {mean_sim:.4f}")
    print(f"Std CLIP similarity:   {std_sim:.4f}")
    print(f"Min CLIP similarity:   {min_sim:.4f}")
    print(f"Max CLIP similarity:   {max_sim:.4f}")
    print("================================================\n")

    if args.save_csv is not None:
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "clip_similarity"])
            for fname, s in zip(all_fnames, sims.tolist()):
                writer.writerow([fname, s])
        print(f"[INFO] Saved per-image CLIP scores to: {args.save_csv}")


if __name__ == "__main__":
    main()
