#!/usr/bin/env python3

import os
import argparse
from pytorch_fid.fid_score import calculate_fid_given_paths

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#
# FID score: 83.5059 -> Tree-Ring
# FID score: 83.2103 -> SpiderMark
# FID score: 81.7437 -> clean
# FID score: 81.7612-> dft_single
# FID score: 83.3128 -> dwt_dct

def main():
    parser = argparse.ArgumentParser(description="Compute FID between two folders")
    parser.add_argument(
        "--real_dir",
        type=str,
        default="./fid_outputs/coco/ground_truth",
        help="Directory containing real/clean images",
    )
    parser.add_argument(
        "--gen_dir",
        type=str,
        default="./watermark_dataset_dwt_dct",
        help="Directory containing generated/watermarked images",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=48,
        help="Batch size for FID calculation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'",
    )
    parser.add_argument(
        "--dims",
        type=int,
        default=2048,
        help="Feature dimensionality (Inception v3 default = 2048)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of dataloader workers",
    )

    args = parser.parse_args()

    # Check directories
    if not os.path.isdir(args.real_dir):
        raise FileNotFoundError(f"Real/ground-truth directory not found: {args.real_dir}")
    if not os.path.isdir(args.gen_dir):
        raise FileNotFoundError(f"Generated/watermarked directory not found: {args.gen_dir}")

    print("====================================================")
    print("Calculating FID...")
    print(f"  Real images:      {args.real_dir}")
    print(f"  Generated images: {args.gen_dir}")
    print(f"  Batch size:       {args.batch_size}")
    print(f"  Device:           {args.device}")
    print("====================================================")

    fid_value = calculate_fid_given_paths(
        [args.real_dir, args.gen_dir],
        batch_size=args.batch_size,
        device=args.device,
        dims=args.dims,
        num_workers=args.num_workers,
    )

    print("\n==================== RESULT ====================")
    print(f"FID score: {fid_value:.4f}")
    print("===============================================\n")


if __name__ == "__main__":
    main()
