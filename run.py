# Import necessary libraries
import argparse  # For parsing command-line arguments
import copy  # To make deep copies of objects
from tqdm import tqdm  # For progress bar
from statistics import mean, stdev  # For computing statistics
from sklearn import metrics  # For performance evaluation (ROC, AUC)

import torch  # PyTorch for deep learning

# Importing diffusion model utilities
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

# Import custom optimization and IO utilities
from optim_utils import *
from io_utils import *

# ------------------------- #
# Argument Parser Function  #
# ------------------------- #
def get_args_parser():
    parser = argparse.ArgumentParser("Detection Training Script", add_help=False)

    # General experiment settings
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='256x256_diffusion')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # Watermarking configuration
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)

    # Image distortion settings
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    return parser

# ------------------------- #
# Main Function             #
# ------------------------- #
def main(args):
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Format timestep string for DDIM inference
    args.timestep_respacing = f"ddim{args.num_inference_steps}"

    # Load diffusion model and configuration
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    # Use mixed precision if specified
    if args.use_fp16:
        model.convert_to_fp16()

    model.eval()  # Set model to evaluation mode

    # Define shape of the generated images
    shape = (args.num_images, 3, args.image_size, args.image_size)

    # Generate ground truth watermarking patch
    gt_patch = get_watermarking_pattern(None, args, device, shape)

    # Store results
    results = []
    no_w_metrics = []
    w_metrics = []

    # Generate and evaluate images for each seed
    for i in tqdm(range(args.start, args.end)):
        seed = i + args.gen_seed

        model_kwargs = {}
        if args.class_cond:
            classes = torch.randint(
                low=0, high=NUM_CLASSES, size=(args.num_images,), device=device
            )
            model_kwargs["y"] = classes

        # Generate image *without* watermark
        set_random_seed(seed)
        init_latents_no_w = torch.randn(*shape, device=device)
        outputs_no_w = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latents_no_w,
            model_kwargs=model_kwargs,
            device=device,
            return_image=True,
        ) # generate the image without watermark
        orig_image_no_w = outputs_no_w[0]

        # ------------------------------------------------------ #
        # Generate watermark and inject it into the latent space #
        # ------------------------------------------------------ #
        
        init_latents_w = copy.deepcopy(init_latents_no_w)
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = diffusion.ddim_sample_loop(
            model=model,
            shape=shape,
            noise=init_latents_w,
            model_kwargs=model_kwargs,
            device=device,
            return_image=True,
        ) # generate the watermarked image
        orig_image_w = outputs_w[0]

        # Apply image distortions for robustness evaluation
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(
            orig_image_no_w, orig_image_w, seed, args
        )

        # Reverse the diffusion process to extract latent representations
        reversed_latents_no_w = diffusion.ddim_reverse_sample_loop(
            model=model,
            shape=shape,
            image=orig_image_no_w_auged,
            model_kwargs=model_kwargs,
            device=device,
        )
        reversed_latents_w = diffusion.ddim_reverse_sample_loop(
            model=model,
            shape=shape,
            image=orig_image_w_auged,
            model_kwargs=model_kwargs,
            device=device,
        )

        # Evaluate watermark presence
        no_w_metric, w_metric = eval_watermark(
            reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args
        )

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric,
        })

        # Store metrics for ROC/AUC calculation (negated as higher score = less watermark)
        no_w_metrics.append(-no_w_metric)
        w_metrics.append(-w_metric)

    # Evaluation: ROC curve and AUC
    preds = no_w_metrics + w_metrics
    t_labels = [0] * len(no_w_metrics) + [1] * len(w_metrics)

    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    
    # TPR@1%FPR
    low = tpr[np.where(fpr < .01)[0][-1]]

    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')

# ------------------------- #
# Script Entry Point        #
# ------------------------- #
if __name__ == '__main__':
    # Initialize argument parser
    parser = argparse.ArgumentParser("Watermarking", parents=[get_args_parser()])
    args = parser.parse_args()

    # Update args with model defaults and specific config file
    args.__dict__.update(model_and_diffusion_defaults())
    args.__dict__.update(read_json(f'{args.model_id}.json'))

    # If no test inference step is specified, use training setting
    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    # Start main procedure
    main(args)
