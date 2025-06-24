import torch
from scipy.stats import norm,truncnorm
from functools import reduce
from scipy.special import betainc
import numpy as np

class Gaussian_Shading:
    def __init__(self, ch_factor, hw_factor, fpr, user_number):
        """
        Initializes the Gaussian Shading watermarking scheme.

        Args:
            ch_factor (int): Channel tiling factor (e.g., 2 means divide channels into 2 tiles).
            hw_factor (int): Height/Width tiling factor.
            fpr (float): Target false positive rate.
            user_number (int): Number of users or identities to support (used in threshold tuning).

        Typical Usage (as used in inference loop):
        >>> watermark = Gaussian_Shading(2, 2, 1e-5, 100)
        >>> latent_with_watermark = watermark.create_watermark_and_return_w()
        >>> outputs = pipe(prompt, latents=latent_with_watermark)
        >>> reversed_latent = pipe.forward_diffusion(image_latent)
        >>> acc = watermark.eval_watermark(reversed_latent)
        """
        self.ch = ch_factor
        self.hw = hw_factor
        self.key = None
        self.watermark = None

        # Latent shape is assumed to be (4, 64, 64) = 16384 elements
        self.latentlength = 4 * 64 * 64

        # Length of the binary watermark to embed
        self.marklength = self.latentlength // (self.ch * self.hw * self.hw)

        # Voting threshold (how many tiles must agree to call a 1)
        self.threshold = 1 if self.hw == 1 and self.ch == 1 else self.ch * self.hw * self.hw // 2

        # True positive counters
        self.tp_onebit_count = 0
        self.tp_bits_count = 0

        # Detection thresholds computed from FPR
        self.tau_onebit = None
        self.tau_bits = None

        # Compute detection thresholds based on false positive rate (FPR)
        for i in range(self.marklength):
            fpr_onebit = betainc(i + 1, self.marklength - i, 0.5)
            fpr_bits = fpr_onebit * user_number
            if fpr_onebit <= fpr and self.tau_onebit is None:
                self.tau_onebit = i / self.marklength
            if fpr_bits <= fpr and self.tau_bits is None:
                self.tau_bits = i / self.marklength

    def truncSampling(self, message):
        """
        Generates a distribution-preserving latent vector by truncated Gaussian sampling.

        Args:
            message (np.array): Flattened binary message (e.g., watermark + key XOR).

        Returns:
            torch.Tensor: Latent tensor of shape (1, 4, 64, 64) with half precision.
        """
        z = np.zeros(self.latentlength)
        denominator = 2.0  # Two intervals for binary encoding

        # Compute interval boundaries (quantiles) in normal distribution
        ppf = [norm.ppf(j / denominator) for j in range(int(denominator) + 1)]

        # For each latent dimension, sample from truncated interval [ppf[bit], ppf[bit+1]]
        for i in range(self.latentlength):
            bit = int(message[i])
            z[i] = truncnorm.rvs(ppf[bit], ppf[bit + 1])

        # Reshape to latent format expected by diffusion model
        z = torch.from_numpy(z).reshape(1, 4, 64, 64).half()
        return z.cuda()

    def create_watermark_and_return_w(self):
        """
        Creates a new watermark and returns the embedded latent vector.

        Returns:
            torch.Tensor: Latent vector `w` with embedded watermark.
        """
        # Generate binary key and watermark
        self.key = torch.randint(0, 2, [1, 4, 64, 64]).cuda()
        self.watermark = torch.randint(0, 2, [1, 4 // self.ch, 64 // self.hw, 64 // self.hw]).cuda()

        # Tile the watermark across latent shape to get sd
        sd = self.watermark.repeat(1, self.ch, self.hw, self.hw)

        # XOR with key to produce encrypted message m
        m = ((sd + self.key) % 2).flatten().cpu().numpy()

        # Use truncated sampling to embed m into a latent
        w = self.truncSampling(m)
        return w

    def diffusion_inverse(self, watermark_sd):
        """
        Recovers watermark from decrypted latent using a majority voting mechanism.

        Args:
            watermark_sd (torch.Tensor): Decrypted latent of shape (1, 4, 64, 64).

        Returns:
            torch.Tensor: Predicted watermark of shape matching self.watermark.
        """
        # Split into tiles across channel, height, and width
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw

        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw

        split_dim1 = torch.cat(torch.split(watermark_sd, tuple(ch_list), dim=1), dim=0)
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=2), dim=0)
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=3), dim=0)

        # Majority vote across all tiles
        vote = torch.sum(split_dim3, dim=0).clone()
        vote[vote <= self.threshold] = 0
        vote[vote > self.threshold] = 1
        return vote

    def eval_watermark(self, reversed_m):
        """
        Evaluates whether a watermark can be extracted from the reversed latent.

        Args:
            reversed_m (torch.Tensor): Latent recovered from image (post DDIM inversion).

        Returns:
            float: Accuracy score (ratio of correctly recovered watermark bits).
        """
        reversed_m = (reversed_m > 0).int()
        reversed_sd = (reversed_m + self.key) % 2
        reversed_watermark = self.diffusion_inverse(reversed_sd)

        correct = (reversed_watermark == self.watermark).float().mean().item()

        if correct >= self.tau_onebit:
            self.tp_onebit_count += 1
        if correct >= self.tau_bits:
            self.tp_bits_count += 1

        return correct

    def get_tpr(self):
        """
        Returns:
            Tuple[int, int]: True positive counts for one-bit and full-bit thresholds.
        """
        return self.tp_onebit_count, self.tp_bits_count