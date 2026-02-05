"""
Visualization Utilities
Plot training curves, generate sample grids
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid, save_image


def save_sample_grid(
    samples: torch.Tensor, save_path: str, nrow: int = 8, denormalize: bool = True
):
    """
    Save a grid of generated samples

    Args:
        samples: [B, C, H, W] tensor of images
        save_path: Path to save image
        nrow: Number of images per row
        denormalize: Whether to denormalize from [-1, 1] to [0, 1]
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if denormalize:
        samples = (samples * 0.5 + 0.5).clamp(0, 1)

    save_image(samples, save_path, nrow=nrow, padding=2)


def plot_training_curve(
    losses: List[float],
    save_path: str,
    title: str = "Training Loss",
    window_size: int = 100,
):
    """
    Plot training curve with moving average

    Args:
        losses: List of loss values
        save_path: Path to save plot
        title: Plot title
        window_size: Window size for moving average
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses, alpha=0.3, label="Loss")
    if len(losses) > window_size:
        moving_avg = np.convolve(
            losses, np.ones(window_size) / window_size, mode="valid"
        )
        axes[0].plot(
            range(window_size - 1, len(losses)),
            moving_avg,
            linewidth=2,
            label=f"Moving Avg (window={window_size})",
        )

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"{title} (Linear Scale)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses, alpha=0.3)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_title(f"{title} (Log Scale)")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_denoising_process(
    intermediates: List[dict], save_path: str, num_samples: int = 4
):
    """
    Visualize the denoising process

    Args:
        intermediates: List of dicts with 't', 'x_t', 'pred_x0' keys
        save_path: Path to save visualization
        num_samples: Number of samples to visualize
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_steps = len(intermediates)

    fig, axes = plt.subplots(
        num_samples, num_steps, figsize=(3 * num_steps, 3 * num_samples)
    )

    if num_samples == 1:
        axes = axes[None, :]

    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    for sample_idx in range(min(num_samples, intermediates[0]["x_t"].shape[0])):
        for step_idx, intermediate in enumerate(intermediates):
            ax = axes[sample_idx, step_idx]

            img = denorm(intermediate["x_t"][sample_idx])
            img = img.permute(1, 2, 0).numpy()

            ax.imshow(img)

            if sample_idx == 0:
                ax.set_title(f"t={intermediate['t']}", fontsize=10)

            ax.axis("off")

    plt.suptitle("Progressive Denoising", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
