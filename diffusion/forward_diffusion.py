"""
Forward Diffusion Process
"""

import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataloader import CIFARDataLoader

from diffusion.noise_schedules import CosineSchedule, LinearSchedule, NoiseSchedule


class ForwardDiffusion:
    """
    Forward diffusion process

    Key equation:
        q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)

    Args:
        noise_schedule: NoiseSchedule object with precomputed values
    """

    def __init__(self, noise_schedule: NoiseSchedule) -> None:
        self.timesteps = noise_schedule.timesteps
        self.schedule = noise_schedule

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample from q(x_t | x_0) directly

        Args:
            x_0: [B, C, H, W] clean images
            t: [B] timestep indices (0 to T-1)
            noise: [B, C, H, W] optional pre-sampled noise

        Returns:
            x_t: [B, C, H, W] noisy images at timestep t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = (
            self.schedule.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1).to(x_0.device)
        )
        sqrt_one_minus_alpha = (
            self.schedule.sqrt_one_minus_alphas_cumprod[t]
            .view(-1, 1, 1, 1)
            .to(x_0.device)
        )
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def q_posterior(
        self, x_0: torch.Tensor, t: torch.Tensor, x_t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and variance of posterior q(x_{t-1} | x_t, x_0)

        Args:
            x_0: [B, C, H, W] clean images
            x_t: [B, C, H, W] noisy images at timestep t
            t: [B] timestep indices

        Returns:
            mean: [B, C, H, W] posterior mean
            variance: [B, C, H, W] posterior variance
        """
        posterior_mean_coef1 = self.schedule.posterior_mean_coef1[t].view(-1, 1, 1, 1)
        posterior_mean_coef2 = self.schedule.posterior_mean_coef2[t].view(-1, 1, 1, 1)
        posterior_variance = self.schedule.posterior_variance[t].view(-1, 1, 1, 1)

        posterior_mean = posterior_mean_coef1 * x_0 + posterior_mean_coef2 * x_t

        return posterior_mean, posterior_variance

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute signal-to-noise ratio at timestep t
        SNR(t) = ᾱ_t / (1 - ᾱ_t)

        Args:
            t: [B] timestep indices

        Returns:
            snr: [B] signal-to-noise ratios
        """
        alphas_cumprod_t = self.schedule.alphas_cumprod[t]
        return alphas_cumprod_t / (1 - alphas_cumprod_t)

    def to(self, device: torch.device) -> "ForwardDiffusion":
        """Move schedule to device"""
        self.schedule.to(device)
        return self

    def visualize_diffusion_steps(
        self, num_images: int = 4, num_steps: int = 8
    ) -> None:
        images, _ = CIFARDataLoader().get_samples(num_images=num_images)
        timesteps = torch.linspace(0, self.timesteps - 1, num_steps).long()

        fig, axes = plt.subplots(
            num_images, num_steps, figsize=(num_steps * 2, num_images * 2)
        )

        for i in range(num_images):
            for j, t in enumerate(timesteps):
                t_batch = t.unsqueeze(0)
                noisy = self.q_sample(images[i : i + 1], t_batch)
                snr = self.get_snr(t).item()

                img = noisy[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()

                axes[i, j].imshow(img)
                axes[i, j].set_title(f"t={t}\nSNR={snr:.2f}")
                axes[i, j].axis("off")
                if i == 0:
                    axes[i, j].set_title(f"t={t.item()}", fontsize=10)

        plt.suptitle("Forward Diffusion Process", fontsize=14)
        plt.tight_layout()
        os.makedirs("./outputs/forward_process", exist_ok=True)
        plt.savefig(
            f"./outputs/forward_process/forward_process_{self.schedule.__class__.__name__}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)


if __name__ == "__main__":
    linear_schedule = LinearSchedule(timesteps=1000)
    cosine_schedule = CosineSchedule(timesteps=1000)
    forward1 = ForwardDiffusion(linear_schedule)
    forward2 = ForwardDiffusion(cosine_schedule)

    forward1.visualize_diffusion_steps()
    forward2.visualize_diffusion_steps()
