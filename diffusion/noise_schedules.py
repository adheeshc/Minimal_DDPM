"""
Noise Schedules for Diffusion Models
Different beta_t schedules for forward diffusion
"""

import math
import os

import matplotlib.pyplot as plt
import torch


class NoiseSchedule:
    """Base class for noise schedules"""

    def __init__(self, timesteps: int) -> None:
        self.timesteps = timesteps
        self.betas = self._compute_betas()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), self.alphas_cumprod[:-1]]
        )

        # For forward process
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For reverse process
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        # Posterior variance
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_variance = torch.clamp(self.posterior_variance, min=1e-20)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance)
        self.posterior_mean_coef1 = (
            self.betas
            * torch.sqrt(self.alphas_cumprod_prev)
            / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * torch.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def _compute_betas(self) -> torch.Tensor:
        """Compute beta schedule"""
        raise NotImplementedError

    def to(self, device: torch.device) -> "NoiseSchedule":
        """Move all tensors to device"""
        for attr_name, attr in self.__dict__.items():
            if isinstance(attr, torch.Tensor):
                setattr(self, attr_name, attr.to(device))
        return self

    def visualize_schedule(self, output_path: str | None = None) -> None:
        """Visualize the noise schedule. Uses precomputed attributes."""
        schedule_name = self.__class__.__name__.replace("Schedule", "")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot betas
        axes[0, 0].plot(self.betas.cpu().numpy(), label=schedule_name)
        axes[0, 0].set_title("Beta Schedules")
        axes[0, 0].set_xlabel("Timestep")
        axes[0, 0].set_ylabel("Beta")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot alphas
        axes[0, 1].plot(self.alphas_cumprod.cpu().numpy(), label=schedule_name)
        axes[0, 1].set_title("Cumulative Alpha")
        axes[0, 1].set_xlabel("Timestep")
        axes[0, 1].set_ylabel("Alpha")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot signal coefficient
        axes[1, 0].plot(self.sqrt_alphas_cumprod.cpu().numpy(), label=schedule_name)
        axes[1, 0].set_title("Signal Coefficient (sqrt(alpha))")
        axes[1, 0].set_xlabel("Timestep")
        axes[1, 0].set_ylabel("sqrt(alpha)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot Signal to Noise Ratio
        snr = self.alphas_cumprod / (1 - self.alphas_cumprod)
        axes[1, 1].plot(snr.cpu().numpy(), label=schedule_name)
        axes[1, 1].set_title("Signal to Noise Ratio")
        axes[1, 1].set_xlabel("Timestep")
        axes[1, 1].set_ylabel("SNR")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        if output_path is None:
            output_path = (
                f"./outputs/noise_schedules/{schedule_name.lower()}_noise_schedules.png"
            )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


class LinearSchedule(NoiseSchedule):
    """
    Linear noise schedule from DDPM paper

    Args:
        timesteps: Number of diffusion steps T
        beta_start: Starting value (default 0.0001)
        beta_end: Ending value (default 0.02)
    """

    def __init__(
        self, timesteps: int, beta_start: float = 0.0001, beta_end: float = 0.02
    ) -> None:
        self.beta_start = beta_start
        self.beta_end = beta_end
        super().__init__(timesteps)

    def _compute_betas(self) -> torch.Tensor:
        """Linear schedule"""
        return torch.linspace(
            self.beta_start, self.beta_end, self.timesteps, dtype=torch.float32
        )


class CosineSchedule(NoiseSchedule):
    """
    Cosine noise schedule from Improved DDPM

    Args:
        timesteps: Number of diffusion steps T
        s: Small offset (default 0.008)
    """

    def __init__(self, timesteps: int, s: float = 0.008) -> None:
        self.s = s
        super().__init__(timesteps)

    def _compute_betas(self) -> torch.Tensor:
        """Cosine schedule"""
        steps = self.timesteps + 1
        t = torch.linspace(0, self.timesteps, steps, dtype=torch.float32)
        func_t = (
            torch.cos(((t / self.timesteps) + self.s) / (1 + self.s) * math.pi * 0.5)
            ** 2
        )
        alphas = func_t / func_t[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        return torch.clamp(betas, min=0.0001, max=0.9999)


def compare_schedules(
    schedules: list[NoiseSchedule],
    output_path: str = "./outputs/noise_schedules/noise_schedules_comparison.png",
) -> None:
    """Compare multiple noise schedules on the same plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for schedule in schedules:
        label = schedule.__class__.__name__.replace("Schedule", "")

        axes[0, 0].plot(schedule.betas.cpu().numpy(), label=label)
        axes[0, 1].plot(schedule.alphas_cumprod.cpu().numpy(), label=label)
        axes[1, 0].plot(schedule.sqrt_alphas_cumprod.cpu().numpy(), label=label)

        snr = schedule.alphas_cumprod / (1 - schedule.alphas_cumprod)
        axes[1, 1].plot(snr.cpu().numpy(), label=label)

    axes[0, 0].set_title("Beta Schedules")
    axes[0, 0].set_xlabel("Timestep")
    axes[0, 0].set_ylabel("Beta")

    axes[0, 1].set_title("Cumulative Alpha")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Alpha")

    axes[1, 0].set_title("Signal Coefficient (sqrt(alpha))")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("sqrt(alpha)")

    axes[1, 1].set_title("Signal to Noise Ratio")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("SNR")
    axes[1, 1].set_yscale("log")

    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    timesteps = 1000
    linear = LinearSchedule(timesteps, beta_start=0.0001, beta_end=0.02)
    cosine = CosineSchedule(timesteps, s=0.008)

    # Individual plots
    linear.visualize_schedule()
    cosine.visualize_schedule()

    # Comparison plot
    compare_schedules([linear, cosine])
