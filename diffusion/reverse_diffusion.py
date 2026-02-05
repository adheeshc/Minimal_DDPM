"""
Reverse Diffusion Process (Sampling)
"""

import torch
import torch.nn as nn
from forward_diffusion import ForwardDiffusion
from tqdm import tqdm


class DDPMSampler:
    """
    DDPM sampling algorithm

    Algorithm 2:
    1. x_T ~ N(0, I)                          [Start from pure noise]
    2. for t = T, ..., 1 do
    3.   z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
    4.   x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z
    5. end for
    6. return x_0

    Args:
        model: Noise prediction network ε_θ(x_t, t)
        forward_diffusion: Forward diffusion process (for schedule)
        variance_type: "posterior" or "beta" for variance schedule
    """

    def __init__(
        self,
        model: nn.Module,
        forward_diffusion: ForwardDiffusion,
        variance_type: str = "posterior",
    ):
        self.model = model
        self.forward = forward_diffusion
        self.schedule = forward_diffusion.schedule
        self.timesteps = forward_diffusion.timesteps
        self.variance_type = variance_type

    @torch.no_grad()
    def p_sample_step(
        self, x_t: torch.Tensor, t: torch.Tensor, clip_denoised: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single denoising step: x_t → x_{t-1}.
        x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z

        Args:
            x_t: [B, C, H, W] noisy images at timestep t
            t: [B] current timestep
            clip_denoised: Whether to clip predicted x_0 to [-1, 1]

        Returns:
            x_{t-1}: [B, C, H, W] less noisy images
            pred_x0: [B, C, H, W] predicted clean image (for visualization)
        """
        predicted_noise = self.model(x_t, t)

        alpha_t = self.schedule.alphas[t].view(-1, 1, 1, 1)
        alpha_bar_t = self.schedule.alphas_cumprod[t].view(-1, 1, 1, 1)
        beta_t = self.schedule.betas[t].view(-1, 1, 1, 1)

        # Predict x_0 from x_t and predicted noise
        # x̂_0 = (x_t - √(1-ᾱ_t) * ε_θ) / √ᾱ_t
        pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(
            alpha_bar_t
        )

        if clip_denoised:
            pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

        # Compute mean
        # μ_θ = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t))
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
        )

        # Get variance
        if self.variance_type == "posterior":
            variance = self.schedule.posterior_variance[t].view(-1, 1, 1, 1)
        else:
            variance = beta_t

        # z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean

        return x_t_minus_1, pred_x0

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        device: str = "cuda",
        return_intermediates: bool = False,
        progress_bar: bool = True,
    ) -> torch.Tensor | tuple[torch.Tensor, list[dict] | None]:
        """
        Full sampling procedure (Algorithm 2)

        1. x_T ~ N(0, I)                          [Start from pure noise]
        2. for t = T, ..., 1 do
        3.   z ~ N(0, I) if t > 1, else z = 0     [Sample noise]
        4.   x_{t-1} = 1/√α_t * (x_t - (1-α_t)/√(1-ᾱ_t) * ε_θ(x_t,t)) + σ_t * z
        5. end for
        6. return x_0

        Args:
            shape: (batch_size, channels, height, width)
            device: Device to run on
            return_intermediates: If True, return intermediate steps
            progress_bar: Show progress bar

        Returns:
            samples: [B, C, H, W] generated images
            intermediates: Optional list of (t, x_t, pred_x0) tuples
        """
        self.model.eval()

        x_t = torch.randn(shape, device=device)
        intermediates = [] if return_intermediates else None

        iterator = reversed(range(self.timesteps))
        if progress_bar:
            iterator = tqdm(list(iterator), desc="Sampling")

        for t_idx in iterator:
            t = torch.full((shape[0],), t_idx, device=device, dtype=torch.long)
            x_t, pred_x0 = self.p_sample_step(x_t, t)

            if intermediates is not None and t_idx % (self.timesteps // 10) == 0:
                intermediates.append(
                    {
                        "t": t_idx,
                        "x_t": x_t.cpu().clone(),
                        "pred_x0": pred_x0.cpu().clone(),
                    }
                )

        if return_intermediates:
            return x_t, intermediates
        return x_t


class DDIMSampler:
    """
    DDIM sampling - deterministic and faster

    Args:
        model: Noise prediction network
        forward_diffusion: Forward diffusion process
        eta: Stochasticity parameter (0 = deterministic, 1 = DDPM)
    """

    def __init__(
        self, model: nn.Module, forward_diffusion: ForwardDiffusion, eta: float = 0.0
    ):
        self.model = model
        self.forward = forward_diffusion
        self.schedule = forward_diffusion.schedule
        self.timesteps = forward_diffusion.timesteps
        self.eta = eta

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        num_steps: int = 50,
        device: str = "cuda",
        progress_bar: bool = True,
    ) -> torch.Tensor:
        """
        DDIM sampling

        Args:
            shape: (batch_size, channels, height, width)
            num_steps: Number of denoising steps
            device: Device to run on
            progress_bar: Show progress bar

        Returns:
            samples: [B, C, H, W] generated images
        """
        self.model.eval()

        step_size = self.timesteps // num_steps
        timesteps = torch.arange(0, self.timesteps, step_size, device=device)
        timesteps = torch.flip(timesteps, [0])

        x_t = torch.randn(shape, device=device)

        iterator = enumerate(timesteps)
        if progress_bar:
            iterator = tqdm(list(iterator), desc=f"DDIM Sampling ({num_steps} steps)")

        for i, t in iterator:
            t_batch = torch.full((shape[0],), t.item(), device=device, dtype=torch.long)

            noise_pred = self.model(x_t, t_batch)

            t_prev = (
                timesteps[i + 1]
                if i < len(timesteps) - 1
                else torch.tensor(0, device=device)
            )

            alpha_t = self.schedule.alphas_cumprod[t]
            alpha_prev = (
                self.schedule.alphas_cumprod[t_prev]
                if t_prev > 0
                else torch.tensor(1.0)
            )

            x_0_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(
                alpha_t
            )
            x_0_pred = torch.clamp(x_0_pred, -1, 1)

            dir_xt = torch.sqrt(1 - alpha_prev) * noise_pred

            if self.eta > 0 and i < len(timesteps) - 1:
                sigma = self.eta * torch.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )
                noise = torch.randn_like(x_t)
                x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise
            else:
                x_t = torch.sqrt(alpha_prev) * x_0_pred + dir_xt

        return x_t
