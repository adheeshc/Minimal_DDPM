"""
Time Embedding Module
Sinusoidal position encoding for timestep conditioning
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """
    Creates a continuous representation of discrete timesteps that:
    - Contains both high and low frequency components
    - Allows model to generalize to unseen timesteps
    - Is deterministic and doesn't require learning

    Formula:
        PE(t, 2i) = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))

    Args:
        dim: Embedding dimension (must be even)
    """

    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of timesteps (0 to T-1)
        Returns:
            embeddings: [B, embedding_dim] tensor
        """

        device = timesteps.device
        half_dim = self.embedding_dim // 2

        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale
        )

        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class TimeEmbeddingMLP(nn.Module):
    """
    MLP to project sinusoidal embeddings to higher dimension and apply
    non-linearity to allow network to learn complex time-dependent patterns.

    Args:
        time_embed_dim: Input dimension (from sinusoidal embedding)
        hidden_dim: Hidden layer dimension (typically 4x time_embed_dim)
    """

    def __init__(self, time_embed_dim: int, hidden_dim: int):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, time_embed_dim),
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_emb: [B, time_embed_dim] sinusoidal embeddings

        Returns:
            processed: [B, time_embed_dim] processed embeddings
        """
        return self.mlp(t_emb)


class TimestepEmbedding(nn.Module):
    """
    Combines sinusoidal encoding with learned MLP projection

    Args:
        dim: Embedding dimension
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

        self.sin_embed = SinusoidalTimeEmbedding(dim)
        self.mlp = TimeEmbeddingMLP(dim, dim * 4)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: [B] tensor of integer timesteps

        Returns:
            embeddings: [B, dim] processed time embeddings
        """
        t_emb = self.sin_embed(timesteps)  # Sinusoidal encoding
        t_emb = self.mlp(t_emb)  # MLP projection

        return t_emb
