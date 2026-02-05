"""
U-Net for noise prediction
"""

import torch
import torch.nn as nn

from .time_embedding import TimestepEmbedding


class ResidualBlock(nn.Module):
    """
    Residual block with time embedding injection.

    Architecture:
        x --> [GN -> SiLU -> Conv] --> [+ time_emb] --> [GN -> SiLU -> Conv] --> + x --> out
    """

    def __init__(self, in_channels: int, out_channels: int, time_embed_dim: int):
        super().__init__()

        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.time_proj = nn.Linear(time_embed_dim, out_channels)

        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.act = nn.SiLU()

        if in_channels != out_channels:
            self.residual = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
            t_emb: [B, time_embed_dim]
        """
        h = self.norm1(x)
        h = self.act(h)
        h = self.conv1(h)

        time_bias = self.time_proj(self.act(t_emb))
        h = h + time_bias.view(time_bias.shape[0], time_bias.shape[1], 1, 1)

        h = self.norm2(h)
        h = self.act(h)
        h = self.conv2(h)

        return h + self.residual(x)


class SimpleUNet(nn.Module):
    """
    A simplified U-Net for noise prediction

    Architecture:
    - Encoder: Downsample with conv layers
    - Middle: Process at lowest resolution
    - Decoder: Upsample with transposed conv
    - Skip connections: Concatenate encoder features to decoder
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_channels: int = 64,
        time_embed_dim: int = 128,
    ):
        super().__init__()

        self.time_embed = TimestepEmbedding(time_embed_dim)

        # Initial projection
        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.enc1 = self._make_layer(base_channels, base_channels, time_embed_dim)
        self.down1 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, stride=2, padding=1
        )

        self.enc2 = self._make_layer(base_channels, base_channels * 2, time_embed_dim)
        self.down2 = nn.Conv2d(
            base_channels * 2, base_channels * 2, kernel_size=3, stride=2, padding=1
        )

        self.enc3 = self._make_layer(
            base_channels * 2, base_channels * 4, time_embed_dim
        )
        self.down3 = nn.Conv2d(
            base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1
        )

        # Middle
        self.middle = self._make_layer(
            base_channels * 4, base_channels * 4, time_embed_dim
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            base_channels * 4, base_channels * 4, kernel_size=4, stride=2, padding=1
        )
        self.dec3 = self._make_layer(
            base_channels * 8, base_channels * 2, time_embed_dim
        )

        self.up2 = nn.ConvTranspose2d(
            base_channels * 2, base_channels * 2, kernel_size=4, stride=2, padding=1
        )
        self.dec2 = self._make_layer(base_channels * 4, base_channels, time_embed_dim)

        self.up1 = nn.ConvTranspose2d(
            base_channels, base_channels, kernel_size=4, stride=2, padding=1
        )
        self.dec1 = self._make_layer(base_channels * 2, base_channels, time_embed_dim)

        # output
        self.out = nn.Conv2d(base_channels, out_channels, 1)

    def _make_layer(
        self, in_channels: int, out_channels: int, embed_dim: int
    ) -> ResidualBlock:
        return ResidualBlock(
            in_channels=in_channels, out_channels=out_channels, time_embed_dim=embed_dim
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] noisy images
            t: [B] timesteps
        Returns:
            noise_pred: [B, C, H, W] predicted noise
        """
        t_emb = self.time_embed(t)

        x = self.conv_in(x)
        h1 = self.enc1(x, t_emb)
        h = self.down1(h1)

        h2 = self.enc2(h, t_emb)
        h = self.down2(h2)

        h3 = self.enc3(h, t_emb)
        h = self.down3(h3)

        h = self.middle(h, t_emb)

        h = self.up3(h)
        h = torch.cat([h, h3], dim=1)
        h = self.dec3(h, t_emb)

        h = self.up2(h)
        h = torch.cat([h, h2], dim=1)
        h = self.dec2(h, t_emb)

        h = self.up1(h)
        h = torch.cat([h, h1], dim=1)
        h = self.dec1(h, t_emb)

        return self.out(h)
