"""
Checkpointing Utilities - Save and load model checkpoints
"""

from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    save_path: str,
    ema_model: Optional[nn.Module] = None,
    config: Optional[Any] = None,
    **kwargs,
):
    """
    Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        step: Current step
        loss: Current loss
        save_path: Path to save checkpoint
        ema_model: Optional EMA model
        config: Optional config object
        **kwargs: Additional items to save
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    if ema_model is not None:
        checkpoint["ema_model_state_dict"] = ema_model.state_dict()

    if config is not None:
        checkpoint["config"] = config

    checkpoint.update(kwargs)

    torch.save(checkpoint, save_path)


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    ema_model: Optional[nn.Module] = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    """
    Load training checkpoint

    Args:
        checkpoint_path: Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        ema_model: Optional EMA model to load state into
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint metadata
    """
    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if ema_model is not None and "ema_model_state_dict" in checkpoint:
        ema_model.load_state_dict(checkpoint["ema_model_state_dict"])

    metadata = {
        "epoch": checkpoint.get("epoch", 0),
        "step": checkpoint.get("step", 0),
        "loss": checkpoint.get("loss", 0.0),
        "config": checkpoint.get("config", None),
    }

    for key, value in checkpoint.items():
        if key not in [
            "model_state_dict",
            "optimizer_state_dict",
            "ema_model_state_dict",
            "epoch",
            "step",
            "loss",
            "config",
        ]:
            metadata[key] = value

    return metadata


class EMA:
    """
    Exponential Moving Average of model parameters

    Maintains a shadow copy of model parameters that is updated
    with exponential moving average

    Args:
        model: Model to track
        decay: EMA decay rate (default 0.9999)
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay

        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Apply EMA parameters to model (for inference)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        """Get state dictionary"""
        return self.shadow

    def load_state_dict(self, state_dict):
        """Load state dictionary"""
        self.shadow = state_dict
