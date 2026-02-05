"""
Configuration Utilities
Load and manage YAML configurations
"""

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    """Model architecture configuration"""

    image_size: int = 32
    in_channels: int = 3
    out_channels: int = 3
    base_channels: int = 64
    channel_multipliers: list = field(default_factory=lambda: [1, 2, 4])
    num_res_blocks: int = 2
    time_embed_dim: int = 128
    dropout: float = 0.0
    use_attention: bool = True
    attention_resolutions: list = field(default_factory=lambda: [16])


@dataclass
class DiffusionConfig:
    """Diffusion process configuration"""

    timesteps: int = 1000
    beta_start: float = 0.0001
    beta_end: float = 0.02
    schedule_type: str = "linear"


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 0.0002
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    ema_decay: float = 0.9999
    use_ema: bool = True


@dataclass
class DataConfig:
    """Data loading configuration"""

    dataset: str = "cifar10"
    data_root: str = "./data"
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    type: str = "adam"
    betas: list = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration"""

    type: str = "constant"
    warmup_steps: int = 0


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration"""

    output_dir: str = "./outputs"
    exp_name: str = "ddpm_cifar10"
    save_every: int = 10
    sample_every: int = 5
    log_every: int = 100
    num_samples: int = 64


@dataclass
class DeviceConfig:
    """Device configuration"""

    accelerator: str = "cuda"
    mixed_precision: bool = False


@dataclass
class SamplingConfig:
    """Sampling configuration"""

    num_inference_steps: int = 1000
    eta: float = 0.0
    variance_type: str = "posterior"


@dataclass
class Config:
    """Complete configuration."""

    model: ModelConfig = field(default_factory=ModelConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Config object with all settings
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        config_dict = yaml.safe_load(f)

    config = Config(
        model=ModelConfig(**config_dict.get("model", {})),
        diffusion=DiffusionConfig(**config_dict.get("diffusion", {})),
        training=TrainingConfig(**config_dict.get("training", {})),
        data=DataConfig(**config_dict.get("data", {})),
        optimizer=OptimizerConfig(**config_dict.get("optimizer", {})),
        scheduler=SchedulerConfig(**config_dict.get("scheduler", {})),
        logging=LoggingConfig(**config_dict.get("logging", {})),
        device=DeviceConfig(**config_dict.get("device", {})),
        sampling=SamplingConfig(**config_dict.get("sampling", {})),
    )

    return config


def save_config(config: Config, save_path: str):
    """
    Save configuration to YAML file.

    Args:
        config: Config object
        save_path: Path to save YAML file
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    config_dict = {
        "model": config.model.__dict__,
        "diffusion": config.diffusion.__dict__,
        "training": config.training.__dict__,
        "data": config.data.__dict__,
        "optimizer": config.optimizer.__dict__,
        "scheduler": config.scheduler.__dict__,
        "logging": config.logging.__dict__,
        "device": config.device.__dict__,
        "sampling": config.sampling.__dict__,
    }

    with open(save_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
