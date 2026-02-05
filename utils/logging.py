"""
Logging Utilities
Training metrics, tensorboard logging
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.tensorboard.writer import SummaryWriter


class TrainingLogger:
    """
    Logger for training metrics and progress

    Logs to:
    - Console (print statements)
    - Text file (training.log)
    - Optional: TensorBoard

    Args:
        log_dir: Directory to save logs
        use_tensorboard: Whether to use TensorBoard
    """

    def __init__(self, log_dir: str, use_tensorboard: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.log_dir / "training.log"
        self._write_header()

        self.use_tensorboard = use_tensorboard
        self.tb_writer = None

        if use_tensorboard:
            try:
                self.tb_writer = SummaryWriter(
                    log_dir=str(self.log_dir / "tensorboard")
                )
                self.log("TensorBoard enabled")
            except ImportError:
                self.log("TensorBoard not available, skipping")
                self.use_tensorboard = False

    def _write_header(self):
        """Write header to log file"""
        with open(self.log_file, "w") as f:
            f.write("=" * 70 + "\n")
            f.write(f"Training Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 70 + "\n\n")

    def log(self, message: str, print_console: bool = True):
        """
        Log a message

        Args:
            message: Message to log
            print_console: Whether to print to console
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"

        if print_console:
            print(message)

        with open(self.log_file, "a") as f:
            f.write(log_line + "\n")

    def log_metrics(self, step: int, metrics: Dict[str, float], prefix: str = "train"):
        """
        Log training metrics

        Args:
            step: Current step
            metrics: Dictionary of metric_name -> value
            prefix: Prefix for metric names (e.g., "train", "val")
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(f"Step {step} | {metrics_str}", print_console=False)

        if self.use_tensorboard and self.tb_writer:
            for metric_name, value in metrics.items():
                self.tb_writer.add_scalar(f"{prefix}/{metric_name}", value, step)

    def log_epoch(self, epoch: int, total_epochs: int, metrics: Dict[str, float]):
        """
        Log epoch summary

        Args:
            epoch: Current epoch
            total_epochs: Total epochs
            metrics: Dictionary of metrics
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        message = f"Epoch [{epoch}/{total_epochs}] | {metrics_str}"
        self.log(message)

    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """
        Log images to TensorBoard

        Args:
            tag: Image tag
            images: [B, C, H, W] tensor of images
            step: Current step
        """
        if self.use_tensorboard and self.tb_writer:
            images = (images * 0.5 + 0.5).clamp(0, 1)

            from torchvision.utils import make_grid

            grid = make_grid(images, nrow=8)
            self.tb_writer.add_image(tag, grid, step)

    def close(self):
        """Close logger and TensorBoard writer"""
        if self.use_tensorboard and self.tb_writer:
            self.tb_writer.close()


class MetricTracker:
    """
    Track metrics during training
    """

    def __init__(self):
        self.metrics = {}
        self.history = {}

    def update(self, metric_name: str, value: float):
        """
        Update a metric

        Args:
            metric_name: Name of metric
            value: Current value
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
            self.history[metric_name] = []

        self.metrics[metric_name].append(value)
        self.history[metric_name].append(value)

    def get_average(self, metric_name: str, last_n: Optional[int] = None) -> float:
        """
        Get average of a metric

        Args:
            metric_name: Name of metric
            last_n: Average over last N values (None for all)

        Returns:
            Average value
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return 0.0

        values = self.metrics[metric_name]
        if last_n:
            values = values[-last_n:]

        return sum(values) / len(values)

    def reset(self, metric_name: Optional[str] = None):
        """
        Reset metrics

        Args:
            metric_name: Specific metric to reset (None for all)
        """
        if metric_name:
            if metric_name in self.metrics:
                self.metrics[metric_name] = []
        else:
            self.metrics = {k: [] for k in self.metrics.keys()}

    def get_summary(self, last_n: Optional[int] = None) -> Dict[str, float]:
        """
        Get summary of all metrics

        Args:
            last_n: Average over last N values

        Returns:
            Dictionary of metric averages
        """
        return {name: self.get_average(name, last_n) for name in self.metrics.keys()}
