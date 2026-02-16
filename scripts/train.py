"""
Main Training Script
Train DDPM on CIFAR-10
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import CIFARDataLoader
from diffusion import (
    CosineSchedule,
    DDPMSampler,
    ForwardDiffusion,
    LinearSchedule,
    NoiseSchedule,
)
from models import SimpleUNet
from tqdm import tqdm
from utils import (
    MetricTracker,
    TrainingLogger,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config,
    save_sample_grid,
    plot_training_curve
)


def parse_args():
    """Argparser"""
    parser = argparse.ArgumentParser(description="Train DDPM on CIFAR-10")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/my_config.yaml",
        help="Path to config file",
    )

    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )
    parser.add_argument(
        "--batch_size", type=int, default=None, help="Override batch size"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=None, help="Override number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Override learning rate"
    )

    args = parser.parse_args()

    return args


class DDPMTrainer:
    """
    DDPM Trainer.

    Handles the complete training loop including:
    - Model and optimizer setup
    - Training loop with loss computation
    - Checkpointing and logging
    - Sample generation
    """

    def __init__(self, config):
        self.config = config
        self.device = torch.device(
            config.device.accelerator if torch.cuda.is_available() else "cpu"
        )

        self.output_dir = Path(config.logging.output_dir) / config.logging.exp_name
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.sample_dir = self.output_dir / "samples"
        self.log_dir = self.output_dir / "logs"

        for dir_path in [self.checkpoint_dir, self.sample_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        save_config(config, self.output_dir / "config.yaml")

        # setup logger and metric tracker
        self.logger = TrainingLogger(self.log_dir, use_tensorboard=config.logging.use_tensorboard)
        self.metric_tracker = MetricTracker()

        # setup model
        self.model = self._build_model()
        self.model.to(self.device)

        num_params = sum(p.numel() for p in self.model.parameters())
        self.logger.log(f"Model parameters: {num_params:,}")

        # setup forward diffusion
        self.noise_schedule = self._create_noise_schedule()
        self.noise_schedule.to(self.device)

        self.forward_diffusion = ForwardDiffusion(self.noise_schedule)
        self.forward_diffusion.to(self.device)

        self.optimizer = self._build_optimizer()

        # Setup dataloader
        self.train_loader = self._get_dataloader()

        self.logger.log(f"Training batches per epoch: {len(self.train_loader)}")

        # Training state
        self.epoch = 0
        self.global_step = 0

    def _build_model(self):
        """Build Simple UNet model from config"""
        model = SimpleUNet(
            in_channels=self.config.model.in_channels,
            out_channels=self.config.model.out_channels,
            base_channels=self.config.model.base_channels,
            time_embed_dim=self.config.model.time_embed_dim,
        ).to(self.device)
        return model

    def _build_optimizer(self):
        """Build optimizer from config"""
        if self.config.optimizer.type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=tuple(self.config.optimizer.betas),
                eps=self.config.optimizer.eps,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.optimizer.type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                betas=tuple(self.config.optimizer.betas),
                eps=self.config.optimizer.eps,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer.type}")

        return optimizer

    def _get_dataloader(self):
        """Create CIFAR-10 dataloader."""
        dataset = CIFARDataLoader(root=self.config.data.data_root)
        dataloader = dataset.get_dataloader(
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=self.config.data.pin_memory,
        )
        return dataloader

    def _create_noise_schedule(self) -> NoiseSchedule:
        cfg = self.config.diffusion
        if cfg.schedule_type == "linear":
            return LinearSchedule(cfg.timesteps, beta_start=cfg.beta_start, beta_end=cfg.beta_end)
        elif cfg.schedule_type == "cosine":
            return CosineSchedule(cfg.timesteps, s=cfg.offset)
        else:
            raise ValueError(f"Unknown schedule type: {cfg.schedule_type}")

    def save(self, filename="checkpoint.pt"):
        """Save checkpoint."""
        save_path = self.checkpoint_dir / filename

        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            step=self.global_step,
            loss=self.metric_tracker.get_average("loss", last_n=100),
            save_path=str(save_path),
            config=self.config,
        )

        self.logger.log(f"Checkpoint saved: {filename}")

    def load(self, checkpoint_path: str):
        """Load checkpoint"""
        metadata = load_checkpoint(
            checkpoint_path, self.model, self.optimizer, device=str(self.device)
        )
        self.epoch = metadata["epoch"]
        self.global_step = metadata["step"]

        self.logger.log(f"Loaded checkpoint from epoch {self.epoch}")

    def compute_loss(self, x_0):
        """
        Compute training loss
        Args:
            x_0: [B, C, H, W] clean images

        Returns:
            loss: Scalar loss value
        """
        batch_size = x_0.shape[0]

        t = torch.randint(
            0,
            self.config.diffusion.timesteps,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        noise = torch.randn_like(x_0)

        x_t = self.forward_diffusion.q_sample(x_0, t, noise)
        predicted_noise = self.model(x_t, t)
        loss = nn.functional.mse_loss(predicted_noise, noise)
        return loss

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch + 1}/{self.config.training.num_epochs}",
        )

        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(self.device)

            # Forward pass
            loss = self.compute_loss(images)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.training.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )

            self.optimizer.step()
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            self.metric_tracker.update("loss", loss_value)

            pbar.set_postfix({"loss": f"{loss_value:.4f}"})

            if self.global_step % self.config.logging.log_every == 0:
                avg_loss = self.metric_tracker.get_average(
                    "loss", last_n=self.config.logging.log_every
                )
                self.logger.log_metrics(
                    self.global_step,
                    {"loss": avg_loss, "lr": self.optimizer.param_groups[0]["lr"]},
                )
            self.global_step += 1

        avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
        return avg_epoch_loss

    def train(self):
        """Main training loop."""
        self.logger.log("=" * 70)
        self.logger.log(
            f"Starting training for {self.config.training.num_epochs} epochs"
        )
        self.logger.log(f"Batch size: {self.config.training.batch_size}")
        self.logger.log(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.log("=" * 70)

        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch

            avg_loss = self.train_epoch()

            self.logger.log_epoch(
                epoch + 1, self.config.training.num_epochs, {"loss": avg_loss}
            )

            # Generate samples
            if (epoch + 1) % self.config.logging.sample_every == 0:
                self.logger.log("Generating samples")
                samples = self.generate_samples()

                save_path = self.sample_dir / f"samples_epoch_{epoch + 1}.png"
                save_sample_grid(samples, str(save_path))
                self.logger.log(f"Samples saved: {save_path.name}")

                # Log to tensorboard
                self.logger.log_images("samples", samples, self.global_step)

            # save checkpoint
            if (epoch + 1) % self.config.logging.save_every == 0:
                self.save(f"checkpoint_epoch_{epoch + 1}.pt")

        # Final checkpoint and samples
        self.logger.log("=" * 70)
        self.logger.log("Training complete!")
        self.save("checkpoint_final.pt")

        self.logger.log("Generating final samples")
        samples = self.generate_samples()
        save_sample_grid(samples, str(self.sample_dir / "samples_final.png"))
        plot_training_curve(
            self.metric_tracker.history["loss"],
            str(self.output_dir / "training_curve.png"),
        )

        self.logger.log("=" * 70)
        self.logger.close()

    @torch.no_grad()
    def generate_samples(self, num_samples=None):
        """Generate samples using trained model"""
        self.model.eval()

        if num_samples is None:
            num_samples = self.config.logging.num_samples

        sampler = DDPMSampler(self.model, self.forward_diffusion)

        samples = sampler.sample(
            shape=(num_samples, 3, 32, 32), device=str(self.device), progress_bar=True
        )

        return samples


def main():
    args = parse_args()

    config = load_config(args.config)
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.num_epochs is not None:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate

    trainer = DDPMTrainer(config)

    if args.resume:
        trainer.load(args.resume)

    trainer.train()


if __name__ == "__main__":
    main()
