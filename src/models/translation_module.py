from typing import Tuple, Optional

import wandb
import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection

from matplotlib import pyplot as plt
from ..utils.utils import undo_transforms


class TranslationModule(LightningModule):
    """
    LightningModule for image-to-image translation tasks, such as transforming
    digital images to appear as if shot on CineStill800T film.
    """

    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_monitor: str = "train/loss",  # Check why we can't have val/loss
    ) -> None:
        """Initialises `TranslationModule`.

        Args:
            net: The neural network model to use for image transformation.
            loss: Loss function used for training.
            optimizer: Optimizer function.
            scheduler: Learning rate scheduler (optional).
            lr_monitor: Metric to monitor for learning rate scheduler (default: val/loss).
        """
        super().__init__()

        # Store hyperparameters
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = net
        self.loss = loss

        # Initialise metrics
        self.metrics = MetricCollection(
            {
                "train/loss": MeanMetric(),
                "val/loss": MeanMetric(),
                "test/loss": MeanMetric(),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor representing a batch of images, shape [batch_size, 3, n, n].

        Returns:
            A tensor of transformed images, shape [batch_size, 3, n, n].
        """
        return self.net(x)

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single step with the given batch, computing loss.

        Args:
            batch: Tuple of (input_images, target_images). Both tensors should
            have the shape [batch_size, 3, n, n].

        Returns:
            Tuple of (loss, predicted_images)
        """
        film, digital = batch
        film_predicted = self.forward(digital)
        loss = self.loss(film_predicted, film)

        return loss, film, digital, film_predicted

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step for processing one batch of data."""
        loss, film, digital, film_predicted = self.step(batch)

        # Log training loss and images
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_batch(film, digital, film_predicted, key="train/images")

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step for processing one batch of data."""
        loss, film, digital, film_predicted = self.step(batch)

        # Log validation loss and images
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self._log_batch(film, digital, film_predicted, key="val/images")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int):
        """Test step for processing one batch of data."""
        loss, _, _, _ = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def _log_batch(
        self,
        film: torch.Tensor,
        digital: torch.Tensor,
        film_predicted: torch.Tensor,
        key: Optional[str] = None,
    ):
        if hasattr(self.logger.experiment, "log"):
            # Create figure
            batch_size = film.size(0)
            fig, axs = plt.subplots(
                nrows=3, ncols=batch_size, figsize=(4 * batch_size, 8)
            )
            if axs.ndim == 1:
                axs = axs[:, None]
            fig.tight_layout(pad=1.0)
            for i in range(batch_size):
                axs[0, i].imshow(undo_transforms(digital[i].cpu()).numpy())
                axs[1, i].imshow(undo_transforms(film[i].cpu()).numpy())
                axs[2, i].imshow(undo_transforms(film_predicted[i].cpu()).numpy())
            axs[0, 0].set_ylabel("Digital")
            axs[1, 0].set_ylabel("Film (Ground Truth)")
            axs[2, 0].set_ylabel("Film (Predicted)")

            # Log to W&B
            self.logger.experiment.log({key: wandb.Image(fig)})

            # Close figure
            plt.close()

    def configure_optimizers(self):
        """Setup the optimizer and the LR scheduler."""
        optimizer = self.hparams.optimizer(params=self.net.parameters())
        if self.hparams.scheduler:
            scheduler = {
                "scheduler": self.hparams.scheduler(optimizer),
                "monitor": self.hparams.lr_monitor,
                "interval": "epoch",
                "frequency": 1,
            }
            return [optimizer], [scheduler]
        return optimizer
