from typing import Tuple

import wandb
import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection


class TranslationModule(LightningModule):
    """Base LightningModule for image-to-image translation tasks, such as transforming digital
    images to appear as if shot on CineStill800T film.

    This class is designed to be architecture and dataset agnostic, focusing on handling square
    images with three data channels. Subclasses should implement specific model architectures and
    training strategies.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module = torch.nn.MSELoss(),
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_monitor: str = "train/loss", # Check why we can't have val/loss
    ) -> None:
        """Initialize the base module.

        Args:
            net: The neural network model to use for image transformation.
            optimizer: Optimizer function.
            loss_fn: Loss function used for training.
            scheduler: Learning rate scheduler (optional).
            lr_monitor: Metric to monitor for learning rate scheduler (default: val/loss).
        """
        super().__init__()

        # Store hyperparameters
        self.save_hyperparameters(logger=False, ignore=["net", "loss_fn"])
        self.net = net
        self.loss_fn = loss_fn

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

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
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
        loss = self.loss_fn(film_predicted, film)
        return loss, film, film_predicted

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step for processing one batch of data."""
        loss, y, y_hat = self.step(batch)
        
        # Log training loss
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log images 
        self._log_images(y, "train/target")
        self._log_images(y_hat, "train/predicted")

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step for processing one batch of data."""
        loss, y, y_hat = self.step(batch)

        # Log validation loss
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Log images
        self._log_images(y, "val/target")
        self._log_images(y_hat, "val/predicted")

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int):
        """Test step for processing one batch of data."""
        loss, _, _ = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
  
    def _log_images(self, images: torch.Tensor, key: str):
        if hasattr(self.logger.experiment, "log"):
            self.logger.experiment.log({key: [wandb.Image(img) for img in images]})
    
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
