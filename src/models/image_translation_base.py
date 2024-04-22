from typing import Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection


class ImageTranslationBase(LightningModule):
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
        lr_monitor: str = "val/loss",
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
        self.save_hyperparameters(logger=False)  # store hyperparameters

        self.net = net

        self.loss_fn = loss_fn

        self.metrics = MetricCollection(
            {
                "train/loss": MeanMetric(),
                "val/loss": MeanMetric(),
                "test/loss": MeanMetric(),
            }
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x: Input tensor representing a batch of images, shape [batch_size, 3, n, n].

        Returns:
            A tensor of transformed images, shape [batch_size, 3, n, n].
        """
        # assert x.shape[1] == 3, "Input tensor should have 3 color channels"
        return self.net(x)

    def step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single step with the given batch, computing loss.

        Args:
            batch: Tuple of (input_images, target_images). Both tensors should have the shape [batch_size, 3, n, n].

        Returns:
            Tuple of (loss, predicted_images)
        """
        x, y = batch
        # assert x.shape == y.shape, "Input and target images must have the same dimensions"
        y_hat = self.forward(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Training step for processing one batch of data."""
        loss, _ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Validation step for processing one batch of data."""
        loss, _ = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Test step for processing one batch of data."""
        loss, _ = self.step(batch)
        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

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
