from typing import Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MetricCollection

from .loss.auto_translate import AutoTranslateLoss
from .net.auto_translate import AutoTranslateNet


class AutoTranslationModule(LightningModule):
    """
    Base module for auto-translation models as seen in "Semi-Supervised
    Raw-to-Raw Mapping": https://arxiv.org/pdf/2106.13883
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        lr_monitor: str = "val/loss",
    ) -> None:
        """Initialize the base module.

        Args:
            net: The neural network model to use for image transformation.
            optimizer: Optimizer function.
            loss: Loss function used for training.
            scheduler: Learning rate scheduler (optional).
            lr_monitor: Metric to monitor for learning rate scheduler (default: val/loss).
        """
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = AutoTranslateNet()
        self.loss = AutoTranslateLoss()

        self.metrics = MetricCollection(
            {
                "train/loss": MeanMetric(),
                "val/loss": MeanMetric(),
                "test/loss": MeanMetric(),
            }
        )

    def forward(
        self, film: torch.Tensor, digital: torch.Tensor, paired: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            digital: Input tensor representing a batch of unpaired images, shape [B_1, 3, n, n].
            film: Input tensor representing a batch of film images, shape [B_2, 3, n, n].
            paired: Input tensor representing a batch of paired images, shape [B_3, 3, n, n, 2].

        Returns:
            digital_reconstructed: Transformed digital images, shape [B_1, 3, n, n].
            film_reconstructed: Transformed film images, shape [B_2, 3, n, n].
            digital_to_film: Transformed digital images from the paired film, shape [B_1, 3, n, n].
            film_to_digital: Transformed film images from the paired digital, shape [B_2, 3, n, n].
            paired_encoder_representation: Latent space representations of the paired images over all encoder layers. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].
        """
        return self.net(film, digital, paired)

    def step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single step with the given batch, computing loss.

        Args:
            batch: Tuple of (digital, film, paired_images). Both tensors should have the shape [B_1/B_2, 3, n, n] respectively.

        Returns Tuple of:
            loss: The computed loss value.
            digital_transformed: Transformed digital images from the paired film, shape [B_1, 3, n, n].
        """
        # Unpack the batch
        film, digital, paired = batch
        film_paired, digital_paired = paired

        # Remove artificial batch dimension
        film, digital = film.squeeze(0), digital.squeeze(0)
        film_paired = film_paired.squeeze(0)
        digital_paired = digital_paired.squeeze(0)
        paired = (film_paired, digital_paired)

        # Forward pass through the model
        (
            digital_reconstructed,
            film_reconstructed,
            digital_to_film,
            film_to_digital,
            paired_encoder_representations,
        ) = self(film, digital, paired)

        # Compute the loss
        loss = self.loss(
            digital,
            film,
            paired,
            digital_reconstructed,
            film_reconstructed,
            digital_to_film,
            film_to_digital,
            paired_encoder_representations,
        )

        return loss, film_paired, digital_to_film

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Training step for processing one batch of data."""
        loss, _, _ = self.step(batch)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Validation step for processing one batch of data."""
        loss, _, _ = self.step(batch)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Test step for processing one batch of data."""
        loss, _, _ = self.step(batch)
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


if __name__ == "__main__":
    # Test the AutoTranslateNet
    model = AutoTranslationModule(
        net=AutoTranslateNet(), optimizer=torch.optim.Adam, scheduler=None
    )
    batch = (
        torch.randn(2, 3, 256, 256),
        torch.randn(2, 3, 256, 256),
        torch.randn(2, 2, 3, 256, 256),
    )
    model.training_step(batch, 0)
