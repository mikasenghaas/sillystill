from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics import MetricCollection
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from ..eval import PieAPP

from .loss.auto_translate import AutoTranslateLoss
from .net.auto_translate import AutoTranslateNet
from .base_module import BaseModule

import wandb
from matplotlib import pyplot as plt
from PIL.Image import Image as PILImage


class AutoTranslationModule(BaseModule):
    """
    Base module for auto-translation models as seen in "Semi-Supervised
    Raw-to-Raw Mapping": https://arxiv.org/pdf/2106.13883
    """

    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        augment: float = 0.0,
        training_patch_size: int = 128,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
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
        super().__init__(
            augment=augment,
            training_patch_size=training_patch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_monitor=lr_monitor,
        )
        self.save_hyperparameters(logger=False, ignore=["net", "loss"])
        self.net = net
        self.loss = loss

        metrics = MetricCollection(
            {
                "ssim": SSIM(),
                "psnr": PSNR(),
                "lpips": LPIPS(),
                "pieapp": PieAPP(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

    def forward(
        self,
        film: torch.Tensor,
        digital: torch.Tensor,
        film_paired: torch.Tensor,
        digital_paired: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            digital: Input tensor representing a batch of unpaired images, shape [B_1, 3, n, n].
            film: Input tensor representing a batch of film images, shape [B_2, 3, n, n].
            film_paired: Input tensor representing a batch of paired images, shape [B_3, 3, n, n].
            digital_paired: Input tensor representing a batch of paired images, shape [B_3, 3, n, n].

        Returns:
            film_reconstructed: Transformed film images, shape [B_2, 3, n, n].
            digital_reconstructed: Transformed digital images, shape [B_1, 3, n, n].
            film_to_digital: Transformed film images from the paired digital, shape [B_2, 3, n, n].
            digital_to_film: Transformed digital images from the paired film, shape [B_1, 3, n, n].
            paired_encoder_representation: Latent space representations of the paired images over all encoder layers. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].
        """
        return self.net(film, digital, film_paired, digital_paired)

    def step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        transforms: torch.nn.Module,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Perform a single step with the given batch, computing loss.

        Args:
            batch: Tuple of (digital, film, paired_images). Both tensors should have the shape [B_1/B_2, 3, n, n] respectively.

        Returns Tuple of:
            loss: The computed loss value.
            digital_transformed: Transformed digital images from the paired film, shape [B_1, 3, n, n].
            component_losses (reconstruction_loss, encoder_loss, paired_loss): Tuple of the individual loss components.
        """
        # Unpack the batch
        film, digital, paired = batch

        # Remove artificial batch dimension
        film, digital, paired = film.squeeze(0), digital.squeeze(0), paired.squeeze(0)

        # Apply transforms
        film = transforms(film)
        digital = transforms(digital)
        paired = transforms(paired)

        # Unpack the paired images
        film_paired, digital_paired = paired

        # Forward pass through the model
        (
            film_reconstructed,
            digital_reconstructed,
            film_to_digital,
            digital_to_film,
            paired_encoder_representations,
        ) = self(film, digital, film_paired, digital_paired)

        # Compute the loss
        losses = self.loss(
            film,
            digital,
            film_paired,
            digital_paired,
            film_reconstructed,
            digital_reconstructed,
            digital_to_film,
            film_to_digital,
            paired_encoder_representations,
        )

        return (
            losses,
            film,
            digital,
            film_paired,
            digital_paired,
            film_reconstructed,
            digital_reconstructed,
            digital_to_film,
            film_to_digital,
        )

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Training step for processing one batch of data."""
        (
            losses,
            film,
            digital,
            film_paired,
            _,
            film_reconstructed,
            digital_reconstructed,
            digital_to_film,
            _,
        ) = self.step(batch, self.train_transform)

        # Extract only reconstructed film and digital
        film_reconstructed = film_reconstructed[: film.shape[0]]
        digital_reconstructed = digital_reconstructed[: digital.shape[0]]

        # Logging
        self.log_dict(
            self._add_prefix(losses, "train/"),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        train_metrics = self.train_metrics(digital_to_film, film)
        self.log_dict(train_metrics, on_step=True, on_epoch=True, prog_bar=True)

        if batch_idx % 10 == 0:
            self.log_images(
                film,
                film_reconstructed,
                key="train/film_reconstructed",
            )
            self.log_images(
                digital,
                digital_reconstructed,
                key="train/digital_reconstructed",
            )
            self.log_images(film_paired, digital_to_film, key="train/digital_to_film")

        return losses["loss"]

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Validation step for processing one batch of data."""
        (
            losses,
            film,
            digital,
            film_paired,
            _,
            film_reconstructed,
            digital_reconstructed,
            digital_to_film,
            _,
        ) = self.step(batch, self.val_transform)

        # Extract only reconstructed film and digital
        film_reconstructed = film_reconstructed[: film.shape[0]]
        digital_reconstructed = digital_reconstructed[: digital.shape[0]]

        # Log losses
        self.log_dict(
            self._add_prefix(losses, "val/"),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log metrics
        val_metrics = self.val_metrics(digital_to_film, film)
        self.log_dict(val_metrics, on_step=True, on_epoch=True, prog_bar=True)

        # Log images
        if batch_idx % 10 == 0:
            self.log_images(
                film,
                film_reconstructed,
                key="val/film_reconstructed",
            )
            self.log_images(
                digital,
                digital_reconstructed,
                key="val/digital_reconstructed",
            )
            self.log_images(film_paired, digital_to_film, key="val/digital_to_film")

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        """Test step for processing one batch of data."""
        # Unpack the batch
        film, digital, paired = batch

        # Remove artificial batch dimension
        film, digital, paired = film.squeeze(0), digital.squeeze(0), paired.squeeze(0)

        # Apply transforms
        film = self.test_transform(film, downsample=2)
        digital = self.test_transform(digital, downsample=2)
        paired = self.test_transform(paired, downsample=2)
        film_paired, digital_paired = paired

        # Forward pass through the model
        (
            film_reconstructed,
            digital_reconstructed,
            film_to_digital,
            digital_to_film,
            paired_encoder_representations,
        ) = self(film, digital, film_paired, digital_paired)

        # Compute the loss
        losses = self.loss(
            film,
            digital,
            film_paired,
            digital_paired,
            film_reconstructed,
            digital_reconstructed,
            digital_to_film,
            film_to_digital,
            paired_encoder_representations,
        )

        # Log metrics
        self.log_dict(
            self._add_prefix(losses, "test/"),
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        # Log all test images
        self.log_images(
            film,
            film_reconstructed,
            key="test/film_reconstructed",
        )

        self.log_images(
            digital,
            digital_reconstructed,
            key="test/digital_reconstructed",
        )

        self.log_images(film_paired, digital_to_film, key="test/digital_to_film")

    def predict(self, digital: PILImage, downsample=1) -> PILImage:
        """Predicts the output of the model for a given input."""
        assert isinstance(digital, PILImage), "Input must be a PIL image."

        # Prepare input
        input = self.infer_transform(digital, downsample=downsample).unsqueeze(0)

        # Forward pass
        output = self.net.predict(input)

        # Prepare output
        film_predicted = self.undo_transform(output)

        return film_predicted

    def log_images(
        self,
        reference: torch.Tensor,
        reconstructed: torch.Tensor,
        key: Optional[str] = None,
    ):
        if self.logger:
            if hasattr(self.logger.experiment, "log"):
                # Create figure
                batch_size = reference.size(0)
                fig, axs = plt.subplots(
                    nrows=2, ncols=batch_size, figsize=(4 * batch_size, 8)
                )
                if axs.ndim == 1:
                    axs = axs[:, None]
                fig.tight_layout(pad=1.0)
                for i in range(batch_size):
                    axs[0, i].imshow(np.array(self.undo_transform(reference[i])))
                    axs[1, i].imshow(np.array(self.undo_transform(reconstructed[i])))
                axs[0, 0].set_ylabel("Reference")
                axs[1, 0].set_ylabel("Reconstructed")

                # Log to W&B
                self.logger.experiment.log({key: wandb.Image(fig)})

                # Close figure
                plt.close()
