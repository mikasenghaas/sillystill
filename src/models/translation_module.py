from typing import Tuple, Optional

import numpy as np
import wandb
import torch
import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from ..eval import PieAPP

from PIL.Image import Image as PILImage
from matplotlib import pyplot as plt

from src.models.transforms import pil_to_plot
from .base_module import BaseModule


class TranslationModule(BaseModule):
    """
    LightningModule for image-to-image translation tasks, such as transforming
    digital images to appear as if shot on CineStill800T film.
    """

    def __init__(
        self,
        net: nn.Module,
        loss: nn.Module,
        augment: float = 0.0,
        training_patch_size: int = 128,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: torch.optim.lr_scheduler._LRScheduler = torch.optim.lr_scheduler.ReduceLROnPlateau,
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
        super().__init__(
            augment=augment,
            training_patch_size=training_patch_size,
            optimizer=optimizer,
            scheduler=scheduler,
            lr_monitor=lr_monitor,
        )

        # Store hyperparameters
        self.save_hyperparameters(logger=False, ignore=["loss", "net"])
        self.net = net
        self.loss = loss

        # Initialise metrics
        # metrics = MetricCollection(
        #     {
        #         "ssim": SSIM(),
        #         "psnr": PSNR(),
        #         # "lpips": LPIPS(net_type="squeeze", normalize=True),
        #         # "pieapp": PieAPP(),
        #     }
        # )
        # self.train_metrics = metrics.clone(prefix="train/")
        # self.val_metrics = metrics.clone(prefix="val/")
        # self.test_metrics = metrics.clone(prefix="test/")
        # self.train_baseline_metrics = metrics.clone(prefix="train/baseline/")
        # self.val_baseline_metrics = metrics.clone(prefix="val/baseline/")
        # self.test_baseline_metrics = metrics.clone(prefix="test/baseline/")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model. The input is expected to be of shape
        [B, C, H, W] and normalised with the common transforms.

        Args:
            x (torch.Tensor): Input tensor representing a batch of images.

        Returns
            torch.Tensor: The output tensor representing the transformed images.
        """
        return self.net(x)

    def step(
        self, batch: torch.Tensor, transform: Optional[nn.Module]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process one batch of data. A batch comes is expected to be of shape
        [2, B, C, H, W] where the first element is the film images and the
        second element is the digital images.

        Args:
            batch (torch.Tensor): A batch of data, shape [2, B, C, H, W].
            transform (Optional[nn.Module]): A transform to apply to the batch.

        Returns:
            torch.Tensor: The loss value for the batch.
            torch.Tensor: The film images.
            torch.Tensor: The digital images.
            torch.Tensor: The predicted film images.
        """
        # Forward pass
        film, digital = transform(batch)
        film_predicted = self.forward(digital)
        losses = self.loss(film_predicted, film)

        return losses, film, digital, film_predicted

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Training step for processing one batch of data."""
        # Forward pass
        losses, film, digital, film_predicted = self.step(batch, self.train_transform)

        # Log training loss, metrics and images
        self.log_dict(
            self._add_prefix(losses, "train/"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # train_metrics = self.train_metrics(
        #     film_predicted.clamp(0 + 1e-5, 1 - 1e-5), film.clamp(0 + 1e-5, 1 - 1e-5)
        # )
        # baseline_metrics = self.train_baseline_metrics(digital, film)
        # self.log_dict(train_metrics, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_dict(baseline_metrics, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.log_images(film, digital, film_predicted, key="train/images")

        return losses["loss"]

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Validation step for processing one batch of data."""
        # Forward pass
        losses, film, digital, film_predicted = self.step(batch, self.train_transform)

        # Log validation loss and images
        self.log_dict(
            self._add_prefix(losses, "val/"),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        # val_metrics = self.val_metrics(
        #     film_predicted.clamp(0 + 1e-5, 1 - 1e-5), film.clamp(0 + 1e-5, 1 - 1e-5)
        # )
        # baseline_metrics = self.train_baseline_metrics(digital, film)
        # self.log_dict(val_metrics, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_dict(baseline_metrics, on_step=False, on_epoch=True, prog_bar=True)
        if batch_idx == 0:
            self.log_images(film, digital, film_predicted, key="val/images")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        """Test step for processing one batch of data."""
        # Forward pass
        film, digital = self.test_transform(batch, downsample=2)
        film_predicted = self.forward(digital)

        test_metrics = self.test_metrics(
            film_predicted.clamp(0 + 1e-5, 1 - 1e-5), film.clamp(0 + 1e-5, 1 - 1e-5)
        )
        # baseline_metrics = self.test_baseline_metrics(digital, film)
        # self.log_dict(test_metrics, on_step=False, on_epoch=True, prog_bar=True)
        # self.log_dict(baseline_metrics, on_step=False, on_epoch=True, prog_bar=True)
        self.log_images(film, digital, film_predicted, key="test/images")

    def predict(self, digital: PILImage, downsample=1) -> PILImage:
        """Predicts the output of the model for a given input."""
        assert isinstance(digital, PILImage), "Input must be a PIL image."

        # Prepare input
        input = self.infer_transform(digital, downsample=downsample).unsqueeze(0)

        # Forward pass
        output = self.forward(input)

        # Prepare output
        film_predicted = self.undo_transform(output.squeeze(0))

        return film_predicted

    def log_images(
        self,
        film: torch.Tensor,
        digital: torch.Tensor,
        film_predicted: torch.Tensor,
        key: Optional[str] = None,
    ):
        if self.logger:
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
                    axs[0, i].imshow(pil_to_plot(self.undo_transform(digital[i])))
                    axs[1, i].imshow(pil_to_plot(self.undo_transform(film[i])))
                    axs[2, i].imshow(
                        pil_to_plot(self.undo_transform(film_predicted[i]))
                    )
                axs[0, 0].set_ylabel("Digital")
                axs[1, 0].set_ylabel("Film (Ground Truth)")
                axs[2, 0].set_ylabel("Film (Predicted)")

                # Log to W&B
                self.logger.experiment.log({key: wandb.Image(fig)})

                # Close figure
                plt.close()
