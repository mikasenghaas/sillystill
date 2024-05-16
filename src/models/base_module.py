from typing import Tuple, Optional

import wandb
import torch
import numpy as np
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MetricCollection
import torchvision.transforms.v2 as T
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)

from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.qnr import QualityWithNoReference as QNR

from matplotlib import pyplot as plt
from ..utils.utils import undo_transforms
from .transforms import (
    get_transform_from_model_input,
    get_transform_to_model_input,
    get_transform_augment,
)


class BaseModule(LightningModule):
    """
    Base module for image-to-image translation tasks, such as transforming
    digital images to appear as if shot on CineStill800T film.
    """

    def __init__(
        self,
        augment: float,
        training_patch_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        lr_monitor: str,  # Check why we can't have val/loss
    ) -> None:
        """Initialises `BaseModel`."""
        super().__init__()

        self.training_patch_size = training_patch_size
        self.transform_to_model_input = get_transform_to_model_input()
        self.transform_from_model_input = get_transform_from_model_input()
        self.augment = get_transform_augment(augment)

    def undo_transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.transform_from_model_input(x)

    def transform_train(self, x: torch.Tensor) -> torch.Tensor:
        transform_train = T.Compose(
            [
                self.augment,
                self.transform_to_model_input,
                T.RandomResizedCrop(self.training_patch_size),
            ]
        )

        return transform_train(x)

    def transform_test(self, x: torch.Tensor) -> torch.Tensor:
        transform_test = T.Compose(
            [
                self.transform_to_model_input,
                T.Resize(
                    (self.get_valid_dim(x.shape[-2]), self.get_valid_dim(x.shape[-1]))
                ),
            ]
        )
        return transform_test(x)

    def get_valid_dim(self, dim: int) -> int:
        return dim // 4 * 4

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
