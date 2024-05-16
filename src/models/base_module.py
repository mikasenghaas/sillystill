from typing import Optional

import torch
from lightning import LightningModule
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F
from PIL.Image import Image as PILImage

from torchmetrics.image.fid import FrechetInceptionDistance as FID
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
from torchmetrics.image.qnr import QualityWithNoReference as QNR

from .transforms import Unnormalize


class BaseModule(LightningModule):
    """
    Base module for image-to-image translation tasks, such as transforming
    digital images to appear as if shot on CineStill800T film.
    """

    def __init__(
        self,
        augment: float,
        training_patch_size: int,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lr_monitor: str = "train/loss",  # Check why we can't have val/loss
    ) -> None:
        """Initialises `BaseModel`."""
        super().__init__()

        # Save hyperparameters
        self.training_patch_size = training_patch_size

        # Initialise transforms
        self.transform = T.Compose(
            [
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.augment = T.Compose(
            [
                T.RandomHorizontalFlip(p=augment),
                T.RandomVerticalFlip(p=augment),
                T.RandomApply(
                    [
                        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                        T.ColorJitter(brightness=(0.6, 1)),
                    ],
                    augment,
                ),
            ]
        )

    def undo_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unnormalizes the input tensor to convert it back to an image.

        Args:
            x (torch.Tensor): The input tensor to unnormalize

        Returns:
            torch.Tensor: The unnormalized tensor
        """
        undo_transform = Unnormalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        return undo_transform(x)

    def train_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor and applies a series of transformations to it to prepare
        it for training. Crucially, it crops the image to a patch of size
        `training_patch_size`. If augmentation is enabled, it also applies
        random transformations to the image.

        Args:
            x (torch.Tensor): The input tensor to transform ([..., H, W])

        Returns:
            torch.Tensor: The transformed tensor
        """
        transform_train = T.Compose(
            [
                self.augment,
                self.transform,
                T.RandomResizedCrop(self.training_patch_size),
            ]
        )

        return transform_train(x)

    def val_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a series of transformations to the input tensor to prepare it
        for validation. This includes resizing the image to the nearest multiple
        of 4.

        Args:
            x (torch.Tensor): The input tensor to transform ([..., H, W])

        Returns:
            torch.Tensor: The transformed tensor
        """
        transform_val = T.Compose(
            [
                self.transform,
                T.RandomResizedCrop(self.training_patch_size),
            ]
        )
        return transform_val(x)

    def test_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies a series of transformations to the input tensor to prepare it
        for testing. This includes resizing the image to the nearest multiple
        of 4.

        Args:
            x (torch.Tensor): The input tensor to transform ([..., H, W])

        Returns:
            torch.Tensor: The transformed tensor
        """
        height = self.get_valid_dim(x.shape[-2])
        width = self.get_valid_dim(x.shape[-1])
        transform_test = T.Compose([self.transform, T.Resize((height, width))])
        return transform_test(x)

    def to_image(self, x: torch.Tensor) -> PILImage:
        return F.to_pil_image(x)

    def get_valid_dim(self, dim: int, downsample: int = 1) -> int:
        """
        Returns the nearest multiple of 4 that is less than or equal to the
        input dimension. This is required because of the network architecture.

        NOTE: This should better be defined in the `src.models.net` modules.

        Args:
            dim (int): The input dimension

        Returns:
            int: The nearest multiple of 4 that is less than or equal to the input
        """
        return int((dim / downsample) // 4 * 4)

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
