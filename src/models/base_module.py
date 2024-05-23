from typing import Optional, Dict

import torch
from lightning import LightningModule
from PIL.Image import Image as PILImage

import src.models.transforms as CT


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
        lr_monitor: str = "val/loss",
    ) -> None:
        """Initialises `BaseModel`."""
        super().__init__()

        # Save hyperparameters
        self.training_patch_size = training_patch_size
        self.augment = augment

    def undo_transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor and undoes the transformations necessary to convert it
        back to an image. This includes converting the tensor to a PIL image.

        NOTE: The T.ToPILImage() function only works on single images [C, H, W]
        and hence this function should be used on a single image at a time.
        """
        assert x.ndim == 3, f"Expected 3D tensor, got {x.ndim}D tensor"
        assert x.shape[0] == 3, f"Expected 3 channels, got {x.shape[0]} channels"

        return CT.FromModelInput()(x)

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
        return CT.TrainTransforms(self.training_patch_size, self.augment)(x)

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
        return CT.TrainTransforms(self.training_patch_size, self.augment)(x)

    def test_transform(self, x: torch.Tensor, downsample: int = 1) -> torch.Tensor:
        """
        Applies a series of transformations to the input tensor to prepare it
        for testing. This includes resizing the image to the nearest multiple
        of 4.

        Args:
            x (torch.Tensor): The input tensor to transform ([..., H, W])

        Returns:
            torch.Tensor: The transformed tensor
        """
        height = self.get_valid_dim(x.shape[-2], downsample=downsample)
        width = self.get_valid_dim(x.shape[-1], downsample=downsample)
        test_transforms = CT.TestTransforms(dim=(height, width))
        return test_transforms(x)

    def infer_transform(self, img: PILImage, downsample: int = 1) -> None:
        height = self.get_valid_dim(img.size[1], downsample=downsample)
        width = self.get_valid_dim(img.size[0], downsample=downsample)
        infer_transforms = CT.TestTransforms(dim=(height, width))
        return infer_transforms(img)

    def get_valid_dim(self, dim: int, downsample: int = 1) -> int:
        """
        Returns the nearest multiple of 8 that is less than or equal to the
        input dimension. This is required because of the network architecture.

        NOTE: This should better be defined in the `src.models.net` modules.

        Args:
            dim (int): The input dimension

        Returns:
            int: The nearest multiple of 4 that is less than or equal to the input
        """
        adjusted_dim = dim // downsample
        valid_dim = (adjusted_dim // 8) * 8
        return valid_dim

    def _add_prefix(self, dict: Dict, prefix: str) -> Dict:
        return {prefix + key: value for key, value in dict.items()}

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
