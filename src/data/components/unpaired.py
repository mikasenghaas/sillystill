import glob
from typing import Dict, Optional

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from src.utils.load import _load_image_from_path


class UnpairedDataset(Dataset):
    """Dataset class for loading images from a single directory."""

    def __init__(
        self,
        image_dir: str,
        patch_size: int = 128,
        augment: Optional[Dict] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Initialises a `UnpairedDataset` instance. This dataset is used to load images from
        a single image directory and apply data augmentation if required. The dataset can be
        truncated to a maximum number of samples, if required.

        Args:
            data_dir (str): Data directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Save hyperparameters
        self.image_dir = image_dir
        self.max_samples = max_samples
        self.patch_size = patch_size

        # Load image paths
        self.image_paths = sorted(glob.glob(f"{image_dir}/*"))

        # Set base transforms (defaults)
        all_transforms = [
            transforms.ToImage(),
            transforms.ToDtype(torch.uint8, scale=True),
            transforms.RandomResizedCrop(size=(patch_size, patch_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

        # Add data augmentation
        if augment is None:
            augment = {
                "flip": False,
                "rotate": False,
                "blur": False,
                "brightness": False,
            }
        for method, active in augment.items():
            if active:
                if method == "flip":
                    all_transforms.insert(3, transforms.RandomHorizontalFlip(p=0.2))
                    all_transforms.insert(3, transforms.RandomVerticalFlip(p=0.2))
                elif method == "rotate":
                    all_transforms.insert(
                        3,
                        transforms.RandomApply(
                            [transforms.RandomRotation(degrees=(0, 360))],
                            0.2,
                        ),
                    )
                elif method == "blur":
                    all_transforms.insert(
                        3,
                        transforms.RandomApply(
                            [
                                transforms.GaussianBlur(
                                    kernel_size=(5, 9), sigma=(0.1, 5.0)
                                )
                            ],
                            0.2,
                        ),
                    )
                elif method == "brightness":
                    all_transforms.insert(
                        3,
                        transforms.RandomApply(
                            [transforms.ColorJitter(brightness=(0.6, 1))],
                            0.2,
                        ),
                    )

        # Add augmentation transform
        self.transforms = transforms.Compose(all_transforms)

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a sample from the dataset at the given index."""
        image_path = self.image_paths[idx]

        image = _load_image_from_path(image_path, as_array=True)

        if self.transforms is not None:
            return self.transforms(image)
        return image
