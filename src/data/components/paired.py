from typing import Dict, List, Optional, Tuple

import glob
import hydra
import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from ...utils.load import load_image_pair, load_metadata


class PairedDataset(Dataset):
    """A PyTorch dataset class for loading processed image pairs of digital and film images."""

    def __init__(
        self,
        data_dir1: str,
        data_dir2: str,
        patch_size: int,
        max_samples: Optional[int] = None,
        augment: Optional[List[Dict]] = None,
    ):
        """Initialises a `PairedDataset` instance. This dataset is used to load
        image pairs from the processed data directory. The dataset assumes that
        the filenames in both directories match for corresponding image pairs.

        Args:
            data_dir1 (str): Data directory for the first set of images.
            data_dir2 (str): Data directory for the second set of images.
            patch_size (int): The size of the patches to extract.
            max_samples (int, optional): The maximum number of samples to load. Defaults to `None`.
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Save hyperparameters
        self.data_dir1 = data_dir1
        self.data_dir2 = data_dir2
        self.max_samples = max_samples
        self.patch_size = patch_size

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

        # Load metadata
        self.image_paths1 = glob.glob(f"{data_dir1}/*")
        self.image_paths2 = glob.glob(f"{data_dir2}/*")

        assert len(self.image_paths1) == len(
            self.image_paths2
        ), "Mismatch in number of images"
        assert set([path.split("/")[-1] for path in self.image_paths1]) == set(
            [path.split("/")[-1] for path in self.image_paths2]
        ), "Mismatch in image filenames"

        # self.keys = list(set(self.meta.keys()) - {9, 11, 40})
        self.idx_to_key = {idx: key for idx, key in enumerate(self.keys)}

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        key = self.idx_to_key[idx]
        film, digital, _ = load_image_pair(
            key, processing_state="processed", as_array=True
        )

        if self.transforms:
            return self.transforms((film, digital))
        return (film, digital)
