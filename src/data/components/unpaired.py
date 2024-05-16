import glob
from typing import Dict, Optional

import torch
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset

from src.utils.load import _load_image_from_path


class UnpairedDataset(Dataset):
    """Dataset class for loading images from a single directory."""

    def __init__(
        self,
        image_dir: str,
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

        # Load image paths
        self.image_paths = sorted(glob.glob(f"{image_dir}/*"))

        # Set base transforms (defaults)
        self.transform = T.Compose([T.ToImage(), T.Resize((2433, 3637))])

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a sample from the dataset at the given index."""
        image_path = self.image_paths[idx]

        image = _load_image_from_path(image_path)

        return self.transform(image)
