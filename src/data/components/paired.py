import torch
from typing import Dict, List, Tuple

import glob
import torchvision.transforms.v2 as T
from torch.utils.data import Dataset
from PIL import Image
from PIL.Image import Image as PILImage

from ...utils.load import _load_image_from_path


class PairedDataset(Dataset):
    """A PyTorch dataset class for loading image pairs from two directories."""

    def __init__(
        self,
        image_dirs: Tuple[str, str],
    ):
        """
        Initialises a `PairedDataset` instance. This dataset is used to load
        image pairs from two data directories. The dataset assumes that the
        filenames in both directories match for corresponding image pairs
        and are in the same format. Data augmentation can be applied to the
        images when loading. The dataset can be truncated to a maximum number
        of samples, if required.

        Args:
            image_dirs (Tuple[str]): Data directory for the first set of images.
            augment (bool, optional): Optional transform to be applied on a sample
            augment_prob (float, optional): Probability of applying the augmentation
        """
        # Save hyperparameters
        self.image_dirs = image_dirs
        self.transform = T.Compose([T.ToImage(), T.Resize((2433, 3637))])

        # Load image paths
        self.image_paths1 = sorted(glob.glob(f"{image_dirs[0]}/*"))
        self.image_paths2 = sorted(glob.glob(f"{image_dirs[1]}/*"))

        # Assertions
        assert len(self.image_paths1) == len(
            self.image_paths2
        ), "Mismatch in number of images"
        assert set([path.split("/")[-1] for path in self.image_paths1]) == set(
            [path.split("/")[-1] for path in self.image_paths2]
        ), "Mismatch in image filenames"

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.image_paths1)

    def __getitem__(self, idx: int) -> Tuple[PILImage, PILImage]:
        """Returns a sample from the dataset at the given index."""
        # Get image paths
        image_path1 = self.image_paths1[idx]
        image_path2 = self.image_paths2[idx]

        # Load image pair
        image1 = _load_image_from_path(image_path1)
        image2 = _load_image_from_path(image_path2)

        return self.transform((image1, image2))
