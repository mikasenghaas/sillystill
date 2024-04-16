from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms

import torch
from typing import Optional, Tuple

from ...utils.load import load_metadata, load_image_pair


class ImagePairDataset(Dataset):
    """
    A PyTorch Dataset class for loading processed image pairs of digital and
    film images.
    """

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """
        Initialises an `ImagePairDataset` instance. This dataset is used to load
        image pairs from the processed data directory. The dataset assumes that
        the filenames in both directories match for corresponding image pairs.

        Args:
            data_dir (str): Data directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Save hyperparameters
        self.data_dir = data_dir
        self.transform = transform

        # Load metadata
        self.meta = load_metadata()

        # Load processed image pairs from the data directory
        self.images = []
        for idx in self.meta.keys():
            if idx not in [9, 11, 40]:  # Skip these images
                film, digital, _ = load_image_pair(
                    idx, processing_state="processed", as_array=True
                )
                self.images.append((film, digital))

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        return self.transform(self.images[idx])
