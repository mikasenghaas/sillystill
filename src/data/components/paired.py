from typing import Optional, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from ...utils.load import load_image_pair, load_metadata


class PairedDataset(Dataset):
    """A PyTorch Dataset class for loading processed image pairs of digital and film images."""

    def __init__(self, data_dir: str, transforms: Optional[transforms.Compose] = None):
        """Initialises an `ImagePairDataset` instance. This dataset is used to load image pairs
        from the processed data directory. The dataset assumes that the filenames in both
        directories match for corresponding image pairs.

        Args:
            data_dir (str): Data directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Save hyperparameters
        self.data_dir = data_dir
        self.transforms = transforms

        # Load metadata
        self.meta = load_metadata()
        self.keys = list(set(self.meta.keys()) - {9, 11, 40})
        self.idx_to_key = {idx: key for idx, key in enumerate(self.keys)}

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.keys)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        key = self.idx_to_key[idx]
        film, digital, _ = load_image_pair(key, processing_state="processed", as_array=True)

        if self.transforms:
            return self.transforms((film, digital))
        return (film, digital)
