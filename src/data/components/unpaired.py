import glob

import torch
from typing import List
from torch.utils.data import Dataset
from PIL.Image import Image as PILImage

from src.utils.load import _load_image_from_path
import src.models.transforms as CT


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

        # Define transforms
        self.transform = CT.ToBatchedTensor()

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> PILImage:
        """Returns a sample from the dataset at the given index."""
        image_path = self.image_paths[idx]

        image = _load_image_from_path(image_path)

        return image

    def collate(self, batch: List[PILImage]) -> torch.Tensor:
        """
        Collates a batch of images into a single tensor. This is required
        because the `DataLoader` collates the batch into a list of tensors.

        Args:
            batch (List[Tuple[PILImage, PILImage]]): The batch of images

        Returns:
            torch.Tensor: The collated batch of images
        """
        # Process batch
        for i, digital in enumerate(batch):
            batch[i] = self.transform(digital)

        # Stack the batch
        batch = torch.stack(batch, dim=0)

        return batch
