import torch
from typing import List, Tuple

import glob
from torch.utils.data import Dataset
from PIL.Image import Image as PILImage
import torchvision.transforms.v2 as T

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
        """
        Returns a sample from the dataset at the given index. Here,
        this is a pair of images from the two directories in the
        PIL format.

        Args:
            idx (int): Index of the sample to be retrieved

        Returns:
            Tuple[PILImage, PILImage]: Image pair
        """
        # Get image paths
        image_path1 = self.image_paths1[idx]
        image_path2 = self.image_paths2[idx]

        # Load image pair
        image1 = _load_image_from_path(image_path1)
        image2 = _load_image_from_path(image_path2)

        return image1, image2

    def collate(self, batch: List[Tuple[PILImage]]) -> torch.Tensor:
        """
        Collates a batch of images into a single tensor. This is required
        because the `DataLoader` collates the batch into a list of tensors.

        Args:
            batch (List[Tuple[PILImage, PILImage]]): The batch of images

        Returns:
            torch.Tensor: The collated batch of images
        """
        # Transfrom for batching (convert to equal sized tensors)
        transform = T.Compose([T.ToImage(), T.Resize((2433, 3637))])

        # Process batch
        for i, (film, digital) in enumerate(batch):
            film = transform(film)
            digital = transform(digital)
            batch[i] = torch.cat([film.unsqueeze(0), digital.unsqueeze(0)], dim=0)

        # Stack the batch
        batch = torch.stack(batch, dim=0)

        # Permute dimensions
        batch = batch.permute(1, 0, 2, 3, 4)

        return batch
