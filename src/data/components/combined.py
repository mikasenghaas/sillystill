from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.data.components.paired import PairedDataset
from src.data.components.unpaired import UnpairedDataset


class CombinedDataset(Dataset):
    """Dataset class for loading a batch of mixed paired and unpiared images"""

    def __init__(
        self,
        digital_dataset: UnpairedDataset,
        film_dataset: UnpairedDataset,
        paired_dataset: PairedDataset,
        num_paired_per_batch: int = 4,
        num_unpaired_per_batch: int = 6,
    ):
        """
        Initialises a `CombinedDataset` instance. This dataset is used to load a batch of mixed paired and unpaired images according to the given hyperparameters.

        Args:
            digital_dataset (Dataset): Unpaired Digital dataset
            film_dataset (Dataset): Unpaired Film dataset
            paired_dataset (Dataset): Paired dataset
            num_paired_per_batch (int): Number of paired images per batch
            num_unpaired_per_batch (int): Number of unpaired images per batch
        """
        # Save hyperparameters
        self.digital_dataset = digital_dataset
        self.film_dataset = film_dataset
        self.paired_dataset = paired_dataset
        self.num_paired_per_batch = num_paired_per_batch
        self.num_unpaired_per_batch = num_unpaired_per_batch

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        no_digital_batches = len(self.digital_dataset) // self.num_unpaired_per_batch
        no_film_batches = len(self.film_dataset) // self.num_unpaired_per_batch
        num_paired_batches = len(self.paired_dataset) // self.num_paired_per_batch
        return max(no_digital_batches, no_film_batches, num_paired_batches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        # Tensor: [B1, C, H, W]
        n, k = len(self.digital_dataset), self.num_unpaired_per_batch
        indices = torch.randint(0, n, (k,))
        digital_images = []
        for i in range(self.num_unpaired_per_batch):
            digital_images.append(self.digital_dataset[indices[i]])
        digital_images = self.digital_dataset.collate(digital_images)

        # Tensor: [B1, C, H, W]
        n, k = len(self.film_dataset), self.num_unpaired_per_batch
        indices = torch.randint(0, n, (k,))
        film_images = []
        for i in range(self.num_unpaired_per_batch):
            film_images.append(self.film_dataset[indices[i]])
        film_images = self.film_dataset.collate(film_images)

        # Tensor: [2, B2, C, H, W]
        n, k = len(self.paired_dataset), self.num_paired_per_batch
        indices = torch.randint(0, n, (k,))
        film_digital_paired = []
        for i in range(self.num_paired_per_batch):
            film, digital = self.paired_dataset[indices[i]]
            film_digital_paired.append((film, digital))
        paired_images = self.paired_dataset.collate(film_digital_paired)

        return film_images, digital_images, paired_images
