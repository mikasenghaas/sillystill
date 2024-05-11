import glob
from typing import Tuple

import torch
from torch.utils.data import Dataset

from src.data.components.paired import PairedDataset
from src.data.components.unpaired import UnpairedDataset


class AutoTranslateDataset(Dataset):
    """Dataset class for loading unpaired images, either digital or film."""

    def __init__(
        self,
        digital_dataset: UnpairedDataset,
        film_dataset: UnpairedDataset,
        paired_dataset: PairedDataset,
        num_paired_per_batch: int = 4,
        num_unpaired_per_batch: int = 6,
    ):
        """Initialises an `ImagePairDataset` instance. This dataset is used to load image pairs
        from the processed data directory. The dataset assumes that the filenames in both
        directories match for corresponding image pairs.

        Args:
            digital_dataset (Dataset): Unpaired Digital dataset
            film_dataset (Dataset): Unpaired Film dataset
            paired_dataset (Dataset): Paired dataset
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

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""

        # Get 3 random unpaired digital images
        indices = torch.randint(0, len(self.digital_dataset), (self.num_unpaired_per_batch,))
        digital_images = []
        for i in range(self.num_unpaired_per_batch):
            digital_images.append(self.digital_dataset[indices[i]])
        digital_images = torch.stack(digital_images)

        # Get no_unpaired_per_batch unpaired film images
        indices = torch.randint(0, len(self.film_dataset), (self.num_unpaired_per_batch,))
        film_images = []
        for i in range(self.num_unpaired_per_batch):
            film_images.append(self.film_dataset[indices[i]])
        film_images = torch.stack(film_images)

        # Get no_paired_per_batch paired images
        indices = torch.randint(0, len(self.paired_dataset), (self.num_paired_per_batch,))
        digital_paired = []
        film_paired = []
        for i in range(self.num_paired_per_batch):
            digital, film = self.paired_dataset[indices[i]]
            digital_paired.append(digital)
            film_paired.append(film)

        paired_images = (torch.stack(digital_paired), torch.stack(film_paired))

        return digital_images, film_images, paired_images
