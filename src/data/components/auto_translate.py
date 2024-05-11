import glob
from typing import Optional, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset

from src.data.components.paired import PairedDataset
from src.data.components.unpaired import UnpairedDataset
from src.utils.load import _load_image_from_path


class AutoTranslateDataset(Dataset):
    """Dataset class for loading unpaired images, either digital or film."""

    def __init__(
        self,
        digital_dataset: UnpairedDataset,
        film_dataset: UnpairedDataset,
        paired_dataset: PairedDataset,
        num_paired_per_batch: int = 4,
        num_unpaired_per_batch: int = 12,
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
        no_paired_batches = len(self.paired_dataset) // self.num_paired_per_batch
        return max(no_digital_batches, no_film_batches, no_paired_batches)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""

        # Get no_unpaired_per_batch unpaired digital images
        digital_images = []
        for i in range(self.num_unpaired_per_batch):
            digital_images.append(self.digital_dataset[idx + i])
        digital_images = torch.stack(digital_images)

        # Get no_unpaired_per_batch unpaired film images
        film_images = []
        for i in range(self.num_unpaired_per_batch):
            film_images.append(self.film_dataset[idx + i])
        film_images = torch.stack(film_images)

        # Get no_paired_per_batch paired images
        paired_images = []
        for i in range(self.num_paired_per_batch):
            paired_images.append(self.paired_dataset[idx + i])
        paired_images = torch.stack(paired_images)

        return digital_images, film_images, paired_images
