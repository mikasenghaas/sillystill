from typing import Optional, Tuple

import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset


class UnpairedDataset(Dataset):
    """"""

    def __init__(self, data_dir: str, transform: Optional[transforms.Compose] = None):
        """Initialises an `ImagePairDataset` instance. This dataset is used to load image pairs
        from the processed data directory. The dataset assumes that the filenames in both
        directories match for corresponding image pairs.

        Args:
            data_dir (str): Data directory
            transform (callable, optional): Optional transform to be applied on a sample
        """
        # Save hyperparameters
        self.data_dir = data_dir
        if transform is None:
            # Default data transforms (TODO: Make these configurable)
            self.transform = transforms.Compose(
                [
                    transforms.ToImage(),
                    transforms.ToDtype(torch.float32, scale=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    transforms.RandomResizedCrop(size=(100, 100), antialias=True),
                ]
            )
        else:
            self.transform = transform

        # TODO: Load all images paths (glob.glob(data_dir/*))
        self.image_paths = []

    def __len__(self) -> int:
        """Returns the length of the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns a sample from the dataset at the given index."""
        path = self.image_paths[idx]

        # Load image from path (src.utils.)
        digital = ...

        if self.transform:
            return self.transform(digital)
        return digital
