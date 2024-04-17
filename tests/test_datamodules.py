import numpy as np
import pytest
import torch
import torchvision.transforms.v2 as transforms

from src.data.components.image_pair_dataset import ImagePairDataset
from src.data.datamodule import DataModule


def test_image_pair_dataset_no_transforms() -> None:
    """Tests `ImagePairDataset` without transforms."""
    dataset = ImagePairDataset(data_dir="/data", transform=None)
    assert dataset.meta and dataset.keys
    assert len(dataset) == 38  # 41 - 3 (skipped images)

    sample = dataset[0]
    assert type(sample) == tuple
    assert all(isinstance(x, np.ndarray) for x in sample)


def test_image_pair_dataset() -> None:
    """Tests `ImagePairDataset` with transforms."""
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImagePairDataset(data_dir="/data", transform=transform)
    assert dataset.meta and dataset.keys
    assert len(dataset) == 38  # 41 - 3 (skipped images)

    sample = dataset[0]
    assert type(sample) == tuple
    assert all(isinstance(x, torch.Tensor) for x in sample)


@pytest.mark.parametrize("batch_size", [4, 8])
def test_image_pair_datamodule(batch_size: int) -> None:
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomResizedCrop(size=(100, 100), antialias=True),
        ]
    )
    dataset = ImagePairDataset(data_dir="/data", transform=transform)
    dm = DataModule(dataset=dataset, batch_size=batch_size)

    assert not dm.data_train and not dm.data_val and not dm.data_test

    dm.setup()
    assert dm.data_train and dm.data_val and dm.data_test
    assert dm.train_dataloader() and dm.val_dataloader() and dm.test_dataloader()

    batch = next(iter(dm.train_dataloader()))
    x, y = batch
    assert len(x) == batch_size
    assert len(y) == batch_size
    assert x.dtype == torch.float32
    assert y.dtype == torch.float32
