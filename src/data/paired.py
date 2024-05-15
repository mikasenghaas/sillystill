import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from .components.paired import PairedDataset


class PairedDigitalFilmDataModule(LightningDataModule):
    """Lightning datamodule for the paired digital-film image pair dataset."""

    def __init__(
        self,
        data_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        batch_size: int = 4,  # Only for train split (else, 1)
        patch_size: int = 128,
        max_samples: Optional[int] = None,
        augment: Optional[List[Dict]] = None,
        num_workers: int = 1,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialise a `DigitalFilmDataModule` which is a Lightning wrapper around
        the paired digital-film image pair dataset. The dataset is split into
        train, validation and test sets.

        Args:
            data_split (Tuple[float, float, float]): The train, validation and test split.
            batch_size (int): The batch size. Defaults to `4`.
            patch_size (int): The patch size. Defaults to `128`.
            max_samples (int, optional): The maximum number of samples to load. Defaults to `None`.
            augment (Optional[List[Dict]]): The data augmentation to apply. Defaults to `None`.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """
        Instantiates the dataset and prepares the data. This method is called
        only once and before `setup()`.

        If not, it raises an error.
        """
        # Default image directories
        self.film_paired_dir = os.path.join("data", "paired", "processed", "film")
        self.digital_paired_dir = os.path.join("data", "paired", "processed", "digital")

        # Assert data exists
        assert os.path.exists(self.film_paired_dir) and os.path.exists(
            self.digital_paired_dir
        ), "Data not found. Try running `git lfs pull`."

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        Args     stage (Optional[str]): The stage being set up. Either `"fit"`,     `"validate"`,
        `"test"`, or `"predict"`.ge
        """
        # Load paired dataset
        self.dataset = PairedDataset(
            image_dirs=(self.film_paired_dir, self.digital_paired_dir),
            patch_size=self.hparams.patch_size,
            max_samples=self.hparams.max_samples,
            augment=self.hparams.augment,
        )

        # Load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=self.dataset,
                lengths=self.hparams.data_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader"""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader"""
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader"""
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
