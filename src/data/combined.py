import os
from typing import Any, Dict, Optional, Tuple, List

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components import CombinedDataset, PairedDataset, UnpairedDataset
from src.data.components.subset import CustomSubset


class CombinedDigitalFilmDataModule(LightningDataModule):
    """Lightning data module for the digital-film image pair dataset."""

    def __init__(
        self,
        paired_image_dirs: Tuple[str, str],
        unpaired_image_dirs: Tuple[str, str],
        data_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_paired_per_batch: int = 4,
        num_unpaired_per_batch: int = 6,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        **kwargs,
    ) -> None:
        """Initialise a `CombinedDigitalFilmDataModule`."""
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Asserts that the raw and processed data directories exist.

        If not, it raises an error.
        """
        # Initialise paths
        self.film_paired_dir, self.digital_paired_dir = self.hparams.paired_image_dirs
        self.film_unpaired_dir, self.digital_unpaired_dir = self.hparams.upaired_image_dirs

        # Asserts data exists
        assert (
            os.path.exists(self.film_paired_dir)
            and os.path.exists(self.digital_paired_dir)
            and os.path.exists(self.film_unpaired_dir)
            and os.path.exists(self.digital_unpaired_dir)
        ), "Data not found. Try running `git lfs pull`."

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        Args     stage (Optional[str]): The stage being set up. Either `"fit"`,     `"validate"`,
        `"test"`, or `"predict"`.ge
        """
        if not self.data_train and not self.data_val and not self.data_test:
            # Instantiate paired and unpaired datasets
            paired_dataset = PairedDataset(
                image_dirs=(self.film_paired_dir, self.digital_paired_dir)
            )
            unpaired_digital_dataset = UnpairedDataset(
                image_dir=self.digital_unpaired_dir,
            )
            unpaired_film_dataset = UnpairedDataset(
                image_dir=self.film_unpaired_dir,
            )

            # Split each of these into train, val, and test
            self.paired_train, self.paired_val, self.paired_test = random_split(
                dataset=paired_dataset,
                lengths=self.hparams.data_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.digital_train, self.digital_val, self.digital_test = random_split(
                dataset=unpaired_digital_dataset,
                lengths=self.hparams.data_split,
                generator=torch.Generator().manual_seed(42),
            )
            self.film_train, self.film_val, self.film_test = random_split(
                dataset=unpaired_film_dataset,
                lengths=self.hparams.data_split,
                generator=torch.Generator().manual_seed(42),
            )

            # Create custom subsets (to keep `collate method`)
            self.paired_train = CustomSubset(paired_dataset, self.paired_train.indices)
            self.paired_val = CustomSubset(paired_dataset, self.paired_val.indices)
            self.paired_test = CustomSubset(paired_dataset, self.paired_test.indices)

            self.digital_train = CustomSubset(
                unpaired_digital_dataset, self.digital_train.indices
            )
            self.digital_val = CustomSubset(
                unpaired_digital_dataset, self.digital_val.indices
            )
            self.digital_test = CustomSubset(
                unpaired_digital_dataset, self.digital_test.indices
            )

            self.film_train = CustomSubset(
                unpaired_film_dataset, self.film_train.indices
            )
            self.film_val = CustomSubset(unpaired_film_dataset, self.film_val.indices)
            self.film_test = CustomSubset(unpaired_film_dataset, self.film_test.indices)

            # Instantiate the AutoTranslateDatasets
            self.data_train = CombinedDataset(
                digital_dataset=self.digital_train,
                film_dataset=self.film_train,
                paired_dataset=self.paired_train,
                num_paired_per_batch=self.hparams.num_paired_per_batch,
                num_unpaired_per_batch=self.hparams.num_unpaired_per_batch,
            )
            self.data_val = CombinedDataset(
                digital_dataset=self.digital_val,
                film_dataset=self.film_val,
                paired_dataset=self.paired_val,
                num_paired_per_batch=self.hparams.num_paired_per_batch,
                num_unpaired_per_batch=self.hparams.num_unpaired_per_batch,
            )
            self.data_test = CombinedDataset(
                digital_dataset=self.digital_test,
                film_dataset=self.film_test,
                paired_dataset=self.paired_test,
                num_paired_per_batch=self.hparams.num_paired_per_batch,
                num_unpaired_per_batch=self.hparams.num_unpaired_per_batch,
            )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
