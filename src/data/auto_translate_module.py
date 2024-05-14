import os
from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.components.auto_translate import AutoTranslateDataset
from src.data.components.paired import PairedDataset
from src.data.components.unpaired import UnpairedDataset


class CombinedDataModule(LightningDataModule):
    """
    Lightning data module for the digital-film image pair dataset.
    """

    def __init__(
        self,
        dataset: Dataset,
        data_dir: str = "data/",
        patch_size: int = 128,
        train_val_test_split: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ) -> None:
        """Initialise a `DataModule`.

        Args:
            data_dir (str): The data directory. Defaults to `"data/"`.
            data_split (Tuple[float, float, float]): The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
            batch_size (int): The batch size. Defaults to `64`.
            num_workers (int): The number of workers. Defaults to `0`.
            pin_memory (bool): Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(logger=False)

        # Save paths
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        self.digital_dir = os.path.join(data_dir, "unsplash_digital")
        self.film_dir = os.path.join(data_dir, "unsplash_film_less_filtered")

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.batch_size_per_device = 1

    def prepare_data(self) -> None:
        """Asserts that the raw and processed data directories exist.

        If not, it raises an error.
        """
        # Asserts data exists
        raw_dir_exists = os.path.exists(self.raw_dir)
        processed_dir_exists = os.path.exists(self.processed_dir)
        assert (
            raw_dir_exists and processed_dir_exists
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
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = (
                self.hparams.batch_size // self.trainer.world_size
            )

        # Instantiate paired and unpaired datasets
        paired_dataset = PairedDataset(
            data_dir=self.processed_dir, patch_size=self.hparams.patch_size
        )
        unpaired_digital_dataset = UnpairedDataset(
            data_dir=self.digital_dir, patch_size=self.hparams.patch_size
        )
        unpaired_film_dataset = UnpairedDataset(
            data_dir=self.film_dir, patch_size=self.hparams.patch_size
        )

        # Split each of these into train, val, and test
        self.paired_train, self.paired_val, self.paired_test = random_split(
            dataset=paired_dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.digital_train, self.digital_val, self.digital_test = random_split(
            dataset=unpaired_digital_dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )
        self.film_train, self.film_val, self.film_test = random_split(
            dataset=unpaired_film_dataset,
            lengths=self.hparams.train_val_test_split,
            generator=torch.Generator().manual_seed(42),
        )

        # Instantiate the AutoTranslateDatasets
        self.data_train = AutoTranslateDataset(
            digital_dataset=self.digital_train,
            film_dataset=self.film_train,
            paired_dataset=self.paired_train,
        )
        self.data_val = AutoTranslateDataset(
            digital_dataset=self.digital_val,
            film_dataset=self.film_val,
            paired_dataset=self.paired_val,
        )
        self.data_test = AutoTranslateDataset(
            digital_dataset=self.digital_test,
            film_dataset=self.film_test,
            paired_dataset=self.paired_test,
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        Returns     train_loader (DataLoader): The train dataloader
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        Returns     val_loader (DataLoader): The validation dataloader
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=True,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        Returns     test_loader (DataLoader): The test dataloader
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        stage (Optional[str]): The stage being torn down. Either `"fit"`, `"validate"`, `"test"`,
        or `"predict"`.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        Returns     A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        Returns     state_dict (Dict[str, Any]): The datamodule state (from `state_dict()`
        """
        pass
