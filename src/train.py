from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import hydra
import torch
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from torchmetrics import MetricCollection
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS

from matplotlib import pyplot as plt

import wandb

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import utilities
from src.data.components.paired import PairedDataset
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
)
from src.eval import PieAPP
import src.models.transforms as CT

# Setup logger
log = RankedLogger(__name__, rank_zero_only=True)

# Resolve torch data types
OmegaConf.register_new_resolver("torch_dtype", lambda x: getattr(torch, x))


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Trains the model. Can additionally evaluate on a testset, using best weights
    obtained during training.

    Args:
        cfg (DictConfig): Training configuration (composed by Hydra)

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: Tuple containing two dictionaries:
            - metric_dict: Dictionary containing training and testing metrics.
            - object_dict: Dictionary containing all objects instantiated during training.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger: Optional[Logger] = None
    if cfg.get("logger"):
        log.info(f"Instantiating logger <{cfg.logger._target_}>")
        logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=logger
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    # Log hyperparameters
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    # Train model
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        # Test model
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if cfg.get("inference"):
        # Test inference
        log.info("Starting inference")

        compute_metrics = MetricCollection(
            {
                "ssim": SSIM().to(device),
                "psnr": PSNR().to(device),
                "lpips": LPIPS(net_type="squeeze", normalize=True).to(device),
                "pieapp": PieAPP().to(device),
            }
        )

        paired_image_data = PairedDataset(cfg.data.image_dirs)
        data = []
        for film, digital in tqdm(paired_image_data):
            model.eval()
            model = model.to(device)

            # Run inference
            with torch.no_grad():
                # Process from PILImage to Tensor
                film = CT.to_infer(film, downsample=2, device=device)
                digital = CT.to_infer(digital, downsample=2, device=device)

                # Run inference
                film_predicted = model(digital).clamp(0 + 1e-5, 1 - 1e-5)

                # Compute metrics
                infer_metrics = compute_metrics(film_predicted, film)
                baseline_metrics = compute_metrics(digital, film)

                # Process from Tensor to PILImage
                film_predicted = CT.FromModelInput()(film_predicted.squeeze())
                digital = CT.FromModelInput()(digital.squeeze())
                film = CT.FromModelInput()(film.squeeze())

                data.append(
                    {
                        "run_name": logger.experiment.name,
                        "digital": wandb.Image(digital),
                        "film": wandb.Image(film),
                        "predicted": wandb.Image(film_predicted),
                        **{k: v.item() for k, v in infer_metrics.items()},
                        **{
                            f"baseline_{k}": v.item()
                            for k, v in baseline_metrics.items()
                        },
                    }
                )

        if logger:
            table = wandb.Table(dataframe=pd.DataFrame(data))
            logger.experiment.log({"inference/table": table})

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Apply extra utilities
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # Safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # Return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
