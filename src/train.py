from typing import Any, Dict, List, Optional, Tuple

import hydra
import torch
import lightning as L
import rootutils
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import utilities
from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    log_hyperparameters,
    task_wrapper,
)

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
    log.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule)

    train_metrics = trainer.callback_metrics

    # Test model
    if cfg.get("test"):
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

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # Apply extra utilities
    extras(cfg)

    # Train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
