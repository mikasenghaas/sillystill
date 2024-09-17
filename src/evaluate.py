# Imports
import os
import json
import wandb
import rootutils
from matplotlib import pyplot as plt

# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.components import PairedDataset
from src.models import TranslationModule
from src.eval import PieAPP
from src.models.transforms import FromModelInput, ToModelInput, TestTransforms, get_valid_dim

import torch
from tqdm import tqdm
from torch.utils.data import Subset
from torchmetrics import MetricCollection
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    PeakSignalNoiseRatio as PSNR,
)

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS


# Constants
api = wandb.Api()

# Define W&B Run ID
USER = "sillystill"
PROJECT = "sillystill"
RUN_ID = "8edwcgyg"
VERSION = "v0"
SAVE_IMAGES = True

# Define local path
LOCAL_PATH = "logs/hydra/runs/2024-05-16_22-10-43/checkpoints/best.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
SUBSET = False
SUBSET_SIZE = 1
DOWNSAMPLE = 2


def main():
    # Download from checkpoint
    if RUN_ID and VERSION:
        try:
            CKPT = f"{USER}/{PROJECT}/model-{RUN_ID}:{VERSION}"
            artifact = api.artifact(CKPT)
            artifact.download()
            path = os.path.join("artifacts", f"model-{RUN_ID}:{VERSION}", "model.ckpt")
            print(f"✅ Successfully downloaded checkpoint from {CKPT} to {path}")
        except Exception as e:
            path = LOCAL_PATH
            print(f"ℹ️ Could not download checkpoint from {CKPT}")
            print(f"✅ Loaded local path {path}")

    # Load the checkpoint
    model = TranslationModule.load_from_checkpoint(path).to(device)

    print(f"✅ Loaded model from {path} (Device: {model.device})")

    # Load images
    film_paired_dir = os.path.join("data", "paired", "processed", "film")
    digital_paired_dir = os.path.join("data", "paired", "processed", "digital")
    digital_film_data = PairedDataset(image_dirs=(film_paired_dir, digital_paired_dir))
    if SUBSET:
        digital_film_data = Subset(digital_film_data, range(SUBSET_SIZE))

    print(f"✅ Loaded {len(digital_film_data)} image pairs")

    # Define metrics
    metrics = MetricCollection(
        {
            "ssim": SSIM().to(device),
            "psnr": PSNR().to(device),
            "lpips": LPIPS().to(device),
            "pieapp": PieAPP().to(device),
        }
    )

    # Run inference on all images
    all_metrics = {}
    for idx, (film, digital) in tqdm(
        enumerate(digital_film_data), total=len(digital_film_data)
    ):
        # Process images to be in the same format as test images
        height = get_valid_dim(film.size[1], downsample=DOWNSAMPLE)
        width = get_valid_dim(film.size[0], downsample=DOWNSAMPLE)
        film_transform = TestTransforms(dim=(height, width))

        height = get_valid_dim(digital.size[1], downsample=DOWNSAMPLE)
        width = get_valid_dim(digital.size[0], downsample=DOWNSAMPLE)
        digital_transform = TestTransforms(dim=(height, width))

        film = film_transform(film)
        digital = digital_transform(digital)

        # Move images to device
        film = film.to(device).unsqueeze(0)
        digital = digital.to(device).unsqueeze(0)

        # Run inference
        film_predicted = model(digital)

        for metric in metrics:
            if metric not in all_metrics:
                all_metrics[metric] = []

            score = metrics[metric](film, film_predicted)

            if isinstance(score, torch.Tensor):
                score = score.item()

            all_metrics[metric].append(score)

        if SAVE_IMAGES:
            # Convert back to PIL images
            film = FromModelInput()(film.squeeze(0))
            digital = FromModelInput()(digital.squeeze(0))
            film_predicted = FromModelInput()(film_predicted.squeeze(0))
            # Save images
            save_dir = f"outputs/{RUN_ID}/{idx}"
            os.makedirs(save_dir, exist_ok=True)
            digital.save(f"{save_dir}/digital.png")
            film.save(f"{save_dir}/film.png")
            film_predicted.save(f"{save_dir}/film_predicted.png")

    # Print metrics
    means = {
        metric: sum(scores) / len(scores) for metric, scores in all_metrics.items()
    }
    for metric, score in means.items():
        print(f"Mean {metric}: {score}")

    # Save metrics
    save_dir = f"outputs/{RUN_ID}"
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = f"outputs/{RUN_ID}/metrics.json"
    means_path = f"outputs/{RUN_ID}/means.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f)
    with open(means_path, "w") as f:
        json.dump(means, f)
    print(f"✅ Saved detailed and mean metrics to {metrics_path}")


if __name__ == "__main__":
    main()
