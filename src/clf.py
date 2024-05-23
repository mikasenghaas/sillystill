import os
import rootutils

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models.resnet import ResNet18_Weights
from tqdm import tqdm

from matplotlib import pyplot as plt


# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import utilities
from src.data.components.paired import PairedDataset
import torchvision.transforms as T
import src.models.transforms as CT


def main():
    # set device1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data paths (grain f/ f)
    film_paired_dir = os.path.join("data", "paired", "processed", "film")
    digital_paired_dir = os.path.join("data", "paired", "processed", "digital")

    # datset
    batch_size = 8
    data = PairedDataset((film_paired_dir, digital_paired_dir), duplicate=1)
    print(f"init dataset (len={len(data)})")

    # dataloader
    loader = DataLoader(data, batch_size=batch_size, collate_fn=data.collate)
    print(f"init loader (len={len(loader)})")

    # model
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    # model = resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 1)  # Binary classification (2 classes)

    # loss, optimser and lr scheduler
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimiser = Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=50, factor=0.5, verbose=True)

    to_model = ResNet18_Weights.IMAGENET1K_V1.transforms()

    # train loop
    model = model.to(device)

    model.train()
    pbar = tqdm(range(50), desc="Loss: X.XXXX")
    for _ in pbar:
        for batch in loader:
            # zero grad
            optimiser.zero_grad()

            # to device
            batch = batch.to(device)

            # transforms
            film, digital = batch
            #  fig, axs = plt.subplots(nrows=2, figsize=(12, 8))
            #  axs[0].imshow(CT.pil_to_plot(digital.cpu()[0].permute(1,2,0)))
            #  axs[1].imshow(CT.pil_to_plot(film.cpu()[0].permute(1,2,0)))
            #  axs[0].set_ylabel("Digital")
            #  axs[1].set_ylabel("Film")
            #  fig.savefig("clf/before_trans.png")
            film = to_model(film)
            digital = to_model(digital)
            # fig, axs = plt.subplots(nrows=2, figsize=(12, 8))
            # axs[0].imshow(CT.pil_to_plot(digital.cpu()[0].permute(1,2,0)))
            # axs[1].imshow(CT.pil_to_plot(film.cpu()[0].permute(1,2,0)))
            # axs[0].set_ylabel("Digital")
            # axs[1].set_ylabel("Film")
            # fig.savefig("clf/after_trans.png")

            # forward pass
            film_predicted = model(film)
            digital_predicted = model(digital)

            # targets
            logits = torch.cat([film_predicted, digital_predicted], dim=0)
            targets = torch.cat([torch.ones_like(film_predicted), torch.zeros_like(digital_predicted)], dim=0)

            # loss
            loss = loss_fn(logits, targets)

            # acc
            preds = (torch.sigmoid(logits) > 0.5)
            acc = (preds == targets).sum() / logits.numel()
            print(preds.flatten().type(torch.Tensor))
            print(targets.flatten())

            # backward pass
            loss.backward()
            optimiser.step()
            # scheduler.step(loss)

            pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {acc.item()}")

    # save model
    torch.save(model.state_dict(), "clf/model.pt")

if __name__ == "__main__":
    main()
