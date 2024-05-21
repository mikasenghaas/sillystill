import os
import rootutils

import numpy as np
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from matplotlib import pyplot as plt


# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Import utilities
from src.data.components.paired import PairedDataset
from src.models.net import UNet
import torchvision.transforms.v2 as T
import src.models.transforms as CT


def main():
    # set device1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data paths
    film_paired_dir = os.path.join("data", "overfit", "film")
    digital_paired_dir = os.path.join("data", "overfit", "film")

    # datset
    data = PairedDataset((film_paired_dir, digital_paired_dir))
    print(f"init dataset (len={len(data)})")

    # dataloader
    loader = DataLoader(data, batch_size=1, collate_fn=data.collate)
    print(f"init loader (len={len(loader)})")

    # model
    model = UNet(hidden_channels=[64, 128])
    print(f"init unet ({sum([np.prod(p.size()) for p in model.parameters()])})")
    print(model)

    # loss, optimser and lr scheduler
    loss_fn = torch.nn.MSELoss()
    optimiser = Adam(model.parameters(), lr=1e-3)
    # scheduler = ReduceLROnPlateau(optimiser, mode="min", patience=3, factor=0.5)

    to_model = T.Compose([CT.ToModelInput(), T.RandomCrop(128)])
    from_model = CT.FromModelInput()

    # train loop
    model = model.to(device)

    model.train()
    pbar = tqdm(range(400), desc="Loss: X.XXXX")
    for epoch in pbar:
        epoch_loss = 0.0
        for batch in loader:
            # zero grad
            optimiser.zero_grad()

            # to device
            batch = batch.to(device)

            # transforms
            film, digital = to_model(batch)

            # forward pass
            film_predicted = model(digital)
            film_predicted = film_predicted.clamp(0, 1)

            # loss
            loss = loss_fn(film, film_predicted)
            epoch_loss += loss.item()

            # backward pass
            loss.backward()
            optimiser.step()
            # scheduler.step()

        if (epoch + 1) % 10 == 0:
            fig, axs = plt.subplots(nrows=3, figsize=(12, 8))
            axs[0].imshow(CT.pil_to_plot(from_model(digital[0])))
            axs[1].imshow(CT.pil_to_plot(from_model(film[0])))
            axs[2].imshow(CT.pil_to_plot(from_model(film_predicted[0])))
            axs[0].set_ylabel("Digital")
            axs[1].set_ylabel("Film (Ground Truth)")
            axs[2].set_ylabel("Film (Predicted)")
            path = f"dev/epoch_{epoch+1}.png"
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fig.savefig(path)
            plt.close()

        pbar.set_description(f"Loss: {(epoch_loss / len(loader)):.4f}")


if __name__ == "__main__":
    main()
