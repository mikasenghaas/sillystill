import os
import sys
import rootutils

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from matplotlib import pyplot as plt


# Setup root
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Local modules
from src.data.components.paired import PairedDataset
from src.models.net.unet import UNet
import src.models.transforms as CT
import src.models.loss as L


def main():
    # get args
    args = sys.argv[1:]

    # set device1
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data paths (grain f/ f)
    film_paired_dir = os.path.join("data", "grain", "film")
    digital_paired_dir = os.path.join("data", "grain", "digital")

    # datset
    batch_size = 1
    train_data = PairedDataset(
        (film_paired_dir, digital_paired_dir), duplicate=batch_size
    )
    test_data = PairedDataset((film_paired_dir, digital_paired_dir), duplicate=1)
    print(f"init dataset (len={len(train_data)})")

    # dataloader
    train_loader = DataLoader(
        train_data, batch_size=batch_size, collate_fn=train_data.collate
    )
    test_loader = DataLoader(test_data, batch_size=1, collate_fn=test_data.collate)
    print(f"init loader (len={len(train_loader)})")

    # model
    model = UNet(hidden_channels=[64, 128, 256], kernel_size=3, with_noise=True)
    print(f"init unet ({sum([np.prod(p.size()) for p in model.parameters()])})")
    print(model)

    # loss, optimser
    loss_fn = L.SillyLoss([L.MSELoss()], weights=[1.0])
    optimiser = Adam(model.parameters(), lr=1e-3)

    to_model = CT.TrainTransforms(patch_size=256, augment=0.0)
    from_model = CT.FromModelInput()

    # train loop
    model = model.to(device)

    if len(args) == 0:
        model.train()
        total_patches = 400
        steps = total_patches // batch_size
        patches_seen = 0
        pbar = tqdm(total=total_patches, desc="Loss: X.XXXX")
        for _ in range(steps):
            epoch_loss = 0.0
            for batch in train_loader:
                # zero grad
                optimiser.zero_grad()

                # to device
                batch = batch.to(device)

                # transforms
                film, digital = to_model(batch)

                # forward pass
                film_predicted = model(digital)

                # loss
                losses = loss_fn(film_predicted, film)
                loss = losses["loss"]
                epoch_loss += loss.item()

                # backward pass
                loss.backward()
                optimiser.step()
                # scheduler.step(loss)

                patches_seen += batch_size

                if patches_seen % 25 == 0:
                    print("plotting")
                    fig, axs = plt.subplots(nrows=3, figsize=(12, 8))
                    axs[0].imshow(CT.pil_to_plot(from_model(digital[0])))
                    axs[1].imshow(CT.pil_to_plot(from_model(film[0])))
                    axs[2].imshow(CT.pil_to_plot(from_model(film_predicted[0])))
                    axs[0].set_ylabel("Digital")
                    axs[1].set_ylabel("Film (Ground Truth)")
                    axs[2].set_ylabel("Film (Predicted)")
                    path = f"dev/patches_{patches_seen}.png"
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    fig.savefig(path)
                    plt.close()

                pbar.set_description(f"Loss: {(epoch_loss / len(train_loader)):.4f}")
                pbar.update(batch_size)

        # save model
        torch.save(model.state_dict(), "dev/model.pt")

    # load model
    print("loading model")
    model.load_state_dict(torch.load("dev/model.pt"))

    # inference
    print("running inference")
    to_infer = CT.TestTransforms((1216, 1816))
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            # to device
            batch = batch.to(device)

            # unpack and resize
            film, digital = batch
            film = to_infer(film)
            digital = to_infer(digital)

            # forward pass
            film_predicted = model(digital)

            # save gens
            from_model(digital[0]).save("dev/digital.png")
            from_model(film[0]).save("dev/film.png")
            from_model(film_predicted[0]).save("dev/infer.png")

            fig, axs = plt.subplots(nrows=3, figsize=(12, 8))
            axs[0].imshow(CT.pil_to_plot(from_model(digital[0])))
            axs[1].imshow(CT.pil_to_plot(from_model(film[0])))
            axs[2].imshow(CT.pil_to_plot(from_model(film_predicted[0])))
            axs[0].set_ylabel("Digital")
            axs[1].set_ylabel("Film (Ground Truth)")
            axs[2].set_ylabel("Film (Predicted)")
            fig.savefig("dev/final_infer.png")


if __name__ == "__main__":
    main()
