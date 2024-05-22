import torch.nn.functional as F
from torch import nn
import torch
from torchmetrics import TotalVariation
from typing import Literal


class GrainLoss(nn.Module):
    """
    Returns a loss value for grain based on total variation loss.
    """

    def __init__(self, mode=Literal["diff", "max"]):
        """
        Instantiates the GrainLoss class.

        Args:
            mode (str): The mode to compute the total variation loss. Can be "diff" or "max". Diff computes the difference between the total variation of the ground truth and predicted images. Max priorities maximum total variation of the predicted image.
        """
        super().__init__()
        self.mode = mode
        self.total_variation = TotalVariation()

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.
            y_hat: The predicted image.

        Returns:
            loss: dict of loss values
        """

        if self.mode == "diff":

            # Compute the total variation loss
            tv_y_hat = self.total_variation(y_hat)
            tv_y = self.total_variation(y)

            # Normalize the total variation loss by the size of the image
            tv_y_hat /= y_hat.numel()
            tv_y /= y.numel()

            # Combine the losses
            loss = F.l1_loss(tv_y_hat, tv_y)

        elif self.mode == "max":
            # Compute the total variation loss
            tv_y_hat = self.total_variation(y_hat)
            tv_y_hat /= y_hat.numel()

            # Loss is the reciprocal of the total variation
            loss = 1 / tv_y_hat

        return {
            "loss": loss,
        }

    def extract_features(self, img):
        """Extracts features from the input image using the VGG19 model.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            dict: A dictionary containing the extracted features from the VGG19 model.
        """

        features = {}
        x = img
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.feature_layers:
                features[name] = x
        return features


if __name__ == "__main__":
    # Test the GrainLoss class
    grain_loss = GrainLoss(mode="diff")
    y = torch.rand(1, 3, 256, 256)
    y_hat = torch.rand(1, 3, 256, 256)
    y_ones = torch.ones(1, 3, 256, 256)
    print("=" * 50)
    print("Diff")
    print("=" * 50)
    loss_rand = grain_loss(y, y_hat)
    print("Random images")
    print(loss_rand)
    loss_identity = grain_loss(y, y)
    print("Same images")
    print(loss_identity)
    loss_ones = grain_loss(y, y_ones)
    print("Rand/Ones images")
    print(loss_ones)

    print("=" * 50)
    print("Max")
    print("=" * 50)
    grain_loss = GrainLoss(mode="max")
    loss = grain_loss(y, y_hat)
    print("Random")
    print(loss)
    loss = grain_loss(y, y_ones)
    print("Ones")
    print(loss)
