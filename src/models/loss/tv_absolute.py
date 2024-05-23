import torch
from torchmetrics.functional.image import total_variation
from typing import Literal
from src.models.loss.base import BaseLoss


class TVAbsoluteLoss(BaseLoss):
    """
    Returns a loss value for grain based on total variation loss.
    """

    def name(self) -> str:
        return "tv_absolute_loss"

    def __init__(self, mode: Literal["max", "min"] = "max", grayscale: bool = True):
        """
        Instantiates the GrainLoss class.

        Args:
            mode (str): The mode to compute the total variation loss. Can be "diff" or "max". Diff computes the difference between the total variation of the ground truth and predicted images. Max priorities maximum total variation of the predicted image.
        """
        super().__init__()
        self.mode = mode
        self.grayscale = grayscale

    def forward(self, pred, target):
        """Compute the loss.

        Args:
            pred: The predicted image.
            target: The ground truth image.

        Returns:
            loss: dict of loss values
        """

        if self.grayscale:
            # Convert the images to grayscale
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        if self.mode == "max":
            # Compute the total variation loss
            tv_pred = total_variation(pred)
            tv_pred /= pred.numel()

            # Loss is the reciprocal of the total variation
            loss = 1 / tv_pred

        elif self.mode == "min":
            # Compute the total variation loss
            tv_pred = total_variation(pred)
            tv_pred /= pred.numel()

            # Loss is the total variation
            loss = tv_pred

        return loss


if __name__ == "__main__":
    y = torch.rand(1, 3, 256, 256)
    y_hat = torch.rand(1, 3, 256, 256)
    y_ones = torch.ones(1, 3, 256, 256)
    print("=" * 50)
    print("Max")
    print("=" * 50)
    grain_loss = TVAbsoluteLoss(mode="max")
    loss = grain_loss(y, y_hat)
    print("Random")
    print(loss)
    loss = grain_loss(y, y_ones)
    print("Ones")
    print(loss)
