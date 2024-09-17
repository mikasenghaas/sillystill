import torch.nn.functional as F
from torch import nn
from src.models.loss.base import BaseLoss


class MSELoss(BaseLoss):
    """Returns MSELoss."""

    def name(self) -> str:
        return "mse_loss"

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.ß
            y_hat: The predicted image.

        Returns:
            loss: dict of the computed loss value.
        """

        # Compute the reconstruction loss
        loss = F.mse_loss(y, y_hat)

        return loss
