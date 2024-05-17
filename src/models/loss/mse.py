import torch.nn.functional as F
from torch import nn


class MSELoss(nn.Module):
    """Returns MSELoss."""

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

        return {"loss": loss}
