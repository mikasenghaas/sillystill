import torch.nn.functional as F
from src.models.loss.base import BaseLoss


class MAELoss(BaseLoss):
    """Returns MAELoss."""

    def name(self) -> str:
        return "mae_loss"

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.ß
            y_hat: The predicted image.

        Returns:
            loss: dict of the computed loss value.
        """

        # Compute the reconstruction loss
        loss = F.l1_loss(y, y_hat)

        return loss
