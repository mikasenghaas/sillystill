import torch.nn.functional as F
from torch import nn


class DSLRLoss(nn.Module):
    """Implements a version of the loss seen in the paper: 'DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks'

    Link: https://arxiv.org/pdf/1704.02470

    TF code: https://github.com/aiff22/DPED

    Contains 4 elements:
    - Colour loss:
    - Texture loss
    - Content loss
    - Total variationa loss
    """

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
