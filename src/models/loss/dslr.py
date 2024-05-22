from typing import Literal
import torch.nn.functional as F
from torch import nn
from .grain import GrainLoss
from .color import ColorLoss
from .mse_vgg import MSEVGGLoss


class DSLRLoss(nn.Module):
    """Implements a version of the loss seen in the paper: 'DSLR-Quality Photos on Mobile Devices with Deep Convolutional Networks'

    Link: https://arxiv.org/pdf/1704.02470

    TF code: https://github.com/aiff22/DPED

    Contains 4 elements:
    - Colour loss: Blurred colour channels MSE
    - Texture loss: NOT YET IMPLEMENTED
    - Content loss: VGG loss
    - Total variational loss: Grain loss
    """

    def __init__(
        self,
        color_weight=1.0,
        tv_weight=1.0,
        vgg_weight=1.0,
        grain_loss_mode=Literal["max", "min", "diff"],
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.color_loss = ColorLoss()
        self.tv_los = GrainLoss(mode=grain_loss_mode)
        self.vgg_loss = MSEVGGLoss(mse_weight=0.0, feature_weight=1.0)

        self.color_weight = color_weight
        self.tv_weight = tv_weight
        self.vgg_weight = vgg_weight

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.ß
            y_hat: The predicted image.

        Returns:
            loss: dict of the computed loss value.
        """
        # Colour loss
        color_loss = self.color_loss(y, y_hat)

        # Texture loss
        tv_loss = self.tv_los(y, y_hat)

        # Content loss
        vgg_loss = self.vgg_loss(y, y_hat)

        # Total loss
        loss = (
            self.color_weight * color_loss
            + self.tv_weight * tv_loss
            + self.vgg_weight * vgg_loss
        )

        return {"loss": loss}
