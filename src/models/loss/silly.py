from enum import Enum
import torch.nn as nn
from src.models.loss import (
    BaseLoss,
    ColorLoss,
    TVAbsoluteLoss,
    TVRelativeLoss,
    MAELoss,
    MSELoss,
    VGGLoss,
    GCLMLoss,
    FrequencyLoss,
    CoBiLoss,
)


# class Loss(Enum):
#     MAE = MAELoss
#     MSE = MSELoss
#     Color = ColorLoss
#     VGG = VGGLoss
#     TV_Absolute = TVAbsoluteLoss
#     TV_Relative = TVRelativeLoss
#     GCLM = GCLMLoss
#     Frequency = FrequencyLoss
#     CoBi = CoBiLoss


class SillyLoss(nn.Module):
    def __init__(self, losses: list[BaseLoss], weights: list[float]):
        super().__init__()

        self.loss_fns = losses
        self.weights = weights

    def forward(self, pred, target):
        """
        Args:
            pred: The predicted image.
            target: The ground truth image.

        Returns:
            loss: dict of loss values, including the total loss ("loss") and individual loss components.
        """

        loss = 0
        loss_dict = {}

        for i, loss_fn in enumerate(self.loss_fns):
            loss_dict[loss_fn.name()] = loss_fn(pred, target)
            loss += self.weights[i] * loss_dict[loss_fn.name()]

        loss_dict["loss"] = loss

        return loss_dict