from enum import Enum
from torch import nn
from src.models.loss import (
    BaseLoss,
    # ColorLoss,
    # TVAbsoluteLoss,
    # TVRelativeLoss,
    # MSELoss,
    # VGGLoss,
    # GCLMLoss,
    # FrequencyLoss,
    # CoBiLoss,
)


# class Loss(Enum):
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

        # for loss in losses:
        #     if not isinstance(loss, Loss):
        #         raise ValueError(f"Loss must be an instance of Loss Enum, got {loss}")

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


if __name__ == "__main__":
    pass
    # Example usage
    # import torch
    # from src.models.loss import Loss

    # # Create a loss function that is a combination of MSE and VGG loss
    # loss = SillyLoss([Loss.MSE, Loss.VGG], [0.5, 0.5])

    # # Generate some random images
    # pred = torch.rand(1, 3, 256, 256)
    # target = torch.rand(1, 3, 256, 256)

    # # Compute the loss
    # loss_dict = loss(pred, target)

    # print(loss_dict)
    # Output: {'mse_loss': tensor(0.0837), 'vgg_loss': tensor(0.0837), 'loss': tensor(0.0837)}
