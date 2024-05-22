import torch.nn.functional as F
from torch import nn
from .mse_vgg import MSEVGGLoss
from .grain import GrainLoss


class MSEVGGGrainLoss(nn.Module):
    """
    Returns a loss value for the combination of MSE and VGG loss.

    Consists of two components:
    - MSE Reconstruction loss for the digital and film images
    - VGG feature loss for the digital and film images
    """

    def __init__(self, mse_weight=1.0, feature_weight=1.0, grain_weight=1.0):
        super().__init__()
        self.reconstruction_weight = mse_weight
        self.feature_weight = feature_weight
        self.grain_weight = grain_weight

        self.mse_vgg = MSEVGGLoss(mse_weight, feature_weight)
        self.grain = GrainLoss()

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.
            y_hat: The predicted image.

        Returns:
            loss: dict of loss values
        """

        # Compute the MSE and VGG loss
        mse_vgg_loss = self.mse_vgg(y, y_hat)
        reconstruction_loss = mse_vgg_loss["reconstruction_loss"]
        feature_loss = mse_vgg_loss["feature_loss"]

        # Compute the grain loss
        grain_loss = self.grain(y, y_hat)

        total_loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.feature_weight * feature_loss
            + self.grain_weight * grain_loss["loss"]
        )

        # Combine the losses
        loss = {
            "reconstruction_loss": reconstruction_loss,
            "feature_loss": feature_loss,
            "grain_loss": grain_loss["loss"],
            "loss": total_loss,
        }

        return loss
