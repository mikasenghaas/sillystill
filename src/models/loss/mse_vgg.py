import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19_bn
from torchvision.models.vgg import VGG19_BN_Weights


class MSEVGGLoss(nn.Module):
    """
    Returns a loss value for the combination of MSE and VGG loss.

    Consists of two components:
    - MSE Reconstruction loss for the digital and film images
    - VGG feature loss for the digital and film images
    """

    def __init__(self, mse_weight=1.0, feature_weight=1.0):
        super().__init__()
        self.reconstruction_weight = mse_weight
        self.feature_weight = feature_weight

        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).features
        self.feature_layers = {"3": "conv1_2", "8": "conv2_2", "17": "conv3_2"}
        self.feature_weights = {"conv1_2": 0.4, "conv2_2": 0.4, "conv3_2": 0.2}

    def forward(self, y, y_hat):
        """Compute the loss.

        Args:
            y: The ground truth image.
            y_hat: The predicted image.

        Returns:
            loss: The computed loss value.
        """

        # Compute the reconstruction loss
        reconstruction_loss = F.mse_loss(y, y_hat)

        # Compute the feature loss
        feature_loss = 0
        y_vgg = self.extract_features(y)
        y_hat_vgg = self.extract_features(y_hat)
        for layer in self.feature_layers.keys():
            y_features = y_vgg[layer]
            y_hat_features = y_hat_vgg[layer]
            feature_loss += self.feature_weights[
                self.feature_layers[layer]
            ] * F.mse_loss(y_features, y_hat_features)

        # Combine the losses
        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.feature_weight * feature_loss
        )

        return loss

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
