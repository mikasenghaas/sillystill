import torch
import torch.nn.functional as F
from src.models.loss.base import BaseLoss

from torchvision.models import vgg19_bn
from torchvision.models.vgg import VGG19_BN_Weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda")


class VGGLoss(BaseLoss):
    def name(self) -> str:
        return "vgg_loss"

    def __init__(self, mse_weight=1.0, feature_weight=1.0):
        super().__init__()
        self.reconstruction_weight = mse_weight
        self.feature_weight = feature_weight

        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).to(device).features
        self.feature_layers = {"3": "conv1_2", "8": "conv2_2", "17": "conv3_2"}
        self.feature_weights = {"conv1_2": 0.4, "conv2_2": 0.4, "conv3_2": 0.2}

    def forward(self, pred, target):
        # Compute the feature loss
        loss = 0
        pred_vgg = self.extract_features(pred)
        target_vgg = self.extract_features(target)
        for layer in self.feature_layers.keys():
            pred_features = pred_vgg[layer]
            target_features = target_vgg[layer]
            weight = self.feature_weights[self.feature_layers[layer]]
            layer_loss = F.mse_loss(pred_features, target_features)
            loss += weight * layer_loss

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
