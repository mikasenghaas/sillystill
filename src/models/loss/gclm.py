import numpy as np
import torch
from skimage.feature import graycomatrix, graycoprops

from src.models.loss.base import BaseLoss


class GCLMLoss(BaseLoss):

    def name(self) -> str:
        return "gclm_loss"

    def __init__(
        self,
        distances=[1],
        angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        contrast_weight=1.0,
        dissimilarity_weight=1.0,
        homogeneity_weight=1.0,
        energy_weight=1.0,
        correlation_weight=1.0,
        normalise=True,
        symmetric=False,
    ):
        """
        Initialize the GCLM loss.

        Args:
            distances (list): List of pixel distances to consider.
            angles (list): List of angles in radians to consider.
            contrast_weight (float): Weight of contrast in loss.
            dissimilarity_weight (float): Weight of dissimilarity in loss.
            homogeneity_weight (float): Weight of homogeneity in loss.
            energy_weight (float): Weight of energy in loss.
            correlation_weight (float): Weight of correlation in loss.
            normalise (bool): Whether to normalise the GLCM matrices.
            symmetric (bool): Whether to use symmetric GLCM matrices.
        """
        super(GCLMLoss, self).__init__()

        self.distances = distances
        self.angles = angles
        self.normalise = normalise
        self.symmetric = symmetric

        self.contrast_weight = contrast_weight
        self.dissimilarity_weight = dissimilarity_weight
        self.homogeneity_weight = homogeneity_weight
        self.energy_weight = energy_weight
        self.correlation_weight = correlation_weight

        # Define dictionary to map properties to weights
        self.property_weights = {
            "contrast": self.contrast_weight,
            "dissimilarity": self.dissimilarity_weight,
            "homogeneity": self.homogeneity_weight,
            "energy": self.energy_weight,
            "correlation": self.correlation_weight,
        }

    def forward(self, pred, target):
        """
        Compute the GCLM loss between the predicted and target images.

        Args:
            pred (torch.Tensor): Predicted image. Shape (B, C, H, W).
            target (torch.Tensor): Target image. Shape (B, C, H, W).

        Returns:
            torch.Tensor: GCLM loss.
        """

        # Convert images to grayscale if they are not
        pred = pred.mean(dim=1, keepdim=False)
        target = target.mean(dim=1, keepdim=False)

        # GCLM does not accept batched inputs so we iterate over the batch
        loss = torch.tensor(0.0)
        for i in range(pred.shape[0]):
            glcm1 = graycomatrix(
                (pred[i].cpu().detach().numpy() * 255).astype(np.ubyte),
                distances=self.distances,
                angles=self.angles,
                normed=self.normalise,
                symmetric=self.symmetric,
            )
            glcm2 = graycomatrix(
                (target[i].cpu().detach().numpy() * 255).astype(np.ubyte),
                distances=self.distances,
                angles=self.angles,
                normed=self.normalise,
                symmetric=self.symmetric,
            )

            # Iterate over GLCM properties and calculate loss with weights and l1 norm
            for prop in self.property_weights.keys():
                prop1 = graycoprops(glcm1, prop)
                prop2 = graycoprops(glcm2, prop)
                prop_loss = torch.Tensor(prop1 - prop2).abs().sum()
                loss += prop_loss * self.property_weights[prop]

        # Take mean of losses over batch
        loss /= pred.shape[0]

        return loss