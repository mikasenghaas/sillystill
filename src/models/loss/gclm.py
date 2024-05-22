from typing import Literal
import numpy as np
import torch
from skimage.feature import graycomatrix, graycoprops

from src.models.loss.base import BaseLoss


class GCLMLoss(BaseLoss):

    def name(self) -> str:
        return "gclm_loss"

    def __init__(
        self,
        mode: Literal["statistical", "l1"] = "statistical",
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

        # Define the properties
        self.distances = distances
        self.angles = angles
        self.normalise = normalise
        self.symmetric = symmetric

        # Define the weight
        self.mode = mode

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
        loss = 0
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

            if self.mode == "l1":
                # Calculate L1 loss between GLCM matrices
                loss += torch.tensor(glcm1 - glcm2).abs().sum()
            else:
                # Calculate loss using GLCM properties
                for prop in self.property_weights.keys():
                    prop1 = graycoprops(glcm1, prop)
                    prop2 = graycoprops(glcm2, prop)
                    prop_loss = torch.tensor(prop1 - prop2).abs().sum()
                    loss += prop_loss * self.property_weights[prop]

        # Take mean of losses over batch
        loss /= pred.shape[0]

        return loss


if __name__ == "__main__":

    # Print a table with headings for overall loss and each of the properties, and overall loss with different distances and angles, normalisation, and symmetricity and rows for random, identity, homogeneous vs random

    import pandas as pd
    import random
    from PIL import Image
    from torchvision.transforms import ToTensor

    # Define the properties
    properties = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]

    # Define the distances and angles
    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    # Define the normalisation and symmetricity
    normalise = [True, False]
    symmetric = [True, False]

    # Define the images
    random_image_1 = torch.rand(2, 3, 256, 256)
    random_image_2 = torch.rand(2, 3, 256, 256)
    homogeneous_image = torch.ones(2, 3, 256, 256)

    image_pairs = [
        (random_image_1, random_image_1),
        (random_image_1, random_image_2),
        (random_image_1, homogeneous_image),
    ]
    image_names = ["Identity", "Random", "Homogeneous"]
    # Define the losses
    losses = []
    loss_names = []
    for distance in distances:
        for angle in angles:
            loss_names.append(f"Distance: {distance}, Angle: {angle}")
            losses.append(GCLMLoss(distances=[distance], angles=[angle]))
    for norm in normalise:
        for sym in symmetric:
            loss_names.append(f"Normalise: {norm}, Symmetric: {sym}")
            losses.append(GCLMLoss(normalise=norm, symmetric=sym))

    # Define the results
    results = []

    # Iterate over the losses
    for loss in losses:
        row = []
        for name, (pred, target) in zip(image_names, image_pairs):
            row.append(loss(pred, target).item())
        results.append(row)

    # Create the dataframe
    df = pd.DataFrame(results, columns=image_names, index=loss_names)
    print(df)

    # Print dataframe of default loss parameters vs the loss with different properties
    loss = GCLMLoss()
    results = []
    for image_name, (pred, target) in zip(image_names, image_pairs):
        row = []
        row.append(loss(pred, target).item())
        for prop in properties:
            # Set loss with weight of 1 for property and 0 for others (which is not default and so must be set explicitly)
            loss_properties = {}
            for prop2 in properties:
                loss_properties[prop2 + "_weight"] = 0.0
                loss_properties[prop + "_weight"] = 1.0
            row.append(GCLMLoss(**loss_properties)(pred, target).item())
        results.append(row)

    # Create the dataframe
    df = pd.DataFrame(results, columns=["Default"] + properties, index=image_names)
    print(df)

    # Print df of l1 vs statistical modes for random, identity, and homogeneous images
    results = []
    for image_name, (pred, target) in zip(image_names, image_pairs):
        row = []
        row.append(GCLMLoss(mode="l1")(pred, target).item())
        row.append(GCLMLoss(mode="statistical")(pred, target).item())
        results.append(row)

    # Create the dataframe
    df = pd.DataFrame(results, columns=["L1", "Statistical"], index=image_names)
    print(df)
