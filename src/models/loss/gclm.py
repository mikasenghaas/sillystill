import numpy as np
import torch
import torch.nn as nn
from skimage.feature import graycomatrix, graycoprops

from src.models.loss.base import BaseLoss


class GCLMLoss(BaseLoss):

    def name(self) -> str:
        return "GCLM_loss"

    def _init_(self):
        super(GCLMLoss, self)._init_()

    def forward(self, pred, target):
        # Convert images to grayscale if they are not
        if pred.shape[1] == 3:
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        glcm_dist = 1  # Distance for GLCM calculation
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

        glcm1 = graycomatrix(
            (pred[0, 0].cpu().detach().numpy() * 255).astype(np.ubyte),
            [glcm_dist],
            angles,
            normed=True,
        )
        glcm2 = graycomatrix(
            (target[0, 0].cpu().detach().numpy() * 255).astype(np.ubyte),
            [glcm_dist],
            angles,
            normed=True,
        )

        loss = 0
        for prop in [
            "contrast",
            "dissimilarity",
            "homogeneity",
            "energy",
            "correlation",
        ]:
            loss += (
                torch.tensor(graycoprops(glcm1, prop) - graycoprops(glcm2, prop))
                .abs()
                .sum()
            )

        return loss
