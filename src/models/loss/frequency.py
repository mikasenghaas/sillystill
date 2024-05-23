import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from src.models.loss.base import BaseLoss


class FrequencyLoss(BaseLoss):
    def name(self) -> str:
        return "frequency_loss"

    def _init_(self):
        super(FrequencyLoss, self)._init_()

    def forward(self, pred, target):
        # Convert images to grayscale if they are not
        pred = pred.mean(dim=1, keepdim=False)
        target = target.mean(dim=1, keepdim=False)

        # Convert images to frequency domain
        fft1 = fft.fft2(pred)
        fft2 = fft.fft2(target)

        loss = F.mse_loss(torch.abs(fft1), torch.abs(fft2))

        return loss
