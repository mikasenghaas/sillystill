import torch.nn.functional as F
import torch
from torchmetrics.functional.image import total_variation
from src.models.loss.base import BaseLoss


class TVRelativeLoss(BaseLoss):
    """
    Returns a loss value for grain based on total variation loss.
    """

    def name(self) -> str:
        return "tv_relative_loss"

    def __init__(self, grayscale: bool = True):
        """
        Instantiates the TVRelative class.
        """
        super().__init__()
        self.grayscale = grayscale

    def forward(self, pred, target):
        """Compute the loss.

        Args:
            pred: The predicted image.
            target: The ground truth image.

        Returns:
            loss: dict of loss values
        """

        if self.grayscale:
            # Convert the images to grayscale
            pred = pred.mean(dim=1, keepdim=True)
            target = target.mean(dim=1, keepdim=True)

        # Compute the total variation loss
        tv_pred = total_variation(pred)
        tv_target = total_variation(target)

        # Normalize the total variation loss by the size of the image
        tv_pred /= pred.numel()
        tv_target /= target.numel()

        # Combine the losses
        loss = F.l1_loss(tv_pred, tv_target)

        return loss


if __name__ == "__main__":
    # Test the GrainLoss class
    grain_loss = TVRelativeLoss()
    y = torch.rand(1, 3, 256, 256)
    y_hat = torch.rand(1, 3, 256, 256)
    y_ones = torch.ones(1, 3, 256, 256)
    print("=" * 50)
    print("Diff")
    print("=" * 50)
    loss_rand = grain_loss(y, y_hat)
    print("Random images")
    print(loss_rand)
    loss_identity = grain_loss(y, y)
    print("Same images")
    print(loss_identity)
    loss_ones = grain_loss(y, y_ones)
    print("Rand/Ones images")
    print(loss_ones)
