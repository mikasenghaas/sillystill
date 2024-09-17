import torch
from torchmetrics import Metric
from torchmetrics.image.fid import FrechetInceptionDistance


class FID(Metric):
    """
    Frechet Inception Distance (FID) metric for evaluating image generation models.

    NOTE: This metric requires a batch of images to be passed to it, not individual images as
    it estimates parameters of the distribution of the generated images and the real images.
    Hence, it cannot be used with a batch_size of 1.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.fid = FrechetInceptionDistance(feature=64, normalize=True)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        self.fid.update(preds, real=False)
        self.fid.update(targets, real=True)

    def compute(self) -> torch.Tensor:
        return self.fid.compute()
