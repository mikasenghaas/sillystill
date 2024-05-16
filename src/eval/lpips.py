from torch import Tensor, uint8
from torchmetrics import Metric
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class LPIPSMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze')

    def denormalize(self, img: Tensor) -> Tensor:
        """
        Denormalize an image tensor. Should make sure that the image data is in the range [-1.0, 1.0].
        Does not permute the dimensions of the image tensor.
        """
        # Denormalise
        mean = Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        mean_adjust = Tensor([-1.0, -1.0, -1.0]).reshape(3, 1, 1)
        std = Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean + mean_adjust

        return img
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        x = preds
        y = target
        x = self.denormalize(x)
        y = self.denormalize(y)
        self.lpips_loss: Tensor = self.lpips(x, y)
    
    def compute(self) -> Tensor:
        # subtract from 1.0 because this library returns the distance, not the similarity
        return 1.0 - self.lpips_loss.item()