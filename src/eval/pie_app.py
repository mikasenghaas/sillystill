from torch import Tensor, uint8
from torchmetrics import Metric
from piq import PieAPP

class PieAPPMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def denormalize(self, img: Tensor) -> Tensor:
        """
        Denormalize an image tensor. Should make sure that the image data is in the range [0.0, 1.0].
        Does not permute the dimensions of the image tensor.
        """
        # Denormalise
        mean = Tensor([0.485, 0.456, 0.406]).reshape(3, 1, 1)
        std = Tensor([0.229, 0.224, 0.225]).reshape(3, 1, 1)
        img = img * std + mean

        # Convert back to uint8
        img = img.type(uint8)

        # print(f"Data values in img: {img.min().item()}, {img.max().item()}")

        return img
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        x = preds
        y = target
        x = self.denormalize(x)
        y = self.denormalize(y)
        self.pieapp_loss: Tensor = PieAPP(reduction='none', stride=32)(x, y)
    
    def compute(self) -> Tensor:
        # subtract from 1.0 because this library returns the distance, not the similarity
        return 1.0 - self.pieapp_loss.item()
        
