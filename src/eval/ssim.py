from torch import Tensor
from torchmetrics import Metric

class SSIMPetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def update(self, preds: Tensor, target: Tensor) -> None:
        pass

    def compute(self) -> Tensor:
        pass