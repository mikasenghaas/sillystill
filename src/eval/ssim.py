from torch import Tensor
from torchmetrics import Metric
from ignite.metrics import SSIM

class SSIMMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ssim = SSIM(data_range=1.0)
    
    def update(self, preds: Tensor, target: Tensor) -> None:
        self.ssim.update(output=(preds, target))
    
    def compute(self) -> Tensor:
        return self.ssim.compute()