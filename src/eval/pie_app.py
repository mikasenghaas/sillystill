import torch
from torchmetrics import Metric
from piq import PieAPP as PieAPPBase


class PieAPP(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pie_app = PieAPPBase(reduction="none", stride=128)
        self.add_state(
            "loss", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state(
            "total", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        assert (
            preds.shape == targets.shape
        ), f"Expected preds and targets to have the same shape, got {preds.shape} and {targets.shape}"
        # Clamp inputs to [0, 1] range
        preds = preds.clamp(0, 1)
        targets = targets.clamp(0, 1)

        # Compute PieAPP error (batched)
        pieapp_dist = self.pie_app(preds, targets)  # [B,]

        self.loss += pieapp_dist.sum()
        self.total += pieapp_dist.shape[0]

    def compute(self) -> torch.Tensor:
        return self.loss / self.total
