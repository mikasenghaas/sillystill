from torch import nn
from abc import abstractmethod, ABC


class BaseLoss(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self) -> str:
        pass
