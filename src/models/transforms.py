import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL.Image import Image as PILImage


def ToModelInput():
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def FromModelInput():
    return T.ToPILImage()


def Augment(augment: float):
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=augment),
            T.RandomVerticalFlip(p=augment),
            T.RandomApply(
                nn.ModuleList([T.GaussianBlur(kernel_size=(27, 27), sigma=(2.5, 5.0))]),
                augment,
            ),
            T.RandomApply(nn.ModuleList([T.ColorJitter(brightness=(0.5, 1))]), augment),
        ]
    )


def pil_to_plot(img: PILImage):
    return np.array(img)


def tensor_to_plot(img: torch.Tensor):
    return img.permute(1, 2, 0).numpy()
