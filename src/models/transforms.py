from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
# import torchvision.transforms.v2 as T
from PIL.Image import Image as PILImage

def ToBatchedTensor():
    return T.Compose([T.ToTensor(), T.Resize((2433, 3637))]) 
    # return T.Compose([T.ToImage(), T.Resize((2433, 3637))]) # transforms.v2

def ToModelInput():
    return T.ToTensor()
    # return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)]) # transforms.v2

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

def TrainTransforms(patch_size: int, augment: float):
    return T.Compose(
        [
            Augment(augment),
            T.RandomCrop(patch_size),
        ]
    )

def TestTransforms(dim: Tuple[int, int]):
    return T.Compose(
        [
            ToModelInput(),
            T.Resize(dim)
        ]
    )

def get_valid_dim(dim: int, downsample: int = 1) -> int:
    """
    Returns the nearest multiple of 8 that is less than or equal to the
    input dimension. This is required because of the network architecture.

    Args:
        dim (int): The input dimension

    Returns:
        int: The nearest multiple of 4 that is less than or equal to the input
    """
    adjusted_dim = dim // downsample
    valid_dim = (adjusted_dim // 8) * 8
    return valid_dim

def pil_to_plot(img: PILImage):
    return np.array(img)


def tensor_to_plot(img: torch.Tensor):
    return img.permute(1, 2, 0).numpy()
