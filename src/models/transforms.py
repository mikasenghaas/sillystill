from typing import Tuple
import numpy as np
import torch
import torch.nn as nn

import torchvision.transforms.v2 as T
from PIL.Image import Image as PILImage


def ToModelInput():
    """
    Transform to convert np.array, PIL image or torch.Tensor to a tensor
    that can be fed into the model (C, H, W) w/ float32 dtype.
    """
    return T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])


def FromModelInput():
    """
    Converts back from model input to a PIL image.
    """
    return T.ToPILImage()


def ToBatchedTensor():
    """
    Used in the dataloader to convert a list of tensors to a batched tensor.
    """
    return T.Compose([ToModelInput(), T.Resize((2433, 3637))])


def Augment(augment: float):
    """
    Applies transform to augment the data. Only works when the input is in
    the "model input" format.
    """
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
    """
    Transform used during training (data is assumed to already be in the
    model input format) because it comes from the dataloader.
    """
    return T.Compose(
        [
            Augment(augment),
            T.RandomCrop(patch_size),
        ]
    )


def TestTransforms(dim: Tuple[int, int]):
    """
    Transform used during inference (data might not be in the model input
    format yet).
    """
    return T.Compose([ToModelInput(), T.Resize(dim)])


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


def to_infer(img, downsample=2, device="cpu"):
    height = get_valid_dim(img.size[1], downsample=downsample)
    width = get_valid_dim(img.size[0], downsample=downsample)
    img_transform = TestTransforms(dim=(height, width))
    img = img_transform(img).unsqueeze(0).clamp(0 + 1e-5, 1 - 1e-5)
    return img.to(device)


def pil_to_plot(img: PILImage):
    return np.array(img)


def tensor_to_plot(img: torch.Tensor):
    return img.permute(1, 2, 0).numpy()
