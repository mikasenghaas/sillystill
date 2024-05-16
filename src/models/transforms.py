import numpy as np
import torch
import torchvision.transforms.v2 as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class Unnormalize(T.Normalize):
    def __init__(self, mean, std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        super().__init__(-mean / std, 1 / std)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def get_transform_to_model_input():
    return T.Compose(
        [
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=MEAN, std=STD),
        ]
    )


def get_transform_from_model_input():
    return T.Compose(
        [
            Unnormalize(MEAN, STD),  # Unnormalize the image
            T.ToPILImage(),  # Convert Tensor to PIL Image
        ]
    )


def get_transform_augment(augment: int):
    if not augment:
        return T.Lambda(lambda x: x)
    return T.Compose(
        [
            T.RandomHorizontalFlip(p=augment),
            T.RandomVerticalFlip(p=augment),
            T.RandomApply(
                [
                    T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                    T.ColorJitter(brightness=(0.6, 1)),
                ],
                augment,
            ),
        ]
    )
