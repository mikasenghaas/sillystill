import torch
import torchvision.transforms.v2 as T


class Unnormalize(T.Normalize):
    def __init__(self, mean, std):
        mean = torch.Tensor(mean)
        std = torch.Tensor(std)
        super().__init__(-mean / std, 1 / std)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


class IdentityTransform:
    def __call__(self, x):
        return x
