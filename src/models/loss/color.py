import torch
import torch.nn.functional as F
import torchvision.transforms as T
from src.models.loss.base import BaseLoss


class ColorLoss(BaseLoss):
    def name(self) -> str:
        return "color_loss"

    def __init__(self, kernel_size=7, sigma=3.0):
        super(ColorLoss, self).__init__()
        self.gaussian_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    def forward(self, enhanced_image, target_image):
        blurred_enhanced = self.gaussian_blur(enhanced_image)
        blurred_target = self.gaussian_blur(target_image)
        loss = F.mse_loss(blurred_enhanced, blurred_target)
        return loss


# Example usage:
if __name__ == "__main__":
    # Assuming enhanced_image and target_image are your input tensors of shape (N, C, H, W)
    enhanced_image = torch.randn(1, 3, 256, 256)
    target_image = torch.randn(1, 3, 256, 256)

    color_loss_fn = ColorLoss(kernel_size=7, sigma=3.0)
    loss = color_loss_fn(enhanced_image, target_image)
    print("Color loss on random noise:", loss.item())

    # Loss on the same image should be 0
    loss = color_loss_fn(enhanced_image, enhanced_image)
    print("Color loss on the same image:", loss.item())

    # Loss on images with ones in RGB channels should be 0
    ones_image = torch.ones_like(enhanced_image)
    loss = color_loss_fn(enhanced_image, ones_image)
    print("Color loss on images with ones in RGB channels:", loss.item())
