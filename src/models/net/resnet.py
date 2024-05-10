import torch
from torch import nn
from torchvision import models

# Taken and modified from https://github.com/yuzhoucw/230pix2pix/blob/master/model_modules.py


class ResNet(nn.Module):
    """A modified version of the ResNet-50 model that includes both downsample and upsample
    processes to transform an image input to a different size output while maintaining the residual
    connections. It utilizes either batch normalization or instance normalization, based on the
    specified norm type.

    Attributes:
        model (nn.Sequential): The sequential container of layers comprising the ResNet-50 architecture
                               with modifications for upsample and additional norm and activation layers.
    """

    def __init__(self, pretrained=True, channels=3, norm="instancenorm"):
        """Initializes the Resnet50 model with optional pretrained weights and specified
        normalization.

        Args:
            pretrained (bool): If True, loads the pretrained weights for ResNet-50; otherwise, initializes from scratch.
            channels (int): The number of channels in the output image; typically 3 for RGB images.
            norm (str): The type of normalization to use, 'batchnorm' for Batch Normalization and
                        'instancenorm' for Instance Normalization.
        """
        super().__init__()
        if norm == "batchnorm":
            norm_layer = nn.BatchNorm2d
        elif norm == "instancenorm":
            norm_layer = nn.InstanceNorm2d
        else:
            raise Exception("Norm not specified!")

        model = []
        res_original = models.resnet50(pretrained=pretrained)
        model += list(res_original.children())[
            :-2
        ]  # Retaining most of the model except the last two layers

        # Upsample process
        in_channels = 2048
        out_channels = in_channels // 2
        for i in range(5):
            model += [
                nn.ConvTranspose2d(
                    in_channels, out_channels, 3, stride=2, padding=1, output_padding=1
                ),
                norm_layer(out_channels),
                nn.ReLU(inplace=True),
            ]
            in_channels = out_channels
            out_channels = in_channels // 2

        model += [nn.ReflectionPad2d(3), nn.Conv2d(64, channels, 7), nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Defines the forward pass of the model.

        Args:
            input (Tensor): The input tensor of shape (N, C, H, W) where
                            N is the batch size, C is the number of channels,
                            H is the height, and W is the width of the image.

        Returns:
            Tensor: The output tensor of the model after applying the series of layers.
        """
        return self.model(input)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet(pretrained=True, channels=3, norm="instancenorm").to(device)
    input_tensor = torch.randn(1, 3, 64, 64).to(device)
    output = model(input_tensor)
    print("Input tensor size:", input_tensor.size())
    print("Output tensor size:", output.size())
    print("Output tensor:", output)
    print(model.parameters())
