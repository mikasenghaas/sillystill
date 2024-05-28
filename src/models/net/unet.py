from typing import List
import torch
from torch import nn


class UNet(nn.Module):
    """
    A UNet-style convolutional neural network. The network consists of an
    encoder path, a bottleneck layer, and a decoder path.
    """

    def __init__(
        self,
        input_output_channels: int = 3,
        kernel_size: int = 3,
        hidden_channels: List[int] = [64, 128, 256],
        with_noise: bool = False,
    ):
        super().__init__()

        # Save hyperparameters
        self.input_output_channels = input_output_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.reverse_hidden_channels = hidden_channels[::-1]
        self.with_noise = with_noise

        # Create the encoder path
        in_channels = input_output_channels + int(with_noise)
        self.encoder = nn.ModuleList()
        for out_channels in hidden_channels[:-1]:
            conv_block = self.conv_block(
                in_channels, out_channels, kernel_size=kernel_size, padding=self.padding
            )
            self.encoder.append(conv_block)
            in_channels = out_channels

        # Create the bottleneck layer
        self.bottleneck = self.conv_block(
            hidden_channels[-2],
            hidden_channels[-1],
            kernel_size=kernel_size,
            padding=self.padding,
        )

        # Create the decoder path
        reversed_hidden_channels = hidden_channels[::-1]
        self.decoder = nn.ModuleList()
        for i in range(len(reversed_hidden_channels) - 1):
            upconv = nn.ConvTranspose2d(
                reversed_hidden_channels[i],
                reversed_hidden_channels[i + 1],
                kernel_size=2,
                stride=2,
            )
            conv_block = self.conv_block(
                2 * reversed_hidden_channels[i + 1],
                reversed_hidden_channels[i + 1],
                kernel_size=kernel_size,
                padding=self.padding,
            )
            self.decoder.append(nn.ModuleList([upconv, conv_block]))

        # Final output layer
        self.final_conv = nn.Conv2d(
            hidden_channels[0], 3, kernel_size=self.kernel_size, padding=self.padding
        )
        # self.final_final_conv = nn.Conv2d(6, 3, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(
        self, in_dim: int, out_dim: int, kernel_size: int = 3, padding: int = 1
    ):
        # [B, in_dim, H, W] -> [B, out_dim, H, W]
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

    def encode(self, x):
        skips = []
        # skips.append(x)

        # Encoder path
        for block in self.encoder:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        return x, skips

    def decode(self, x, skips):
        # Decoder path
        skips = skips[::-1]
        for (upconv, block), skip in zip(self.decoder, skips):
            x = upconv(x)  # [B, X, 128, 128]
            x = torch.cat([x, skip], dim=1)  # [B, 2X, 128, 128]
            x = block(x)  #

        # Final output layer
        x = self.final_conv(x)

        # RGB skip
        # x = torch.cat([x, skips[-1]], dim=1)

        # Final conv
        # x = self.final_final_conv(x)

        return x

    # num_channels: 3, 64, 128, 64, 3
    # num_channels: 3, 64, 128, 64, 3

    def forward(self, x):
        # Add noise if specified
        if self.with_noise:
            B, _, H, W = x.shape
            noise = torch.rand((B, 1, H, W)).to(x.device)
            x = torch.cat([x, noise], dim=1)

        x, skips = self.encode(x)
        x = self.decode(x, skips)

        return x
