from typing import List
import torch
from torch import nn


class ConvNet(nn.Module):
    """
    A simple convolutional encoder-decoder neural network (without skip connections)
    """

    def __init__(
        self,
        input_output_channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256],
    ):
        super().__init__()

        # Save hyperparameters
        self.input_output_channels = input_output_channels
        self.hidden_channels = hidden_channels
        self.reverse_hidden_channels = hidden_channels[::-1]

        # Dimensions
        self.encoder_channels = list(
            zip(
                [self.input_output_channels] + self.hidden_channels[:-1],
                self.hidden_channels,
            )
        )
        self.decoder_channels = list(
            zip(
                self.reverse_hidden_channels,
                self.reverse_hidden_channels[1:] + [self.input_output_channels],
            )
        )

        # Create the encoder path
        self.encoder = nn.ModuleList(
            [
                self.conv_block(in_channels, out_channels)
                for in_channels, out_channels in self.encoder_channels
            ]
        )
        self.decoder = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            in_channels, out_channels, kernel_size=2, stride=2
                        ),
                        self.conv_block(out_channels, out_channels),
                    ]
                )
                for in_channels, out_channels in self.decoder_channels
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def conv_block(self, in_dim: int, out_dim: int, kernel: int = 3, pad: int = 1):
        return nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=kernel, padding=pad),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=kernel, padding=pad),
            nn.ReLU(),
        )

    def encode(self, x):
        # Encoder path
        for block in self.encoder:
            x = block(x)
            x = self.pool(x)

        return x

    def decode(self, x):
        # Decoder path
        for upconv, block in self.decoder:
            x = upconv(x)
            x = block(x)

        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)

        return x
