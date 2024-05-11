import torch
from torch import nn


class UNet(nn.Module):
    """A U-Net style network for use on its own or as part of the AutoTranslateNet.

    Exposes an encode and decode method to allow for easy use in the AutoTranslateNet.
    """

    def __init__(self, start_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        in_channels = 3
        out_channels = 3

        # Generate feature list
        features = [start_dim * 2**i for i in range(num_layers)]

        # Create the encoder path
        for feature in features[:-1]:
            conv_block = self.conv_block(in_channels, feature)
            self.enc_blocks.append(conv_block)
            in_channels = feature

        # Create the bottleneck layer
        self.bottleneck = self.conv_block(features[-2], features[-1])

        # Create the decoder path
        reversed_features = features[::-1]
        for i in range(len(reversed_features) - 1):
            upconv = nn.ConvTranspose2d(
                reversed_features[i], reversed_features[i + 1], kernel_size=2, stride=2
            )
            conv_block = self.conv_block(2 * reversed_features[i + 1], reversed_features[i + 1])
            self.dec_blocks.append(nn.ModuleList([upconv, conv_block]))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )
        return block

    def encode(self, x):
        skips = []

        # Encoder path
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        return x, skips

    def decode(self, x, skips):
        # Decoder path
        skips = skips[::-1]
        for (upconv, block), skip in zip(self.dec_blocks, skips):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # Output layer
        x = self.final_conv(x)

        return x

    def forward(self, x):
        x, skips = self.encode(x)
        x = self.decode(x, skips)

        return x


if __name__ == "__main__":
    model = UNet(features=[16, 32, 64, 128, 256])
    x = torch.randn(1, 3, 128, 128)
    output = model(x)
