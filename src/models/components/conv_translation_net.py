import torch
from torch import nn
from torch.nn import functional as F


class ConvTranslationNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[32, 64, 128]):
        super(ConvTranslationNet, self).__init__()
        self.enc_blocks = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
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
            upconv = nn.ConvTranspose2d(reversed_features[i], reversed_features[i + 1], kernel_size=2, stride=2)
            conv_block = self.conv_block(2*reversed_features[i + 1], reversed_features[i + 1])
            self.dec_blocks.append(nn.ModuleList([upconv, conv_block]))

        # Final output layer
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, padding=1):
        block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, x):
        skips = []
        
        # Encoder path
        for block in self.enc_blocks:
            x = block(x)
            skips.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        skips = skips[::-1]
        for (upconv, block), skip in zip(self.dec_blocks, skips):
            x = upconv(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x)

        # Output layer
        x = self.final_conv(x)
 
        return x
