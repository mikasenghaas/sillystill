from typing import List
import torch
from torch import nn


class FFNet(nn.Module):
    """A simple FFNN encoder-decoder network."""

    def __init__(
        self,
        input_output_size: int = 28,
        hidden_dims: List[int] = [16, 8],
    ) -> None:
        """
        Initialize a `FFFN` module.

        Args:
            input_output_size (int): The size of the input and output tensors.
            hidden_layers (List[int]): A list of hidden layer sizes.
        """
        super().__init__()

        self.input_output_size = input_output_size
        self.hidden_dims = hidden_dims
        self.reverse_hidden_dims = hidden_dims[::-1]

        # Dimensions
        self.encoder_dims = list(
            zip([self.input_output_size] + self.hidden_dims[:-1], self.hidden_dims)
        )
        self.decoder_dims = list(
            zip(
                self.reverse_hidden_dims,
                self.reverse_hidden_dims[1:] + [self.input_output_size],
            )
        )

        # Encoder
        self.encoder = nn.ModuleList(
            [
                self._linear_block(in_dim, out_dim)
                for in_dim, out_dim in self.encoder_dims
            ]
        )

        self.decoder = nn.ModuleList(
            [
                self._linear_block(in_dim, out_dim)
                for in_dim, out_dim in self.decoder_dims
            ]
        )

    def _linear_block(self, in_dim: int, out_dim: int) -> nn.Module:
        """
        Create a linear block with batch normalization and ReLU activation.
        """
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform a single forward pass through the network. The input tensor
        is passed through the encoder and decoder and is assumed to be 4D with
        shape (batch_size, channels, width, height).  First, the input tensor is
        flattened to a 2D tensor, then passed through the encoder and decoder.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.
        """
        # Flatten
        B, C, H, W = x.size()
        x = x.view(B, -1)

        # Encoder
        for block in self.encoder:
            x = block(x)

        # Decoder
        for block in self.decoder:
            x = block(x)

        # Reshape
        x = x.view(B, C, H, W)

        return x
