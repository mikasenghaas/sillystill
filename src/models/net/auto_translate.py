from typing import List, Tuple
import torch
from torch import nn

from src.models.net.unet import UNet


class AutoTranslateNet(nn.Module):
    """
    The AutoTranslateNet is an implementation of the network seen in
    'Semi-Supervised Raw-to-Raw mapping' https://arxiv.org/pdf/2106.13883

    The network consists of two auto-encoders, one for the digital domain and
    one for the film domain.

    There is then a translation network that is trained to map the latent space
    of the source domain auto-encoder to the latent space of the target domain
    auto-encoder.
    """

    def __init__(
        self,
        input_output_channels: int = 3,
        hidden_channels: List[int] = [64, 128, 256],
    ):
        super().__init__()

        # Initialise digital and film auto-encoders
        self.digital_autoencoder = UNet(input_output_channels, hidden_channels)
        self.film_autoencoder = UNet(input_output_channels, hidden_channels)

    def forward(
        self,
        film: torch.Tensor,
        digital: torch.Tensor,
        film_paired: torch.Tensor,
        digital_paired: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[Tuple[torch.Tensor, torch.Tensor]],
    ]:
        """
        Forward pass through the model.

        Args:
            digital: Input tensor representing a batch of unpaired images, shape [B_1, 3, n, n].
            film: Input tensor representing a batch of film images, shape [B_2, 3, n, n].
            paired: Input tensor representing a batch of paired images, shape [B_3, 3, n, n, 2].

        Returns:
            film_reconstructed: Transformed film images, shape [B_2, 3, n, n].
            digital_reconstructed: Transformed digital images, shape [B_1, 3, n, n].
            film_to_digital: Transformed film images from the paired digital, shape [B_2, 3, n, n].
            digital_to_film: Transformed digital images from the paired film, shape [B_1, 3, n, n].
            paired_encoder_representation: Latent space representations of the paired images over all encoder layers. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].
        """
        # Get all digital/film images
        all_film = torch.cat([film, film_paired], dim=0)
        all_digital = torch.cat([digital, digital_paired], dim=0)

        # Auto-encode the digital and film images
        digital_reconstructed = self.digital_autoencoder(all_digital)
        film_reconstructed = self.film_autoencoder(all_film)

        # Encode the paired images separately
        digital_latent, digital_skips = self.digital_autoencoder.encode(digital_paired)
        film_latent, film_skips = self.film_autoencoder.encode(film_paired)

        # Transform the paired images by using the decoder of the other domain
        digital_to_film = self.film_autoencoder.decode(digital_latent, digital_skips)
        film_to_digital = self.digital_autoencoder.decode(film_latent, film_skips)

        # Get list of tuples of encoder representations
        paired_encoder_representations = []
        for i in range(len(digital_skips)):
            paired_encoder_representations.append((digital_skips[i], film_skips[i]))

        return (
            film_reconstructed,
            digital_reconstructed,
            film_to_digital,
            digital_to_film,
            paired_encoder_representations,
        )

    def predict(self, digital: torch.Tensor) -> torch.Tensor:
        """
        Predict the transformed digital image from the paired film image.

        Args:
            digital: Input tensor representing a batch of unpaired images, shape [B, 3, n, n].

        Returns:
            film: Transformed film images, shape [B, 3, n, n].
        """

        encoded_digital, skips = self.digital_autoencoder.encode(digital)

        return self.film_autoencoder.decode(encoded_digital, skips)


if __name__ == "__main__":
    # Test the AutoTranslateNet
    digital = torch.rand(4, 3, 256, 256)
    film = torch.rand(4, 3, 256, 256)
    paired = (torch.rand(4, 3, 256, 256), torch.rand(4, 3, 256, 256))

    model = AutoTranslateNet()
    (
        digital_reconstructed,
        film_reconstructed,
        digital_to_film,
        film_to_digital,
        paired_encoder_representations,
    ) = model(digital, film, paired)
    print(
        digital_reconstructed.shape,
        film_reconstructed.shape,
        digital_to_film.shape,
        film_to_digital.shape,
    )
    print(
        len(paired_encoder_representations),
        paired_encoder_representations[0][0].shape,
        paired_encoder_representations[0][1].shape,
    )
