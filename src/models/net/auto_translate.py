import torch
from torch import nn

from src.models.net.unet import UNet


class AutoTranslateNet(nn.Module):
    """
    The AutoTranslateNet is an implementation of the network seen in 'Semi-Supervised Raw-to-Raw mapping' https://arxiv.org/pdf/2106.13883

    The network consists of two auto-encoders, one for the digital domain and one for the film domain.

    There is then a translation network that is trained to map the latent space of the source domain auto-encoder to the latent space of the target domain auto-encoder.
    """

    def __init__(self, start_dim=32, num_layers=3):
        super().__init__()

        # Create auto_encoders
        self.digital_autoencoder = UNet(start_dim=start_dim, num_layers=num_layers)
        self.film_autoencoder = UNet(start_dim=start_dim, num_layers=num_layers)

    def forward(self, digital, film, paired):
        """Forward pass through the model.

        Args:
            digital: Input tensor representing a batch of unpaired images, shape [B_1, 3, n, n].
            film: Input tensor representing a batch of film images, shape [B_2, 3, n, n].
            paired: Input tensor representing a batch of paired images, shape [B_3, 3, n, n, 2].

        Returns:
            digital_reconstructed: Transformed digital images, shape [B_1, 3, n, n].
            film_reconstructed: Transformed film images, shape [B_2, 3, n, n].
            digital_to_film: Transformed digital images from the paired film, shape [B_1, 3, n, n].
            film_to_digital: Transformed film images from the paired digital, shape [B_2, 3, n, n].
            paired_encoder_representation: Latent space representations of the paired images over all encoder layers. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].
        """

        paired_digital_in = paired[:, 0]
        paired_film_in = paired[:, 1]

        # Get all digital/film images by concatenating the digital (B_1, 3, n, n) and paired images (B_3, 2, 3, n, n,)
        all_digital = torch.cat([digital, paired_digital_in], dim=0)
        all_film = torch.cat([film, paired_film_in], dim=0)

        # Auto-encode the digital and film images
        digital_reconstructed = self.digital_autoencoder(all_digital)
        film_reconstructed = self.film_autoencoder(all_film)

        # Encode the paired images
        digital_latent, digital_skips = self.digital_autoencoder.encode(paired_digital_in)
        film_latent, film_skips = self.film_autoencoder.encode(paired_film_in)

        # Transform the paired images by using the decoder of the other domain
        digital_to_film = self.film_autoencoder.decode(digital_latent, digital_skips)
        film_to_digital = self.digital_autoencoder.decode(film_latent, film_skips)

        # Get list of tuples of encoder representations
        paired_encoder_representations = []
        for i in range(len(digital_skips)):
            paired_encoder_representations.append((digital_skips[i], film_skips[i]))

        return (
            digital_reconstructed,
            film_reconstructed,
            digital_to_film,
            film_to_digital,
            paired_encoder_representations,
        )

    def predict(self, digital):
        """Predict the transformed digital image from the paired film image.

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
    paired = torch.rand(4, 2, 3, 256, 256)

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
