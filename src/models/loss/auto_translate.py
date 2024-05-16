import torch
import torch.nn.functional as F
from torch import nn
from src.models.loss.simple_combined import SimpleCombinedLoss


class AutoTranslateLoss(nn.Module):
    """Returns a loss value for the auto-translate model.

    Consists of three components:
    - Reconstruction loss for the digital and film images
    - Encoder representations loss for the paired images
    - Overall reconstruction/transformation loss for paired images

    The result is a weighted sum of the three components.
    """

    def __init__(
        self,
        reconstruction_weight=1.0,
        encoder_weight=1.0,
        paired_reconstruction_weight=1.0,
        paired_loss_fn=SimpleCombinedLoss(),
        do_penalise_film_transformation=False,
    ):
        """
        Instantiate the loss function.

        Args:
            reconstruction_weight (float): Weight for the reconstruction loss.
            encoder_weight (float): Weight for the encoder loss.
            paired_reconstruction_weight (float): Weight for the paired reconstruction loss.
            paired_loss_fn (nn.Module): Loss function for the paired images.
            do_penalise_film_transformation (bool): Whether to penalise the transformation of the film->digital images.
        """
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.encoder_weight = encoder_weight
        self.paired_reconstruction_weight = paired_reconstruction_weight
        self.paired_loss_fn = paired_loss_fn
        self.do_penalise_film_transformation = do_penalise_film_transformation

    def forward(
        self,
        digital_in,
        film_in,
        paired_in,
        digital_reconstructed,
        film_reconstructed,
        digital_transformed,
        film_transformed,
        paired_encoder_representations,
    ):
        """Compute the loss.

        Args:
            digital_in: Input tensor representing a batch of unpaired images, shape [B_1, 3, n, n].
            film_in: Input tensor representing a batch of film images, shape [B_2, 3, n, n].
            paired_in: Input tensor representing a batch of paired images, shape [B_3, 3, n, n, 2].
            digital_reconstructed: Reconstructed digital images, shape [B_1, 3, n, n].
            film_reconstructed: Reconstructed film images, shape [B_2, 3, n, n].
            digital_out: Transformed digital images, shape [B_1, 3, n, n].
            film_out: Transformed film images, shape [B_2, 3, n, n].
            paired_encoder_representations: Latent space representation of the paired images. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].

        Returns:
            loss: Dict of the loss values.
        """

        paired_digital_in = paired_in[0]
        paired_film_in = paired_in[1]

        all_digital_in = torch.cat([digital_in, paired_digital_in], dim=0)
        all_film_in = torch.cat([film_in, paired_film_in], dim=0)

        # Reconstruction loss
        digital_loss = F.mse_loss(digital_reconstructed, all_digital_in)
        film_loss = F.mse_loss(film_reconstructed, all_film_in)

        # Encoder loss
        encoder_loss = 0
        for digital_latent, film_latent in paired_encoder_representations:
            encoder_loss += F.mse_loss(digital_latent, film_latent)
        encoder_loss /= len(paired_encoder_representations)

        # Paired transformation loss
        paired_digital_loss = self.paired_loss_fn(
            paired_digital_in, digital_transformed
        )["loss"]
        paired_film_loss = 0
        if self.do_penalise_film_transformation:
            paired_film_loss_dict = self.paired_loss_fn(
                paired_film_in, film_transformed
            )["loss"]

        # Compute total loss
        loss = (
            self.reconstruction_weight * (digital_loss + film_loss)
            + self.encoder_weight * encoder_loss
            + self.paired_reconstruction_weight
            * (paired_digital_loss + paired_film_loss)
        )
        return {
            "loss": loss,
            "reconstruction_loss": digital_loss + film_loss,
            "encoder_loss": encoder_loss,
            "paired_reconstruction_loss": paired_digital_loss + paired_film_loss,
        }
