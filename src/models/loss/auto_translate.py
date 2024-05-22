import torch
import torch.nn.functional as F
from torch import nn
from src.models.loss.mse import MSELoss


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
        paired_loss_fn=MSELoss(),
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
        film,
        digital,
        film_paired,
        digital_paired,
        film_reconstructed,
        digital_reconstructed,
        digital_to_film,
        film_to_digital,
        paired_encoder_representations,
    ):
        """Compute the loss.

        Args:
            film (torch.Tensor): Input tensor representing a batch of film images, shape [B_1, 3, n, n].
            digital (torch.Tensor): Input tensor representing a batch of unpaired images, shape [B_2, 3, n, n].
            film_paired (torch.Tensor): Input tensor representing a batch of paired film images, shape [B_3, 3, n, n].
            digital_paired (torch.Tensor): Input tensor representing a batch of paired digital images, shape [B_3, 3, n, n].
            film_reconstructed (torch.Tensor): Reconstructed film images, shape [B_1, 3, n, n].
            digital_reconstructed (torch.Tensor): Reconstructed digital images, shape [B_2, 3, n, n].
            digital_to_film (torch.Tensor): Transformed digital images, shape [B_1, 3, n, n].
            film_to_digital (torch.Tensor): Transformed film images, shape [B_2, 3, n, n].
            paired_encoder_representations (List[Tuple[torch.Tensor, torch.Tensor]]): Latent space representation of the paired images. List of tuples of tensors (digital_latent, film_latent), each tuple containing the latent space representation of the digital and film images, each shape [B_3, channels, n, n].

        Returns:
            loss: Dict of the loss values.
        """
        # Combine unpaired and paired images
        all_film = torch.cat([film, film_paired], dim=0)
        all_digital = torch.cat([digital, digital_paired], dim=0)

        # Reconstruction loss
        digital_loss = F.mse_loss(digital_reconstructed, all_digital)
        film_loss = F.mse_loss(film_reconstructed, all_film)

        # Encoder loss
        encoder_loss = 0
        for digital_latent, film_latent in paired_encoder_representations:
            encoder_loss += F.mse_loss(digital_latent, film_latent)
        encoder_loss /= len(paired_encoder_representations)

        # Paired transformation loss
        losses = self.paired_loss_fn(digital_paired, film_to_digital)
        paired_digital_loss = losses["loss"]
        paired_film_loss = 0
        if self.do_penalise_film_transformation:
            paired_film_loss = self.paired_loss_fn(film_paired, digital_to_film)

        # Compute individual losses
        reconstruction_loss = digital_loss + film_loss
        paired_reconstruction_loss = paired_digital_loss + paired_film_loss

        # Compute total loss
        loss = (
            self.reconstruction_weight * reconstruction_loss
            + self.encoder_weight * encoder_loss
            + self.paired_reconstruction_weight * paired_reconstruction_loss
        )

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "encoder_loss": encoder_loss,
            "paired_reconstruction_loss": paired_reconstruction_loss,
        }
