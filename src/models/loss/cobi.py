import torch
import torch.nn as nn
from torchvision.models import vgg19_bn
from torchvision.models.vgg import VGG19_BN_Weights
from src.models.loss.base import BaseLoss

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)


class CoBiLoss(BaseLoss):
    """
    A PyTorch implementation of the Contextual Bilateral (CoBi) Loss, as described in "Zoom to Learn, Learn to Zoom" (https://arxiv.org/pdf/1905.05169). This implementation has been adapted for PyTorch from the original code found at https://github.com/ceciliavision/zoom-learn-zoom.

    The CoBi loss builds upon the Contextual (CX) Loss introduced in "The Contextual Loss for Image Transformation with Non-Aligned Data" (https://arxiv.org/pdf/1803.02077), which is designed to handle non-aligned image data.

    The CX Loss is computed as the sum of the minimum distances between feature maps of two images, defined as:
        CX(P, Q) = 1/N * sum(min(Sim(p_i, q_i) for each p_i in P and q_i in Q))
    where 'Sim' represents a similarity function based on cosine similarity.

    The CoBi Loss extends the CX Loss by incorporating an L2 spatial distance component, making it more robust to spatial discrepancies:
        CoBi(P, Q) = 1/N * sum(min((1 - w_d) * Sim(p_i, q_i) + w_d * D(p_i, q_i) for each p_i in P and q_i in Q))
    where 'w_d' is a weighting factor for the spatial distance and 'D' represents the L2 distance between the features.

    The final CoBi Loss for a model is a weighted combination of CoBi computed on RGB patches and VGG19 features:
        CoBiLoss = alpha * CoBi(RGB, n) + (1 - alpha) * CoBi(VGG19)
    where 'alpha' is a weight balancing the two terms, 'n' is the size of RGB patches used, and the VGG19 features considered are from layers 'conv1_2', 'conv2_2', and 'conv3_2'.
    """

    def name(self) -> str:
        return "cobi_loss"

    def __init__(
        self,
        alpha=0.5,
        patch_size=64,
        ws=0.1,
        epsilon=1e-5,
        bandwidth=0.1,
        verbose=False,
    ):
        """Initialize a `CoBiLoss` module with optional verbosity.

        Args:
            alpha (float, optional): Weighting factor between RGB and VGG feature losses, higher means more importance of RGB features. Defaults to `0.5`.
            patch_size (int, optional): Size of the patches extracted for RGB features. Defaults to `15`.
            ws (float, optional): Weighting factor for the spatial component in the loss. Defaults to `0.1`.
            epsilon (float, optional): Small value to avoid division by zero in normalization. Defaults to `1e-5`.
            bandwidth (float, optional): Bandwidth used in the softmax transformation of distances to similarities. Defaults to `0.1`.
            verbose (bool, optional): Enables printing of progress messages during computation. Defaults to False.
        """
        super().__init__()
        self.alpha = alpha
        self.patch_size = patch_size
        self.ws = ws
        self.epsilon = epsilon
        self.bandwidth = bandwidth
        self.verbose = verbose
        self.vgg = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT).to(device).features
        self.feature_layers = {"3": "conv1_2", "8": "conv2_2", "17": "conv3_2"}
        self.feature_weights = {"conv1_2": 0.4, "conv2_2": 0.4, "conv3_2": 0.2}

        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False

        if self.verbose:
            print("Initialized CoBiLoss with the following settings:")
            print(
                f"  alpha: {self.alpha}, patch_size: {self.patch_size}, ws: {self.ws}"
            )
            print(f"  feature_layers used: {self.feature_layers.keys()}")

    def forward(self, x, y):
        """Forward pass of the loss calculation.

        Args:
            x (torch.Tensor): The input tensor containing the batch of source images.
            y (torch.Tensor): The target tensor containing the batch of target images.

        Returns:
            torch.Tensor: The calculated CoBi Loss.
        """
        if self.verbose:
            print("Starting loss computation...")

        features_x = self.extract_features(x)
        features_y = self.extract_features(y)

        vgg_loss = 0
        for layer in self.feature_layers.keys():
            layer_loss = self.compute_cobi_loss(features_x[layer], features_y[layer])
            vgg_loss += self.feature_weights[self.feature_layers[layer]] * layer_loss
            if self.verbose:
                print(f"Loss for layer {layer}: {layer_loss.item()}")

        # Compute RGB patch-based loss
        rgb_loss = self.compute_cobi_loss_with_patches(x, y)
        if self.verbose:
            print(f"RGB patch-based loss: {rgb_loss.item()}")

        loss = self.alpha * rgb_loss + (1 - self.alpha) * vgg_loss

        if self.verbose:
            print(f"Final computed CoBiLoss: {loss.item()}")

        return {"loss": loss, "vgg_loss": vgg_loss, "rgb_loss": rgb_loss}

    def extract_features(self, img):
        """Extracts features from the input image using the VGG19 model.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            dict: A dictionary containing the extracted features from the VGG19 model.
        """
        if self.verbose:
            print("Extracting features using VGG19...")
        features = {}
        x = img
        for name, module in self.vgg._modules.items():
            x = module(x)
            if name in self.feature_layers:
                features[name] = x
        return features

    def compute_cobi_loss_with_patches(self, x, y):
        """Compute the CoBi loss using RGB patches.

        Args:
            x (torch.Tensor): The tensor containing source image patches.
            y (torch.Tensor): The tensor containing target image patches.

        Returns:
            torch.Tensor: The computed loss for RGB patches.
        """
        patches_x = x.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches_y = y.unfold(2, self.patch_size, self.patch_size).unfold(
            3, self.patch_size, self.patch_size
        )
        patches_x = patches_x.contiguous().view(
            x.size(0), x.size(1), -1, self.patch_size * self.patch_size
        )
        patches_y = patches_y.contiguous().view(
            y.size(0), y.size(1), -1, self.patch_size * self.patch_size
        )
        return self.compute_cobi_loss(patches_x, patches_y)

    def compute_cobi_loss(self, p, q):
        """Compute the contextual bilateral loss between two sets of features.

        Args:
            p (torch.Tensor): Feature set from source images.
            q (torch.Tensor): Feature set from target images.

        Returns:
            torch.Tensor: The computed CoBi loss.
        """
        if self.verbose:
            print("Calculating CoBi loss between feature sets...")

        n, c, h, w = p.size()
        p_flat = p.view(n, c, h * w).permute(0, 2, 1)  # Shape (n, h*w, c)
        q_flat = q.view(n, c, h * w).permute(0, 2, 1)  # Shape (n, h*w, c)

        # Compute cosine distances
        cos_similarity = torch.bmm(p_flat, q_flat.transpose(1, 2)) / (
            p_flat.norm(dim=2, keepdim=True)
            * q_flat.norm(dim=2, keepdim=True).transpose(1, 2)
            + self.epsilon
        )
        cos_distance = 1 - cos_similarity  # Shape (n, h*w, h*w)

        # Normalize distances
        min_cos_dist = torch.min(cos_distance, dim=2, keepdim=True)[0]
        normalized_dist = cos_distance / (min_cos_dist + self.epsilon)

        # Shift from distances to similarities
        wij = torch.exp((1 - normalized_dist) / self.bandwidth)
        cx_ij = wij / torch.sum(wij, dim=2, keepdim=True)

        # And back to distances for use in CoBi (which uses distances) as oppose to CX
        cobi_ij = 1 - cx_ij

        # Calculate spatial distances
        spatial_dist = self.calculate_spatial_distances(n, h, w, p.device)

        # Combine contextual and spatial distances
        combined_measure = (1 - self.ws) * cobi_ij + self.ws * spatial_dist

        # Compute the minimum combined measure for each feature vector in p
        min_combined_measure, _ = torch.min(
            combined_measure, dim=2
        )  # Min across all j for each i

        # Average these minimum values across all vectors in the batch
        score = torch.mean(min_combined_measure)

        # Take negative log to get the loss
        loss = -torch.log(1 - score)

        return loss

    def calculate_spatial_distances(self, n, h, w, device):
        """Calculate spatial distances for a grid of features.

        Args:
            n (int): Batch size of the input tensors.
            h (int): Height of the feature grid.
            w (int): Width of the feature grid.
            device (torch.device): The device on which calculations are performed.

        Returns:
            torch.Tensor: A tensor containing the Euclidean distances between each pair
            of points in the grid. The tensor will have shape (1, h*w, h*w), where each
            entry (i, j) represents the distance between points i and j.
        """
        if self.verbose:
            print("Calculating spatial distances...")

        # Generate spatial coordinates
        coords = (
            torch.stack(
                torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij"), dim=-1
            )
            .to(device)
            .float()
        )

        # Flatten coordinates and expand to match batch size
        coords = (
            coords.view(h * w, 2).unsqueeze(0).expand(n, h * w, 2)
        )  # Expand along batch size

        # Compute pairwise Euclidean distances
        spatial_dist = torch.norm(coords.unsqueeze(2) - coords.unsqueeze(1), dim=3)

        return spatial_dist
