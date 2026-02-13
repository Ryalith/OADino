from typing import Sequence

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torchvision import transforms
from transformers import PreTrainedModel


class VAE(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, x):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def reparametrization(self, mean, var):
        raise NotImplementedError

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


class ConvVAE16(VAE):
    """Variationnal autoencoders encoding dino patches (14*14 a priori)
    The paper uses 64*64 patches resized from 14*14 but not sure why they do it
    This Small 16*16 VAE is used instead for faster testing
    """

    def __init__(self, latent_base=64, prior_dim=32):
        """Latent dim will be latent_base*16"""
        super().__init__()

        self.input_size = 16
        self.latent_base = latent_base
        self.latent_dim = latent_base * 16
        self.prior_dim = prior_dim

        # Encoder: 16x16 -> 8x8 -> 4x4 -> 2x2 -> flatten
        self.encoder = nn.Sequential(
            # Input: (batch, 3, 16, 16)
            nn.Conv2d(
                3, latent_base, kernel_size=3, stride=2, padding=1
            ),  # -> (batch, lb, 8, 8)
            nn.BatchNorm2d(latent_base),
            nn.ReLU(),
            nn.Conv2d(
                latent_base, latent_base * 2, kernel_size=3, stride=2, padding=1
            ),  # -> (batch, 2lb, 4, 4)
            nn.BatchNorm2d(latent_base * 2),
            nn.ReLU(),
            nn.Conv2d(
                latent_base * 2, latent_base * 4, kernel_size=3, stride=2, padding=1
            ),  # -> (batch, 4lb, 2, 2)
            nn.BatchNorm2d(latent_base * 4),
            nn.ReLU(),
            nn.Flatten(),  # -> (batch, 4lb*2*2 = 16lb)
        )

        # Mean and logvar layers for latent space
        self.mean_layer = nn.Sequential(nn.Linear(latent_base * 4 * 2 * 2, prior_dim))
        self.logvar_layer = nn.Sequential(nn.Linear(latent_base * 4 * 2 * 2, prior_dim))

        # Decoder: latent -> 2x2 -> 4x4 -> 8x8 -> 16x16
        self.decoder = nn.Sequential(
            nn.Linear(prior_dim, 16 * latent_base),
            nn.ReLU(inplace=True),
            nn.Unflatten(1, (latent_base * 4, 2, 2)),  # -> (batch, 4lb, 2, 2)
            nn.ConvTranspose2d(
                latent_base * 4, latent_base * 2, kernel_size=4, stride=2, padding=1
            ),  # -> (batch, 2lb, 4, 4)
            nn.BatchNorm2d(latent_base * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                latent_base * 2, latent_base, kernel_size=4, stride=2, padding=1
            ),  # -> (batch, lb, 8, 8)
            nn.BatchNorm2d(latent_base),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                latent_base, 3, kernel_size=4, stride=2, padding=1
            ),  # -> (batch, 3, 16, 16)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


class ConvVAE64(VAE):
    """Variationnal autoencoders encoding dino patches (14*14 a priori)
    The paper uses 64*64 patches resized from 14*14 but not sure why they do it
    """

    def __init__(self, latent_base=32, prior_dim=32):
        """Latent dim will be latent_base*32"""
        super().__init__()

        self.input_size = 64
        self.latent_base = latent_base
        self.latent_dim = latent_base * 32
        self.prior_dim = prior_dim

        # Encoder: 64x64 -> 31x31 -> 14x14 -> 6x6 -> 2x2 -> flatten
        self.encoder = nn.Sequential(
            # Input: (batch, 3, 64, 64)
            nn.Conv2d(
                3, latent_base, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, lb, 31, 31)
            nn.ReLU(),
            nn.Conv2d(
                latent_base, latent_base * 2, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 2lb, 14, 14)
            nn.ReLU(),
            nn.Conv2d(
                latent_base * 2, latent_base * 4, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 4lb, 6, 6)
            nn.ReLU(),
            nn.Conv2d(
                latent_base * 4, latent_base * 8, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 8lb, 2, 2)
            nn.ReLU(),
            nn.Flatten(),  # -> (batch, 8lb*4*4)
        )

        # Mean and logvar layers for latent space
        self.mean_layer = nn.Sequential(nn.Linear(latent_base * 8 * 2 * 2, prior_dim))
        self.logvar_layer = nn.Sequential(nn.Linear(latent_base * 8 * 2 * 2, prior_dim))

        # Decoder: latent -> 2x2 -> 6x6 -> 14x14 -> 31x31 -> 64x64
        self.decoder = nn.Sequential(
            nn.Linear(prior_dim, latent_base * 8 * 2 * 2),
            nn.ReLU(),
            nn.Unflatten(1, (latent_base * 8, 2, 2)),
            nn.ConvTranspose2d(
                latent_base * 8, latent_base * 4, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 4lb, 6, 6)
            nn.ReLU(),
            nn.ConvTranspose2d(
                latent_base * 4, latent_base * 2, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 2lb, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(
                latent_base * 2, latent_base, kernel_size=5, stride=2, padding=0
            ),  # -> (batch, 1lb, 31, 31)
            nn.ReLU(),
            nn.ConvTranspose2d(
                latent_base, 3, kernel_size=4, stride=2, padding=0
            ),  # -> (batch, 1lb, 64, 64)
        )

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar


class OADinoPreProcessor(nn.Module):
    def __init__(self, backbone: nn.Module):
        """The Processor now expects the backbone to be a model loaded from torch.hub
        For example:
            dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        """
        super().__init__()
        self.backbone = backbone

    @staticmethod
    def get_median_mask(flat_patches, pca_q=None, pca_niter=2):
        device = flat_patches.device
        fg_pca = PCA(n_components=1)
        patch_pca = fg_pca.fit_transform(flat_patches.detach().cpu())
        mask = patch_pca > np.median(patch_pca)
        mask = torch.from_numpy(mask)
        return ~mask.to(device).squeeze(-1)

    @staticmethod
    def get_mean_mask(flat_patches, pca_q=None, pca_niter=2):
        flattened_patches_centered = flat_patches - flat_patches.mean(dim=0)
        U, S, V = torch.pca_lowrank(flattened_patches_centered, pca_q, False, pca_niter)
        patch_pca = torch.matmul(flattened_patches_centered, V[:, 0])
        return patch_pca > patch_pca.mean()

    def segment_images(self, images, pca_q=None, pca_niter=2, add_opt_outputs=False):
        """Segment images to isolate objects based on pca and DINOV2 features
        images are expected to come in batches of shape (batch_size, 3, 224, 224)
        images are expected to be scaled in 0, 1 then normalized with a transform like
        T.Normalize([0.5], [0.5])
        """
        with torch.no_grad():
            outputs = self.backbone.forward_features(images)

            # (batch_size, 1+n_patches, hidden_size)
            backbone_patches = outputs["x_norm_patchtokens"]

            batch_size, n_patches, hidden_size = backbone_patches.shape
            _, _, height, width = images.shape
            patch_size = 14  # DINOv2 patch size 14

            # 1. Foreground separation of patches

            flattened_patches = backbone_patches.reshape((-1, hidden_size))
            rough_mask = self.get_median_mask(flattened_patches, pca_q, pca_niter)
            # 2. Refining object consistency

            masked_patches = flattened_patches[rough_mask]
            refined_mask = self.get_median_mask(masked_patches, pca_q, pca_niter)
            # refined_mask = self.get_mean_mask(masked_patches, pca_q, pca_niter)

            # 3. Remapping to image space

            mask = rough_mask.clone()
            mask[rough_mask] = refined_mask

            pixel_mask = mask.reshape(
                (batch_size, 1, height // patch_size, 1, width // patch_size, 1)
            )
            pixel_mask = pixel_mask.expand((-1, 3, -1, patch_size, -1, patch_size))
            pixel_mask = pixel_mask.reshape((batch_size, 3, height, width))
            pixel_mask = pixel_mask.float()
            pixel_mask = (
                1 - pixel_mask
            )  # true means masked, but true is 1. and we want 0. to mask
            segmented_images = -1 * pixel_mask + images * (1 - pixel_mask)

            if add_opt_outputs:
                return (
                    segmented_images,
                    mask.reshape(batch_size, n_patches),
                    backbone_patches,
                    rough_mask.reshape(batch_size, n_patches),
                )
            else:
                return segmented_images, mask.reshape(batch_size, n_patches)

    def create_patches(self, images):
        # sizes listed in the paper are 518 and 14
        # in practice not sure this is actually the case ?
        batch_size, channels, height, width = images.shape

        patch_size = 14  # DINOv2 patch size 14

        # Calculate number of patches
        num_patches_h = height // patch_size  # 518 // 14 = 37
        num_patches_w = width // patch_size  # 518 // 14 = 37

        # Reshape to extract patches
        # (batch, 3, 224, 224) -> (batch, 3, 37, 14, 37, 14)
        patches = images.reshape(
            batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size
        )

        # Rearrange dimensions: (batch, 3, 37, 14, 37, 14) -> (batch, 37, 37, 3, 14, 14)
        patches = patches.permute(0, 2, 4, 1, 3, 5)

        # Flatten patches: (batch, 37, 37, 3, 14, 14) -> (batch, 37*37, 3, 14, 14)
        patches = patches.reshape(
            batch_size, num_patches_h * num_patches_w, 3, patch_size, patch_size
        )

        return patches

    def get_global_features_and_patches(self, images, pca_q=None, pca_niter=2):
        """Create global features, divide image into patches, and mask non object related patches
        returns:
        global_features:    (batch_size, feature_size_backbone)
        object_patches:     (batch_size, n_patches, n_channels, patch_size, patch_size)
        mask:               (batch_size, n_patches)
        """
        segmented_images, mask = self.segment_images(images, pca_q, pca_niter)
        object_patches = self.create_patches(segmented_images)
        global_features = self.backbone.forward_features(segmented_images)[
            "x_norm_clstoken"
        ]

        return global_features, object_patches, mask


# First implementation of the Oh-A-Dino model
# The idea is to, during training:
# - call segment_image to get segmented inputs for the rest of the model and the mask
# - call create_patches to create a set of patches
# - call encode_decode_patches to get the latent and reconstructed patches
# - compute the paper's vae encoder loss and backprop
class OADinoModel(nn.Module):
    def __init__(self, vae: VAE):
        super().__init__()
        self.vae = vae
        self.transform = transforms.Resize(self.vae.input_size)

    def get_features(
        self, global_features, object_patches, mask
    ) -> Sequence[torch.tensor]:
        """Create OADino features from global_features (dino on masked input) and patches
        Notes:
        - Some reviews mention using other segmentation methods (like using UNets), we could add implementations ?
        """
        masked_object_patches = object_patches[mask]
        masked_object_patches = self.transform(masked_object_patches)
        object_features, _ = self.vae.encode(masked_object_patches)

        oadino_features = []
        feature_idx = 0  # Track position in flattened object_features

        for idx in range(mask.shape[0]):
            # Count non-masked patches for this batch element
            n_not_masked = mask[idx].sum().item()

            if n_not_masked > 0:
                # Get global feature for this batch element: shape (ng,)
                global_feature = global_features[idx]  # shape: (ng,)

                # Get corresponding local features: shape (n_not_masked, nl)
                local_features = object_features[
                    feature_idx : feature_idx + n_not_masked
                ]  # shape: (n_not_masked, nl)

                # Expand global feature to match: shape (n_not_masked, ng)
                global_expanded = global_feature.unsqueeze(0).expand(n_not_masked, -1)

                # Concatenate: shape (n_not_masked, ng+nl)
                combined = torch.cat([global_expanded, local_features], dim=-1)

                oadino_features.append(combined)
                feature_idx += n_not_masked
            else:
                # No non-masked patches for this batch element
                # Return empty tensor with correct shape (0, ng+nl)
                empty = torch.empty(
                    0,
                    global_features.shape[1] + object_features.shape[1],
                    device=global_features.device,
                    dtype=global_features.dtype,
                )
                oadino_features.append(empty)

        return oadino_features

    def encode_decode_object_patches(self, object_patches, mask):
        """Take object_patches and a mask and return"""
        masked_object_patches = object_patches[mask]
        masked_object_patches = self.transform(masked_object_patches)
        return self.vae(masked_object_patches)