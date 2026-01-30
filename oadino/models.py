from typing import Sequence

import torch
from torch import nn
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


# Simple VAE for testing, A UNet might be better for images
class DenseVAE(VAE):
    def __int__(self, layer_dims: Sequence[int], Act=nn.ReLU):
        super().__init__()

        layers_e = []
        layers_d = []

        for i in range(len(layer_dims) - 1):
            layers_e.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            layers_e.append(Act())
        layers_e.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        for i in reversed(range(len(layer_dims) - 1)):
            layers_d.append(nn.Linear(layer_dims[i + 1], layer_dims[i]))
            layers_d.append(Act())
        layers_d.append(nn.Linear(layer_dims[1], layer_dims[0]))

        self.mean_layer = nn.Linear(layer_dims[-1], 2)
        self.logvar_layer = nn.Linear(layer_dims[-1], 2)

        self.encoder = nn.Sequential(*layers_e)
        self.decoder = nn.Sequential(*layers_d)

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(var.device)
        z = mean + var * epsilon
        return z

    def encode(self, x):
        h = self.encoder(x)
        mean, logvar = self.mean_layer(h), self.logvar_layer(h)
        return mean, logvar

    def decode(self, z):
        return self.decoder(z)


# First implementation of the Oh-A-Dino model
# The idea is to, during training:
# - call segment_image to get segmented inputs for the rest of the model and the mask
# - call create_patches to create a set of patches
# - call encode_decode_patches to get the latent and reconstructed patches
# - compute the paper's vae encoder loss and backprop
class OADinoModel(nn.Module):
    def __init__(self, processor, backbone: PreTrainedModel, vae: VAE):
        super().__init__()
        self.processor = processor
        self.backbone = backbone
        self.vae = vae
        pass

    @staticmethod
    def _get_mask(self, flat_patches, pca_q, pca_niter):
        flattened_patches_centered = flat_patches - flat_patches.mean(dim=0)
        U, S, V = torch.pca_lowrank(flattened_patches_centered, pca_q, False, pca_niter)
        patch_pca = torch.matmul(flattened_patches_centered, V[:, 0])
        return patch_pca > patch_pca.median()

    def segment_image(self, image, pca_q, pca_niter):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt")
            outputs = self.backbone(**inputs)

            # (batch_size, 1+n_patches, hidden_size)
            backbone_lhs = outputs.last_hidden_state
            backbone_patches = backbone_lhs[:, 1:, :]

            batch_size, n_patches, hidden_size = backbone_lhs.shape
            n_patches -= 1

            # 1. Foreground separation of patches

            flattened_patches = backbone_patches.reshape((-1, hidden_size))
            rough_mask = self._get_mask(flattened_patches, pca_q, pca_niter)

            # 2. Refinig object consistency

            masked_patches = flattened_patches[rough_mask]
            mask = self._get_mask(masked_patches, pca_q, pca_niter)

            # 3. Remapping to image space

            pixel_mask = mask.reshape(batch_size, n_patches, 1)
            pixel_mask._expand(-1, -1, hidden_size)
            segmented_inputs = inputs
            segmented_inputs.pixel_values = inputs.pixel_values * pixel_mask

            return segmented_inputs, mask

    def create_patches(self, inputs):
        # TODO create patches from input_pixel values and the model's patch size
        # A priori image size is 518*518
        # A priori dinoV2 uses 14*14 patches

        pixel_values = inputs.pixel_values  # (batch, 3, 518, 518)
        batch_size, channels, height, width = pixel_values.shape

        patch_size = 14  # DINOv2 patch size

        # Calculate number of patches
        num_patches_h = height // patch_size  # 518 // 14 = 37
        num_patches_w = width // patch_size  # 518 // 14 = 37

        # Reshape to extract patches
        # (batch, 3, 224, 224) -> (batch, 3, 37, 14, 37, 14)
        patches = pixel_values.reshape(
            batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size
        )

        # Rearrange dimensions: (batch, 3, 16, 14, 16, 14) -> (batch, 37, 37, 3, 14, 14)
        patches = patches.permute(0, 2, 4, 1, 3, 5)

        # Flatten patches: (batch, 16, 16, 3, 14, 14) -> (batch, 37*37, 588)
        patches = patches.reshape(batch_size, num_patches_h * num_patches_w, -1)

        return patches

        pass

    def forward(self, image, pca_q=None, pca_niter=2):
        """
        Perform feature extraction on image patches (masked through PCA) using the model's VAE

        Notes:
        - Some reviews mention using other segmentation methods (like using UNets), we could add implementations ?
        """
        segmented_inputs, mask = self.segment_image(image, pca_q, pca_niter)

        object_patches = self.create_patches(segmented_inputs)

        # Get local representation: encoded patches
        encoded_patches = self.vae.encode(object_patches)

        # Get global representation: cls token on segmented image
        outputs = self.backbone(**segmented_inputs)
        cls_token = outputs.last_hidden_state[:, 0, :].unsqueeze(1)
        cls_token._expand(-1, encoded_patches.shape[1], -1)
        full_features = torch.cat([cls_token, encoded_patches], dim=1)

        return full_features

    def encode_decode_patches(self, patches):
        encoded_patches = self.vae.encode(patches)
        decoded_patches = self.vae.decode(encoded_patches)

        return encoded_patches, decoded_patches
