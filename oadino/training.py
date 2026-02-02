import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .models import OADinoModel, OADinoPreProcessor


class PreProcessedTensorStorage(Dataset):
    def __init__(
        self,
        save_dir: Path,
        n_samples,
        feature_dim,
        n_channels,
        n_patches,
        patch_size,
        flush_freq=64,
        mode="w+",  # 'w+' for write, 'r' for read
    ):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)

        self.n_samples = n_samples
        self.feature_dim = feature_dim
        self.n_channels = n_channels
        self.n_patches = n_patches
        self.patch_size = patch_size
        self.flush_freq = flush_freq
        self.current_idx = 0

        # Create memory-mapped arrays
        self.features = np.memmap(
            self.save_dir / "features.dat",
            dtype="float32",
            mode=mode,
            shape=(n_samples, feature_dim),
        )
        self.patches = np.memmap(
            self.save_dir / "patches.dat",
            dtype="float32",
            mode=mode,
            shape=(n_samples, n_patches, n_channels, patch_size, patch_size),
        )
        self.masks = np.memmap(
            self.save_dir / "masks.dat",
            dtype="bool",
            mode=mode,
            shape=(n_samples, n_patches),
        )

    def add_batch(self, features_batch, patches_batch, masks_batch):
        batch_size = features_batch.shape[0]
        end_idx = self.current_idx + batch_size

        if end_idx > self.n_samples:
            raise ValueError(
                f"Trying to add batch beyond allocated storage. "
                f"Current: {self.current_idx}, Batch size: {batch_size}, "
                f"Total allocated: {self.n_samples}"
            )

        self.features[self.current_idx : end_idx] = features_batch.cpu().numpy()
        self.patches[self.current_idx : end_idx] = patches_batch.cpu().numpy()
        self.masks[self.current_idx : end_idx] = masks_batch.cpu().numpy()

        # Flush to disk periodically
        if end_idx // self.flush_freq > self.current_idx // self.flush_freq:
            self.features.flush()
            self.patches.flush()
            self.masks.flush()

        self.current_idx = end_idx

    def finalize(self):
        """Flush all data to disk and save metadata"""
        self.features.flush()
        self.patches.flush()
        self.masks.flush()

        # Save metadata
        metadata = {
            "n_samples": self.n_samples,
            "actual_samples": self.current_idx,  # Actual number of samples written
            "feature_dim": self.feature_dim,
            "n_channels": self.n_channels,
            "n_patches": self.n_patches,
            "patch_size": self.patch_size,
        }

        with open(self.save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def from_dir(save_dir: Path):
        """Load a preprocessed dataset from disk"""
        save_dir = Path(save_dir)

        if not save_dir.exists():
            raise FileNotFoundError(f"Directory {save_dir} does not exist")

        metadata_path = save_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                "Make sure the dataset was finalized properly."
            )

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # Create instance with read mode
        storage = PreProcessedTensorStorage(
            save_dir=save_dir,
            n_samples=metadata["actual_samples"],  # Use actual samples written
            feature_dim=metadata["feature_dim"],
            n_channels=metadata["n_channels"],
            n_patches=metadata["n_patches"],
            patch_size=metadata["patch_size"],
            mode="r",  # Read-only mode
        )

        # Set current_idx to the number of samples actually written
        storage.current_idx = metadata["actual_samples"]

        return storage

    # PyTorch Dataset interface methods
    def __len__(self):
        """Return the number of samples actually written"""
        return self.current_idx

    def __getitem__(self, idx):
        """Get a single sample"""
        if idx >= self.current_idx:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.current_idx} samples"
            )

        return {
            "features": torch.from_numpy(self.features[idx].copy()),
            "patches": torch.from_numpy(self.patches[idx].copy()),
            "masks": torch.from_numpy(self.masks[idx].copy()),
        }

    def __del__(self):
        """Cleanup: ensure data is flushed when object is deleted"""
        if hasattr(self, "features"):
            try:
                self.features.flush()
                self.patches.flush()
                self.masks.flush()
            except:
                pass  # Already closed or in read mode


def get_preprocessed_data(
    dataset,
    dataset_name,
    image_transform,
    image_size,
    preprocessor,
    base_dir,
    batch_size,
):
    model_name = preprocessor.backbone.config.name_or_path.replace("/", "_")
    savedir = base_dir / dataset_name / model_name

    if savedir.exists():
        try:
            return PreProcessedTensorStorage.from_dir(savedir)
        except FileNotFoundError as e:
            print(f"Found directory {savedir} but could not load dataset")
            print(f"{e}")
            print("Data will be reprocessed")
            pass

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return preprocess_and_save_dataset(
        dataloader,
        dataset_name,
        image_transform,
        image_size,
        preprocessor,
        base_dir,
        batch_size,
    )


def preprocess_and_save_dataset(
    dataloader,
    dataset_name: str,
    image_transform,
    image_size,
    preprocessor: OADinoPreProcessor,
    base_dir: Path,
):
    """
    Preprocess a dataset and save to disk using memory-mapped storage.

    Args:
        dataloader: DataLoader providing batches of images
        dataset_name: Name of the dataset (e.g., 'imagenet', 'cifar10')
        image_transform: Transform to apply to images
        image_size: Size of images after transformation
        preprocessor: OADinoPreProcessor instance
        base_dir: Base directory for saving preprocessed data
    """
    model_name = preprocessor.backbone.config.name_or_path.replace("/", "_")
    savedir = base_dir / dataset_name / model_name
    savedir.mkdir(exist_ok=True, parents=True)

    # Calculate total number of samples
    # Assuming dataloader.dataset has __len__
    try:
        n_samples = len(dataloader.dataset)
    except (TypeError, AttributeError):
        # If dataset doesn't have length, count batches
        print("Dataset length unknown, counting batches...")
        n_samples = sum(
            batch[0].shape[0] if isinstance(batch, (list, tuple)) else batch.shape[0]
            for batch in dataloader
        )
        # Reset dataloader (this might not work for all dataloaders)
        print(f"Counted {n_samples} samples")

    preprocessed_tensor_storage = PreProcessedTensorStorage(
        save_dir=savedir,
        n_samples=n_samples,
        feature_dim=preprocessor.backbone.config.hidden_size,
        n_channels=3,
        n_patches=(image_size // preprocessor.backbone.config.patch_size) ** 2,
        patch_size=preprocessor.backbone.config.patch_size,
    )

    print(f"Processing {n_samples} samples...")
    print(f"Saving to: {savedir}")

    samples_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Get preprocessed features
            feats, patches, masks = preprocessor.get_global_features_and_patches(
                ..., pca_q=None, pca_niter=2
            )

            # Add to storage
            preprocessed_tensor_storage.add_batch(feats, patches, masks)
            samples_processed += feats.shape[0]

    print("Finalizing dataset...")
    preprocessed_tensor_storage.finalize()

    print(f"Dataset saved to {savedir}")
    print(f"Total samples processed: {samples_processed}")

    return preprocessed_tensor_storage


def vae_loss(x, x_hat, mean, log_var, beta=1e-4):
    reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + beta * kld


class Trainer:
    def __init__(
        self,
        preprocessor: OADinoPreProcessor,
        model: OADinoModel,
        dataset: Dataset,
        dataset_name: str,
        image_transform,
        image_size,
    ):
        self.preproc = preprocessor
        self.model = model
        self.dataset = dataset
        self.dset_name = dataset_name
        self.transform = image_transform
        self.img_size = image_size

    def train(
        self,
        base_dir: Path,
        Optimizer=torch.optim.Adam,
        lr=1e-3,
        preproc_batch_size=64,
        train_batch_size=64,
        device=torch.device("cpu"),
        loss_beta=1e-4,
    ):
        """
        base_dir is where all data are saved by default
        please don't use a path within the git to not add too much data in it
        """
        self.preproc.to(device)
        self.preproc.eval()

        preprocessed_dataset = get_preprocessed_data(
            self.dataset,
            self.dset_name,
            self.transform,
            self.img_size,
            self.preproc,
            base_dir,
            preproc_batch_size,
        )

        # we delete the preproc to save RAM
        del self.preproc

        self.model.to(device)
        self.model.train()

        dataloader = DataLoader(preprocessed_dataset, batch_size=train_batch_size)
        optimizer = Optimizer(self.model.parameters(), lr=lr)

        for batch_idx, batch in dataloader:
            patches = batch["patches"].to(device)
            masks = batch["masks"].to(device)

            x_hat, mean, logvar = self.model.encode_decode_object_patches(
                patches, masks
            )

            loss = vae_loss(patches, x_hat, mean, logvar, loss_beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
