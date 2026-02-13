from pathlib import Path

import torch
from datasets import concatenate_datasets
from torchvision import transforms as T

from oadino.clevr import load_clevertex_dataset, load_clevr_dataset
from oadino.models import ConvVAE16, ConvVAE64, OADinoModel, OADinoPreProcessor
from oadino.training import Trainer

# TODO: create merged dataset with images from both CLEVR and CLEVRTEX with 4K images
# Train a CONVVAE64 Save Checkpoints,
# Probably train for more epochs than with the VAE16

### LOAD DATASETS

img_size = 224

transform = T.Compose(
    [
        T.Lambda(lambda img: img.convert("RGB")),  # Convert RGBA/grayscale to RGB
        T.Resize(img_size + int(img_size * 0.01) * 10),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

clevr_path = Path("/ssd2/mldata/CLEVR_v1.0/images/")
clevr_train_dataset, clevr_test_dataset, _ = load_clevr_dataset(
    clevr_path, 2048, transform
)

clevrtex_path = Path("/ssd2/mldata/CLEVRTexV2/clevrtexv2_full/")
clevrtex_train_dataset, clevrtex_test_dataset, _ = load_clevertex_dataset(
    clevrtex_path, 2048, transform
)

# Merge train datasets
combined_train = concatenate_datasets([clevr_train_dataset, clevrtex_train_dataset])

# Merge test datasets
combined_test = concatenate_datasets([clevr_test_dataset, clevrtex_test_dataset])


def apply_transform(batch):
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch


combined_train.set_transform(apply_transform)
combined_test.set_transform(apply_transform)

train_dataset_name = "CLEVR_CLEVRTex_train_4K_224"

### INSTANCIATE MODELS
img_size = 224
# dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
# dinov2_vitb14 = dinov2_vitb14.cuda()
# dinov2_vitb14.eval()
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_vits14 = dinov2_vits14.cuda()
dinov2_vits14.eval()


model = OADinoModel(ConvVAE64())
# model = OADinoModel(ConvVAE64())
# pre_processor = OADinoPreProcessor(dinov2_vitb14)
pre_processor = OADinoPreProcessor(dinov2_vits14)

### Train Models

trainer = Trainer(
    pre_processor,
    model,
    # "dinov2_vitb14",
    "dinov2_vits14",
    combined_train,
    train_dataset_name,
    combined_test,
    224,
)
trainer.train("/ssd2/mldata/oadino", 20)
