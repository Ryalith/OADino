from pathlib import Path

import torch
from datasets import concatenate_datasets
from torchvision import transforms as T

from oadino.clevr import load_clevertex_dataset, load_clevr_dataset
from oadino.models import ConvVAE16, ConvVAE64, OADinoModel, OADinoPreProcessor
from oadino.training import Trainer

### Training Variables

# Path to the CLEVR images directory, should contain three directories named test train val
clevr_path = Path("some/path/CLEVR_v1.0/images/")

# Path to the CLEVRTex images directory, should contain 50 directories named 0; 1; ...
# Themselves containing images
clevrtex_path = Path("some/path/clevrtexv2_full/")

# number of images on which to perform training
# half of the images will come from CLEVR and the other for CLEVRTex
num_training_images = 4096

# The name of the dataset
# The trainer will save the preprocessed data under the specified name
# If there is already a directory with the specified name the training will skip preprocessing and load the data instead
# For 4K points expect around an hour of pre-processing
base_dir = "some/path/to/a/dir"
train_dataset_name = "CLEVR_CLEVRTex_train_4K_224"

# The kind of VAE used for the model
# The paper takes 14x14 patches, rescales them to 64x64 and feeds inputs to a VAE
# We also used a smaller 16x16 VAE for faster testing
# For 4k points expect several dozen minutes for VAE16 and several hours for VAE64
vae = ConvVAE64()
# vae = ConvVAE16()

# If you want to resume training from a previous run you can point to the weights of the specific runs
# By default runs are saved under base_dir/runs/
# There are also tensoboard logs
# Some logs were saved in this git to checkout
resume_checkpoint = None
# resume_checkpoint = (
#     "runs/CLEVR_CLEVRTex_train_4K_224_VAE64_20260213_221018/checkpoints/best_model.pt"
# )

num_epochs = ...

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

clevr_train_dataset, clevr_test_dataset, _ = load_clevr_dataset(
    clevr_path, num_training_images // 2, transform
)

clevrtex_train_dataset, clevrtex_test_dataset, _ = load_clevertex_dataset(
    clevrtex_path, num_training_images // 2, transform
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


### INSTANCIATE MODELS
img_size = 224
# dinov2_vitb14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
# dinov2_vitb14 = dinov2_vitb14.cuda()
# dinov2_vitb14.eval()
dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
dinov2_vits14 = dinov2_vits14.cuda()
dinov2_vits14.eval()


model = OADinoModel(vae)
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
trainer.train(
    base_dir=base_dir,
    num_epochs=num_epochs,
    resume_from_checkpoint=resume_checkpoint,
)
