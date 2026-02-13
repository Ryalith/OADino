from pathlib import Path

from datasets import Dataset, Features, Value
from datasets import Image as HFImage
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel

from oadino.models import OADinoModel, OADinoPreProcessor, ConvVAE16
from oadino.training import Trainer

## Loading Data


# Create dataset from image folder
def create_hf_dataset(image_dir, maxsize=-1):
    image_paths = sorted(Path(image_dir).glob("*.png"))

    # Create dataset dict
    data_dict = {
        "image": [str(p) for p in image_paths[:maxsize]],
        "filename": [p.name for p in image_paths[:maxsize]],
    }

    dataset = Dataset.from_dict(
        data_dict,
        features=Features(
            {
                "image": HFImage(),
                "filename": Value("string"),
            }
        ),
    )

    return dataset


# Load and transform
train_dataset = create_hf_dataset("C:/Users/maril/OneDrive/Desktop/GitHub/OADino/oadino/dataset/multi/CLEVR_v1.0/images/train", maxsize=-1)
test_dataset = create_hf_dataset("C:/Users/maril/OneDrive/Desktop/GitHub/OADino/oadino/dataset/multi/CLEVR_v1.0/images/test", maxsize=-1)

# Set format for PyTorch
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


def transform_batch(batch):
    batch["image"] = [transform(img.convert("RGB")) for img in batch["image"]]
    return batch


train_dataset = train_dataset.with_transform(transform_batch)
train_dataset_name = "CLEVR_train_full_224"
test_dataset = test_dataset.with_transform(transform_batch)

## Loading Backbone Models
hf_cache = "C:/Users/maril/OneDrive/Desktop/GitHub/OADino/oadino/cache"

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dino_processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small", cache_dir=hf_cache
)
dino_model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=hf_cache).to(device)

## Preparing training loop

pre_processor = OADinoPreProcessor(dino_processor, dino_model)
vae = ConvVAE16()
model = OADinoModel(vae)

trainer = Trainer(
    pre_processor, model, train_dataset, train_dataset_name, test_dataset, 224
)

## Execute training loop

trainer.train("C:/Users/maril/OneDrive/Desktop/GitHub/OADino/oadino")
