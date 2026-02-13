from pathlib import Path
from datasets import Dataset, Features, Value
from datasets import Image as HFImage
import random


def create_clevr_hf_dataset(image_dir, maxsize=-1):
    """Read the files in the CLEVR dataset and create a huggingface dataset from it"""
    image_paths = sorted(Path(image_dir).glob("*.png"))

    # Create dataset dict
    data_dict = {
        "image": [str(p) for p in image_paths[:maxsize]],
        "image_id": [p.name for p in image_paths[:maxsize]],
    }

    dataset = Dataset.from_dict(
        data_dict,
        features=Features(
            {
                "image": HFImage(),
                "image_id": Value("string"),
            }
        ),
    )

    return dataset


def load_clevr_dataset(images_path: Path, maxsize, transform):
    """ Return train test and val split datasets of CLEVR of size maxsize, with the given transformation on images
    /!\\ THE DATASET IS USED FOR UNSUPERVISED TRAINING, NO LABELS /!\\
    """

    # Load and transform
    train_dataset = create_clevr_hf_dataset(images_path / "train", maxsize=maxsize)
    test_dataset = create_clevr_hf_dataset(images_path / "test", maxsize=maxsize)
    val_dataset = create_clevr_hf_dataset(images_path / "val", maxsize=maxsize)

    def transform_batch(batch):
        batch["image"] = [transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    train_dataset = train_dataset.with_transform(transform_batch)
    test_dataset = test_dataset.with_transform(transform_batch)
    val_dataset = val_dataset.with_transform(transform_batch)

    return train_dataset, test_dataset, val_dataset


def get_clevertex_image_paths(base_dir, max_size=-1, seed=42):
    """
    Collect all CLEVRTex image paths from the directory structure.

    Args:
        base_dir: Base directory containing subdirectories 0-49
        max_size: Maximum number of images per split. If -1, use all available images divided by 3

    Returns:
        List of tuples (image_path, image_id)
    """
    image_paths = []

    base_path = Path(base_dir)

    # Iterate through subdirectories 0-49
    for i in range(50):
        subdir = base_path / str(i)
        if not subdir.exists():
            raise FileNotFoundError(subdir)

        # Find all CLEVRTEXv2_full_* directories
        for clevrtex_dir in subdir.glob("CLEVRTEXv2_full_*"):
            if not clevrtex_dir.is_dir():
                continue

            # Extract the sequence number from directory name
            # CLEVRTEXv2_full_****** -> ******
            dir_name = clevrtex_dir.name
            if dir_name.startswith("CLEVRTEXv2_full_"):
                sequence = dir_name.replace("CLEVRTEXv2_full_", "")

                # Construct the image filename
                image_filename = f"CLEVRTEXv2_full_{sequence}_0003.png"
                image_path = clevrtex_dir / image_filename

                if image_path.exists():
                    # Use the full relative path as ID
                    image_id = f"{i}/{clevrtex_dir.name}/{image_filename}"
                    image_paths.append((str(image_path), image_id))

    # Shuffle to ensure random distribution across splits
    random.seed(seed)
    random.shuffle(image_paths)

    # Determine actual size per split
    total_images = len(image_paths)
    if max_size == -1:
        size_per_split = total_images // 3
    else:
        size_per_split = min(max_size, total_images // 3)

    print(f"Found {total_images} CLEVRTex images")
    print(f"Using {size_per_split} images per split")

    return image_paths, size_per_split


def load_clevertex_dataset(base_dir, max_size, transform, seed=42):
    """
    Create a HuggingFace dataset from CLEVRTex images with 33/33/33 train/val/test split.

    Args:
        base_dir: Base directory containing CLEVRTex subdirectories 0-49
        max_size: Maximum number of images per split. If -1, use all available (total/3)

    Returns:
        DatasetDict with train, validation, and test splits
    """
    # Get all image paths
    all_paths, size_per_split = get_clevertex_image_paths(base_dir, max_size, seed)

    # Split into train/val/test (33/33/33)
    train_paths = all_paths[:size_per_split]
    val_paths = all_paths[size_per_split : 2 * size_per_split]
    test_paths = all_paths[2 * size_per_split : 3 * size_per_split]

    # Create datasets for each split
    def create_split_dataset(paths):
        data = {
            "image": [str(path) for path, _ in paths],
            "image_id": [img_id for _, img_id in paths],
        }
        return Dataset.from_dict(
            data, features=Features({"image": HFImage(), "image_id": Value("string")})
        )

    train_dataset = create_split_dataset(train_paths)
    val_dataset = create_split_dataset(val_paths)
    test_dataset = create_split_dataset(test_paths)

    def transform_batch(batch):
        batch["image"] = [transform(img.convert("RGB")) for img in batch["image"]]
        return batch

    train_dataset = train_dataset.with_transform(transform_batch)
    test_dataset = test_dataset.with_transform(transform_batch)
    val_dataset = val_dataset.with_transform(transform_batch)

    return train_dataset, test_dataset, val_dataset

