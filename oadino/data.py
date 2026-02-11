import json
import pickle
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pickle
import torch
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from models import OADinoPreProcessor, OADinoModel, ConvVAE16
from transformers import AutoImageProcessor, AutoModel



# ------------------------------------------------------------------------------------------- #
#                     Extract Single and Multi-Object images and metadata                     #
# ------------------------------------------------------------------------------------------- #

@dataclass
class ClevrRecord:
	image_path: Path
	color: List[str]
	shape: List[str]
	size: List[str]
	material: List[str]
	pixel_coords: List[Sequence[float]]
	world_coords: List[Sequence[float]]
	num_objects: int


def load_clevr_scenes(scenes_json: Path) -> List[dict]:
	scenes_json = Path(scenes_json)
	with open(scenes_json, "r", encoding="utf-8") as f:
		data = json.load(f)
	return data.get("scenes", [])


def build_clevr_records(
	image_dir: Path,
	scenes_json: Path,
	keep_single: bool,
	max_images: Optional[int] = None,
) -> List[ClevrRecord]:
	"""
	Build a list of image records with CLEVR attributes.

	Args:
		image_dir: Directory containing CLEVR images.
		scenes_json: CLEVR scenes json.
		keep_single: If True keep only single-object scenes, else keep multi-object.
		max_images: Optional cap on number of records.
	"""
	image_dir = Path(image_dir)
	scenes = load_clevr_scenes(scenes_json)
	records: List[ClevrRecord] = []

	for scene in scenes:
		objects = scene.get("objects", [])
		is_single = len(objects) == 1
		if keep_single != is_single:
			continue

		filename = scene.get("image_filename")
		if filename is None:
			image_index = scene.get("image_index")
			split = scene.get("split", "val")
			if image_index is None:
				continue
			filename = f"CLEVR_{split}_{image_index:06d}.png"

		image_path = image_dir / filename
		if not image_path.exists():
			continue

		colors = [o.get("color") for o in objects]
		shapes = [o.get("shape") for o in objects]
		sizes = [o.get("size") for o in objects]
		materials = [o.get("material") for o in objects]
		pixel_coords = [o.get("pixel_coords") for o in objects]
		world_coords = [o.get("3d_coords") for o in objects]

		records.append(
			ClevrRecord(
				image_path=image_path,
				color=colors,
				shape=shapes,
				size=sizes,
				material=materials,
				pixel_coords=pixel_coords,
				world_coords=world_coords,
				num_objects=len(objects),
			)
		)

		if max_images is not None and len(records) >= max_images:
			break

	return records



# ------------------------------------------------------------------------------------------- #
#               Take test image, build a packet of images and extract CLS token               #
# ------------------------------------------------------------------------------------------- #

def _load_image(path: Path) -> Image.Image:
	img = Image.open(path).convert("RGB")
	return img


def build_image_packet(
	target_image: Image.Image,
	pool_images: Sequence[Path],
	num_images: int,
	rng: random.Random,
) -> List[Image.Image]:
	"""
	Build a packet of images by sampling from a pool and appending the target.
	"""
	if num_images < 2:
		raise ValueError("num_images must be >= 2")

	pool = [p for p in pool_images]
	if len(pool) == 0:
		raise ValueError("pool_images is empty")

	k = min(num_images - 1, len(pool))
	sampled = rng.sample(pool, k=k)
	images = [_load_image(p) for p in sampled]
	images.append(target_image)

	return images



# ------------------------------------------------------------------------------------------- #
#             Take test image, segment into object patches, and get VAE latents               #
# ------------------------------------------------------------------------------------------- #

@torch.no_grad()
def extract_embeddings_for_image(
	preprocessor: OADinoPreProcessor,
	model: OADinoModel,
	packet_images: List[Image.Image],
	target_index: int,
	pca_q: Optional[int],
	pca_niter: int,
	device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	"""
	Extract DINO global features and VAE latents for a target image in a packet.

	Returns:
		global_feature: (hidden_size,)
		latents_mean: (n_masked_patches, latent_dim)
		patch_mask: (n_patches,)
	"""
	preprocessor.backbone.to(device)
	preprocessor.backbone.eval()
	model.to(device)
	model.eval()

	global_features, object_patches, masks = preprocessor.get_global_features_and_patches(
		packet_images, pca_q=pca_q, pca_niter=pca_niter
	)

	global_feature = global_features[target_index]
	patches = object_patches[target_index]
	mask = masks[target_index]

	if mask.sum().item() == 0:
		raise ValueError("No foreground patches found for image")

	masked_patches = patches[mask]
	resize = transforms.Resize((model.vae.input_size, model.vae.input_size))
	masked_patches = resize(masked_patches)

	mean, logvar = model.vae.encode(masked_patches.to(device))

	return global_feature.detach().cpu(), mean.detach().cpu(), mask.detach().cpu()


def _record_to_metadata(record: ClevrRecord) -> Dict[str, list]:
	return {
		"color": record.color,
		"shape": record.shape,
		"size": record.size,
		"material": record.material,
		"pixel_coords": record.pixel_coords,
		"world_coords": record.world_coords,
		"num_objects": record.num_objects,
	}



# ------------------------------------------------------------------------------------------- #
#           SAVE embedding space: DINO CLS, latents, and labels per test image                #
# ------------------------------------------------------------------------------------------- #

def save_embeddings(
	output_dir: Path,
	tag: str,
	embeddings: List[torch.Tensor],
	latents: List[torch.Tensor],
	masks: List[torch.Tensor],
	metadata: List[Dict[str, list]],
):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	data = {
		"dino_cls": embeddings,
		"vae_latents": latents,
		"patch_masks": masks,
	}

	with open(output_dir / f"{tag}_embeddings.pkl", "wb") as f:
		pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

	with open(output_dir / f"{tag}_metadata.pkl", "wb") as f:
		pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)


def run_embedding_extraction(
	preprocessor: OADinoPreProcessor,
	model: OADinoModel,
	image_dir: Path,
	scenes_json: Path,
	output_dir: Path,
	keep_single: bool,
	num_samples: int = 1000,
	num_packet_images: int = 50,
	seed: int = 42,
	pca_q: Optional[int] = None,
	pca_niter: int = 2,
	device: torch.device = torch.device("cpu"),
	save_every: int = 100,
):
	"""
	Main pipeline that reproduces the paper's embedding extraction with OADino models.
	"""
	rng = random.Random(seed)
	transform = transforms.Compose(
		[
			transforms.Resize(520),
			transforms.CenterCrop(518),
		]
	)

	records = build_clevr_records(
		image_dir=image_dir,
		scenes_json=scenes_json,
		keep_single=keep_single,
		max_images=num_samples,
	)

	if len(records) == 0:
		raise ValueError("No records found for the given split")

	pool_paths = [r.image_path for r in records]

	dino_cls_list: List[torch.Tensor] = []
	latents_list: List[torch.Tensor] = []
	masks_list: List[torch.Tensor] = []
	metadata_list: List[Dict[str, list]] = []

	for idx, record in enumerate(tqdm(records, desc="Extracting embeddings")):
		try:
			target_image = _load_image(record.image_path)
			packet = build_image_packet(
				target_image=target_image,
				pool_images=pool_paths,
				num_images=num_packet_images,
				rng=rng,
				transform=transform,
			)

			global_feature, latents_mean, mask = extract_embeddings_for_image(
				preprocessor=preprocessor,
				model=model,
				packet_images=packet,
				target_index=len(packet) - 1,
				pca_q=pca_q,
				pca_niter=pca_niter,
				device=device,
			)

			dino_cls_list.append(global_feature)
			latents_list.append(latents_mean)
			masks_list.append(mask)
			metadata_list.append(_record_to_metadata(record))

		except Exception as exc:
			# Skip failed samples and continue
			print(f"Skipping image {record.image_path.name}: {exc}")
			continue

		if save_every > 0 and (idx + 1) % save_every == 0:
			tag = "single" if keep_single else "multi"
			save_embeddings(
				output_dir=output_dir,
				tag=f"{tag}_partial_{idx + 1}",
				embeddings=dino_cls_list,
				latents=latents_list,
				masks=masks_list,
				metadata=metadata_list,
			)

	tag = "single" if keep_single else "multi"
	save_embeddings(
		output_dir=output_dir,
		tag=tag,
		embeddings=dino_cls_list,
		latents=latents_list,
		masks=masks_list,
		metadata=metadata_list,
	)
	


# ------------------------------------------------------------------------------------------- #
#                   Load OADino pre-processor and OADino trained model                        #
# ------------------------------------------------------------------------------------------- #

device = "cuda" if torch.cuda.is_available() else "cpu"

dino_processor = AutoImageProcessor.from_pretrained(
    "facebook/dinov2-small", cache_dir='cache'
)
dino_model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir='cache')

preprocessor = OADinoPreProcessor(dino_processor, dino_model)

# move OADino trained model file to folder which will contain results associated to it
curr_model = 'CLEVR_train_4K_224_20260202_210355'

model_dir = Path(f'data/{curr_model}/best_model.pt')
vae = ConvVAE16()
model = OADinoModel(vae)
model_load = torch.load(model_dir, map_location=device)
model.load_state_dict(model_load["model_state_dict"])
model.to(device)
model.eval()



# ------------------------------------------------------------------------------------------- #
#                       Extract embeddings for single-object images                           #
# ------------------------------------------------------------------------------------------- #

seed = 42
pca_q = None
pca_niter = 2
save_every = 100

image_dir = Path('./dataset/single/images_single')
scenes_json = Path('./dataset/single/CLEVR_single_scenes.json')
output_dir = Path('.data/CLEVR_train_4K_224_20260202_210355/embeddings/single_object')
keep_single = True

num_samples = 10
# num_samples = 510
num_packet_images = 50

run_embedding_extraction(preprocessor, model, image_dir, scenes_json, output_dir, keep_single, num_samples, num_packet_images, seed, pca_q, pca_niter, device, save_every, transform)



# ------------------------------------------------------------------------------------------- #
#                 Load and check single-object saved embeddings and metadata                  #
# ------------------------------------------------------------------------------------------- #

# Load embeddings
with open("data/CLEVR_train_4K_224_20260202_210355/embeddings/single_object/single_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Check structure
print("Embeddings keys:", embeddings.keys())
print("dino_cls length:", len(embeddings["dino_cls"]))
print("dino_cls[0] shape:", embeddings["dino_cls"][0].shape)  # Should be (hidden_size,)
print("vae_latents[0] shape:", embeddings["vae_latents"][0].shape)  # Should be (n_masked_patches, latent_dim)
print("patch_masks[0] shape:", embeddings["patch_masks"][0].shape)  # Should be (n_patches,)

# Load metadata
with open("data/CLEVR_train_4K_224_20260202_210355/embeddings/single_object/single_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("\nMetadata length:", len(metadata))
print("Metadata[0] keys:", metadata[0].keys())
print("Metadata[0]['num_objects']:", metadata[0]["num_objects"])  # Should be 1 for single
print("Metadata[0]['color']:", metadata[0]["color"])  # Should be list of 1 string



# ------------------------------------------------------------------------------------------- #
#                       Extract embeddings for multi-object images                            #
# ------------------------------------------------------------------------------------------- #

seed = 42
pca_q = None
pca_niter = 2
save_every = 100

image_dir = Path('./dataset/multi/CLEVR_v1.0/images/val')
scenes_json = Path('./dataset/multi/CLEVR_v1.0/scenes/CLEVR_val_scenes.json')
output_dir = Path('.data/CLEVR_train_4K_224_20260202_210355/embeddings/multi_object')

keep_single = False

num_samples = 10
# num_samples = 5010
num_packet_images = 50

run_embedding_extraction(preprocessor, model, image_dir, scenes_json, output_dir, keep_single, num_samples, num_packet_images, seed, pca_q, pca_niter, device, save_every, transform)



# ------------------------------------------------------------------------------------------- #
#                 Load and check multi-object saved embeddings and metadata                   #
# ------------------------------------------------------------------------------------------- #

# Load embeddings
with open("data/CLEVR_train_4K_224_20260202_210355/embeddings/multi_object/multi_embeddings.pkl", "rb") as f:
    embeddings = pickle.load(f)

# Check structure
print("Embeddings keys:", embeddings.keys())
print("dino_cls length:", len(embeddings["dino_cls"]))
print("dino_cls[0] shape:", embeddings["dino_cls"][0].shape)  # Should be (hidden_size,)
print("vae_latents[0] shape:", embeddings["vae_latents"][0].shape)  # Should be (n_masked_patches, latent_dim)
print("patch_masks[0] shape:", embeddings["patch_masks"][0].shape)  # Should be (n_patches,)

# Load metadata
with open("data/CLEVR_train_4K_224_20260202_210355/embeddings/multi_object/multi_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print("\nMetadata length:", len(metadata))
print("Metadata[0] keys:", metadata[0].keys())
print("Metadata[0]['num_objects']:", metadata[0]["num_objects"])  # Should be 1 for single
print("Metadata[0]['color']:", metadata[0]["color"])  # Should be list of 1 string