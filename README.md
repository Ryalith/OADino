# OADino

An implementation of the architecture proposed in "Oh-A-DINO: Understanding and Enhancing Attribute-Level Information in Self-Supervised Object-Centric Representations"

# Usage

Dependencies for the module are found in `pyproject.toml`

## Training

To train a new model you can reuse the code found in `training_loop.py`.
Further information is listed in the file to reproduce the results.

## Visualisations

Visualisation of the pre processing (segmentation and masking based on DINOv2 features) can be found in `visualisation_preprocess.ipynb`

Visualisation of the post processing (reconstruction of the object patches from the VAE) can be found in `visualisation_postprocess.ipynb`

## Evaluation

Evaluation metrics (top-10 precision with without ablation) and visualizations (top-8 retrievals on CLEVR and CLEVRTex) can be found in `final_evaluation.ipynb`

# References

https://arxiv.org/abs/2503.09867v3
