# Hypersim dataset

For many fundamental scene understanding tasks, it is difficult or impossible
to obtain per-pixel ground truth labels from real images. We address this
challenge with Hypersim, a photorealistic synthetic dataset for holistic indoor
scene understanding. To create our dataset, we leverage a large repository of
synthetic scenes created by professional artists, and we generate 77,400 images
of 461 indoor scenes with detailed per-pixel labels and corresponding ground
truth geometry. Our dataset: (1) relies exclusively on publicly available 3D
assets; (2) includes complete scene geometry, material information, and
lighting information for every scene; (3) includes dense per-pixel semantic
instance segmentations for every image; and (4) factors every image into
diffuse reflectance, diffuse illumination, and a non-diffuse residual term
that captures view-dependent lighting effects. Together, these features make
our dataset well-suited for geometric learning problems that require direct 3D
supervision, multi-task learning problems that require reasoning jointly over
multiple input and output modalities, and inverse rendering problems.

For more details, see: [Hypersim](https://machinelearning.apple.com/research/hypersim)

## Prepare dataset

1. Download and unzip dataset files:
    ```bash
    wget https://raw.githubusercontent.com/apple/ml-hypersim/6cbaa80207f44a312654e288cf445016c84658a1/code/python/tools/dataset_download_images.py

    # general usage
    python dataset_download_images.py \
        --downloads_dir /path/to/download \
        --decompress_dir /path/to/uncompressed/hypersim
    ```

2. Convert dataset:
    ```bash
    # general usage
    python -m nicr_scene_analysis_datasets.datasets.hypersim.prepare_dataset \
        /path/where/to/datasets/hypersim \
        /path/to/uncompressed/hypersim \
        [--additional-subsamples N1 N2] \
        [--multiprocessing]
    ```

With Arguments:
- `--additional_subsamples`:
  for additional subsampled versions of the dataset
- `--multiprocessing`:
  if multiprocessing should be used
