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

## Notes

> Hypersim uses non-standard perspective projection matrices (with 
tilt-shift photography parameters) in most scenes. As common frameworks, such as 
MIRA or ROS, do not support this projection, we convert the camera parameters if 
possible or project the data/annotations back to a standard camera ignoring the 
tilt-shift parameters. Note that this is not a perfect conversion and introduces 
some artifacts, i.e., void pixels as we only back-project points without 
contradictions. Void is assigned to ~5% of the pixels.
However, rendering full images with a standard perspective projection
requires buying the dataset meshes.
For more details, see [this issue](https://github.com/apple/ml-hypersim/issues/24).
To disable this conversion and to stick to original images, pass the
`--no-tilt-shift-conversion` parameter to the prepare script.

> We observed that merging semantic and instance labels in order to derive 
panoptic labels might slightly change the semantic in few images. This is 
because there are some pixels that belong to a thing class but are not assigned 
to any instance (instance=0), e.g., in scene ai_052_001, a lamp is labeled as 
lamp but is not annotated as instance. Panoptic merging assigns void for those 
pixels. There is no workaround for this issue. Affected scenes: 
valid: ai_023_003, ai_041_003, ai_052_001, ai_052_003 -> 1576566 pixels (0.03%);
test: ai_005_001, ai_008_005, ai_008_005, ai_022_001 -> 801359 pixels (0.01%).
Computing mIoU in [0, 1] to semantic / panoptic_semantic as ground truth 
changes the result by ~0.0001-0.0002 - so it is not a big issue and negligible.

> We further observed that some images are not correctly annotated. There are 
instances that are assigned to multiple semantic classes. While most overlaps 
are with void (unlabeled textures -> void label), we observed other issues for:
ai_017_004: semantic classes 35 + 40: lamp + otherprop -> some small stuff in 
the background; ai_021_008: semantic classes 12 + 35 -> kitchen counter + lamp 
belong to same instance -> might be an annotation fault; ai_022_009: semantic 
classes 1 + 8 -> door frame labeled as wall, but door instance contains both 
the door frame and the door.

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
    nicr_sa_prepare_dataset hypersim \
        /path/where/to/datasets/hypersim \
        /path/to/uncompressed/hypersim \
        [--additional-subsamples N1 N2] \
        [--n-processes N]
    ```

With arguments:
- `--additional_subsamples`:
  For additional subsampled versions of the dataset.
- `--n-processes`:
  Number of worker processes to spawn.
- `--no-tilt-shift-conversion`:
  Disable projecting the data/annotations back to a standard camera ignoring the 
  tilt-shift parameters (use this to create dataset compatible with < v050).
