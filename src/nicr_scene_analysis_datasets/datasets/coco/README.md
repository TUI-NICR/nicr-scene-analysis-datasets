# COCO dataset

COCO is a large-scale object detection, segmentation, and captioning dataset.
It contains over 200.000 labeled images with 80 object and 91 stuff categories
for panoptic segmentation.

For more details, see: [COCO dataset](https://cocodataset.org/#home)

## Prepare dataset
1. Convert the dataset
    ```bash
    # general usage
    nicr_sa_prepare_dataset coco \
        /path/where/to/store/coco/
    ```

2. (Optional) Generate auxiliary data
    > **Note**: To use auxiliary data generation, the package must be installed with the `withauxiliarydata` option:
    > ```bash
    > pip install -e .[withauxiliarydata]
    > ```
  
    ```bash
    # for auxiliary data such as synthetic depth and rgb/panoptic embeddings
    nicr_sa_generate_auxiliary_data \
        --dataset coco \
        --dataset-path /path/to/already/prepared/coco/dataset \
        --auxiliary-data depth image-embedding panoptic-embedding \
        --embedding-estimator-device cuda \
        --embedding-estimators alpha_clip__l14-336-grit-20m \
        --depth-estimator-device cuda \
        --depth-estimators depthanything_v2__indoor_large \
        --cache-models
    ```

    With arguments:
    - `--dataset-path`:
        Path to the prepared COCO dataset.
    - `--auxiliary-data`:
        Types of auxiliary data to generate:
        - `depth`: Generates synthetic depth images from RGB.
        - `image-embedding`: Uses Alpha-CLIP to generate an embedding for the entire image.
        - `panoptic-embedding`: Uses Alpha-CLIP to generate an embedding for each panoptic mask.
    - `--depth-estimator-device`:
        Device to use for depth estimation (`cpu` or `cuda`).
    - `--depth-estimators`:
        Depth estimator(s) to use. Use `depthanything_v2__indoor_large` to match DVEFormer.
    - `--embedding-estimator-device`:
        Device to use for embedding estimation (`cpu` or `cuda`).
    - `--embedding-estimators`:
        Embedding estimator(s) to use. Use `alpha_clip__l14-336-grit-20m` to match DVEFormer.
    - `--cache-models`:
        Cache models locally to avoid reloading them in future runs.
