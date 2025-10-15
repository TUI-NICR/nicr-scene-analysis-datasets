# NYUv2 dataset

The NYU-Depth V2 dataset is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
It contains 1449 densely labeled pairs of aligned RGB and depth images.

For more details, see: [NYU Depth Dataset V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)

> As of Nov 2022, [precomputed normals](https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz) are not publicly available any longer. 
  We are trying to reach the authors. 
  Normal extraction is optional for now.

## Prepare dataset

1. Download and convert the dataset to the desired format:

  ```bash
  # general usage
  nicr_sa_prepare_dataset nyuv2 \
      /path/where/to/store/nyuv2
  ```

2. (Optional) Generate auxiliary data
  > **Note**: To use auxiliary data generation, the package must be installed with the `withauxiliarydata` option:
  > ```bash
  > pip install -e .[withauxiliarydata]
  > ```

  ```bash
  # for auxiliary data such as synthetic depth and rgb/panoptic embeddings
  nicr_sa_generate_auxiliary_data \
      --dataset nyuv2 \
      --dataset-path /path/to/already/prepared/nyuv2/dataset\
      --auxiliary-data depth image-embedding panoptic-embedding \
      --embedding-estimator-device cuda \
      --embedding-estimators alpha_clip__l14-336-grit-20m \
      --depth-estimator-device cuda \
      --depth-estimators depthanything_v2__indoor_large \
      --cache-models
  ```
  With arguments:
  - `--dataset-path`:
    Path to the prepared NYUv2 dataset.
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

