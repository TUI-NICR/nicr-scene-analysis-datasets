# ScanNet dataset

ScanNet is an RGB-D video dataset containing 2.5 million views in more than 1500 scans, annotated with 3D camera poses, surface reconstructions, and instance-level semantic segmentations.
For more details, see: [ScanNet v2](http://www.scan-net.org/)

Note: 3D meshes and surface reconstructions are not included in the preparation of the dataset.


## Prepare dataset
1. Download the Dataset:

    To be able to download the dataset fill out the [ScanNet Terms of Use](http://kaldir.vc.in.tum.de/scannet/ScanNet_TOS.pdf) and send it to them at scannet@googlegroups.com. Once your request is approved, you will receive a `download_scannet.py` script.

    Execute it with:
    ```bash
    # general usage
    python download-scannet.py -o /path/where/to/download/ScanNet
    ```

2. Convert dataset:

    ```bash
    # general usage (note that one process might use more than 3GB RAM)
    nicr_sa_prepare_dataset scannet \
        /path/where/to/download/ScanNet \
        /path/where/to/convert/ScanNet \
        [--n-processes N] \
        [--subsample N0]
        [--additional-subsamples N1 N2]
        [--label-map-file /path/to/scannet-labels.combined.tsv]
    ```
      With arguments:
    - `--n-processes`:
      The number of worker processes to spawn.
    - `--subsample`
      The subsample that is exported to the output folder.
    - `--additional_subsamples`:
      Tor additional subsampled versions of the dataset.
    - `--label-map-file`:
      Path to scannet-labels.combined.tsv, if not specified assumed to be located
      in source dir.


3. (Optional) Generate auxiliary data:
  > **Note**: To use auxiliary data generation, the package must be installed with the `withauxiliarydata` option:
  > ```bash
  > pip install -e .[withauxiliarydata]
  > ```

    ```bash
    # for auxiliary data such as synthetic depth and rgb/panoptic embeddings
    nicr_sa_generate_auxiliary_data \
        --dataset scannet \
        --dataset-path /path/to/already/prepared/ScanNet/dataset \
        --auxiliary-data depth image-embedding panoptic-embedding \
        --embedding-estimator-device cuda \
        --embedding-estimators alpha_clip__l14-336-grit-20m \
        --depth-estimator-device cuda \
        --depth-estimators depthanything_v2__indoor_large \
        --cache-models
    ```
    With arguments:
    - `--dataset-path`:
      Path to the prepared ScanNet dataset.
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