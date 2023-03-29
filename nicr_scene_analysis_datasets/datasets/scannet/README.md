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

With Arguments:
- `--n-processes`:
  the number of worker processes to spawn.
- `--subsample`
  the subsample that is exported to the output folder
- `--additional_subsamples`:
  for additional subsampled versions of the dataset
- `--label-map-file`:
  path to scannet-labels.combined.tsv, if not specified assumed to be located in source dir
