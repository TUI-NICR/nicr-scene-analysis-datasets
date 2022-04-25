# SUNRGB-D Dataset

The SUNRGB-D dataset is comprised of images of four different cameras, i.e.,
Intel Realsense, Asus Xtion, and Microsoft Kinect v1 and v2.
It contains all images from NYUv2, manually selected images from Berkeley
B3DO and SUN3D as well as newly shot images.

It contains 10.335 densely labeled pairs of aligned RGB and depth images.

For more details, see: [SUNRGB-D dataset](https://rgbd.cs.princeton.edu/)

## Prepare dataset
```bash
# general usage
python -m nicr_scene_analysis_datasets.datasets.sunrgbd.prepare_dataset \
    /path/where/to/store/sunrgbd \
    --create-instances \
    --copy-instances-from-nyuv2 \
    --nyuv2-path /path/to/already/prepared/nyuv2/
```

> Note: NYUv2 matching requires NYUv2 prepared first.

With Arguments:
- `--create-instances`:
  whether instances should be created by matching 3D boxes with point clouds
- `--copy-instances-from-nyuv2`:
  whether instances and orientations should copied from (already prepared!) NYUv2 dataset
- `--nyuv2-path /path/to/datasets/nyuv2`:
  path to (already prepared!) NYUv2 dataset when using `copy-instances-from-nyuv2`
