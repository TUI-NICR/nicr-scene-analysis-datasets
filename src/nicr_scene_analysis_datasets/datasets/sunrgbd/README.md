# SUNRGB-D Dataset

The SUNRGB-D dataset is comprised of images of four different cameras, i.e.,
Intel Realsense, Asus Xtion, and Microsoft Kinect v1 and v2.
It contains all images from NYUv2, manually selected images from Berkeley
B3DO and SUN3D as well as newly shot images.

It contains 10,335 densely labeled pairs of aligned RGB and depth images.

For more details, see: [SUNRGB-D dataset](https://rgbd.cs.princeton.edu/)

We further extracted dense 2d instance annotations from annotated 3d boxes to 
enable panoptic segmentation on SUNRGB-D. Over time, we created two versions 
for additional instance annotations:
- 'emsanet': this initial version was created for training EMSANet (efficient 
  panoptic segmentation) - see IJCNN 2022 paper - and was also used for 
  EMSAFormer (efficient panoptic segmentation) - see IJCNN 2023 paper
- 'panopticndt': this revised version was created along with the work for 
  PanopticNDT (panoptic mapping) - see IROS 2023 paper, it refines large parts 
  of the instance extraction (see changelog for v0.6.0 of this package).


## Prepare dataset
```bash
# general usage (latest PanopticNDT version)
nicr_sa_prepare_dataset sunrgbd \
    /path/where/to/store/sunrgbd \
    --create-instances \
    --copy-instances-from-nyuv2 \
    --nyuv2-path /path/to/already/prepared/nyuv2/

# general usage (EMSANet version - use this version to reproduce results 
# reported in EMSANet or EMSAFormer paper)
nicr_sa_prepare_dataset sunrgbd \
    /path/where/to/store/sunrgbd \
    --create-instances \
    --instances-version emsanet \
    --copy-instances-from-nyuv2 \
    --nyuv2-path /path/to/already/prepared/nyuv2/
```

> Note: NYUv2 matching requires NYUv2 prepared first.

With arguments:

- `--create-instances`:
  Whether instances should be created by matching 3D boxes with point clouds.
- `--instances-version`:
  Version of instance annotations to extract, see notes above.
- `--copy-instances-from-nyuv2`:
  Whether instances and orientations should copied from (already prepared!) 
  NYUv2 dataset.
- `--nyuv2-path /path/to/datasets/nyuv2`:
  Path to (already prepared!) NYUv2 dataset when using 
  `copy-instances-from-nyuv2`.
