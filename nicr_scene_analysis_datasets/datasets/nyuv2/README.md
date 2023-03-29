# NYUv2 dataset

The NYU-Depth V2 dataset is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
It contains 1449 densely labeled pairs of aligned RGB and depth images.

For more details, see: [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

> As of Nov 2022, [precomputed normals](https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz) are not publicly available any longer. We are trying to reach the authors. Normal extraction is optional for now.

## Prepare dataset
```bash
# general usage
nicr_sa_prepare_dataset nyuv2 \
    /path/where/to/store/nyuv2
```
