# NYUv2 dataset

The NYU-Depth V2 dataset is comprised of video sequences from a variety of indoor scenes as recorded by both the RGB and Depth cameras from the Microsoft Kinect.
It contains 1449 densely labeled pairs of aligned RGB and depth images.

For more details, see: [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)

## Prepare dataset
```bash
# general usage
python -m nicr_scene_analysis_datasets.datasets.nyuv2.prepare_dataset \
    /path/where/to/store/nyuv2
```
