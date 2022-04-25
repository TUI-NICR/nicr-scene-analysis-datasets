# COCO Dataset

COCO is a large-scale object detection, segmentation, and captioning dataset.
It contains over 200.000 labeled images with 80 object and 91 stuff categories
for panoptic segmentation.

For more details, see: [COCO dataset](https://cocodataset.org/#home)

## Prepare dataset
```bash
# general usage
python -m nicr_scene_analysis_datasets.datasets.coco.prepare_dataset \
    /path/where/to/store/coco/
```
