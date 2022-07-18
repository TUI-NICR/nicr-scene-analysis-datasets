# NICR Scene Analysis Datasets

This repository contains code to prepare and use common datasets for scene analysis tasks.

> Note that this package is used in ongoing research projects and will be extended and maintained as needed.

Currently, this packages features the following datasets and annotations:

| Dataset                                                               | Updated/Tested |   Type    | Semantic | Instance | Orientations |  Scene   |  Normal  | 3D Boxes | Extrinsics | Intrinsics |
|:----------------------------------------------------------------------|:--------------:|:---------:|:--------:|:--------:|:------------:|:--------:|:--------:|:--------:|:----------:|:----------:|
| [COCO](https://cocodataset.org/#home)                                 | v030/v040      | RGB       | &#10003; | &#10003; |              |          |          |          |            |            |
| [Cityscapes](https://www.cityscapes-dataset.com/)                     | v030/v040      | RGB-D*    | &#10003; |          |              |          |          |          |            |            |
| [Hypersim](https://machinelearning.apple.com/research/hypersim)       | v030/v040      | RGB-D     | &#10003; | &#10003; | (&#10003;)** | &#10003; | &#10003; | &#10003; | &#10003;   | &#10003;   |
| [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)     | v030/v040      | RGB-D     | &#10003; | &#10003; | &#10003;***  | &#10003; | &#10003; |          |            |            |
| [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html)  | v030/v040      | RGB-D     | &#10003; |          |              |          |          |          |            |            |
| [SUNRGB-D](https://rgbd.cs.princeton.edu/)                            | v030/v040      | RGB-D     | &#10003; | &#10003; |   &#10003;   | &#10003; |          | &#10003; | &#10003;   | &#10003;   |

\* Both depth and disparity are available.  
\*\* Orientations are available but not consistent for instances of the same semantic class (see Hypersim).  
\*\*\* Annotated by hand in 3D for instances of some relevant semantic classes.  

## License and Citations
The source code is published under Apache 2.0 license, see [license file](LICENSE) for details.

If you use the source code, please cite the papers related to this work:

**Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments** ([arXiv](https://arxiv.org/abs/2207.04526)):
>Seichter, D., Fischedick, B., Köhler, M., Gross, H.-M.
*Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments*,
to appear in IEEE International Joint Conference on Neural Networks (IJCNN), 2022.

```bibtex
@article{emsanet2022,
title={Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments},
author={Seichter, Daniel and Fischedick, S{\"o}hnke and K{\"o}hler, Mona and Gross, Horst-Michael},
journal={arXiv preprint arXiv:2207.04526},
year={2022}
}
```
Note that the preprint was accepted to be published in IEEE International Joint Conference on Neural Networks (IJCNN) 2022.

**Efficient and Robust Semantic Mapping for Indoor Environments** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9812205), [arXiv](https://arxiv.org/pdf/2203.05836.pdf)):
>Seichter, D., Langer, P., Wengefeld, T., Lewandowski, B., Höchemer, D., Gross, H.-M.
*Efficient and Robust Semantic Mapping for Indoor Environments*
in IEEE International Conference on Robotics and Automation (ICRA), pp. 9221-9227, 2022.

```bibtex
@inproceedings{semanticndtmapping2022icra,
  author={Seichter, Daniel and Langer, Patrick and Wengefeld, Tim and Lewandowski, Benjamin and H{\"o}chemer, Dominik and Gross, Horst-Michael},
  booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
  title={Efficient and Robust Semantic Mapping for Indoor Environments}, 
  year={2022},
  pages={9221-9227},
  doi={10.1109/ICRA46639.2022.9812205}}
```

**Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9561675),  [arXiv](https://arxiv.org/pdf/2011.06961.pdf)):
>Seichter, D., Köhler, M., Lewandowski, B., Wengefeld T., Gross, H.-M.
*Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis*
in IEEE International Conference on Robotics and Automation (ICRA), pp. 13525-13531, 2021.

```bibtex
@inproceedings{esanet2021icra,
title={Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis},
author={Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
year={2021},
pages={13525-13531}
}
```

## Installation
```bash
git clone https://github.com/TUI-NICR/nicr-scene-analysis-datasets.git
cd /path/to/this/repository

# for preparation and usage
pip install .[withpreparation] [--user]

# for usage only
pip install . [--user]
```

## Prepare dataset

Please follow the instructions given in the respective dataset folder to prepare the datasets.
- [Cityscapes](nicr_scene_analysis_datasets/datasets/cityscapes)
- [Hypersim](nicr_scene_analysis_datasets/datasets/hypersim)
- [NYUv2](nicr_scene_analysis_datasets/datasets/nyuv2)
- [SceneNet RGB-D](nicr_scene_analysis_datasets/datasets/scenenetrgbd)
- [SUNRGB-D](nicr_scene_analysis_datasets/datasets/sunrgbd)


## Usage
In the following, an example for Hypersim is given.

With `sample_keys` you can specify what a sample of your dataset should contain.
```python
from nicr_scene_analysis_datasets import Hypersim

dataset_path = '/path/to/prepared/hypersim'
sample_keys = (
    'identifier',    # helps to know afterwards which sample was loaded
    'rgb', 'depth',    # camera data
    'rgb_intrinsics', 'depth_intrinsics', 'extrinsics',    # camera parameters
    'semantic', 'instance', 'orientations', '3d_boxes', 'scene', 'normal'    # tasks
)
dataset_train = Hypersim(
    dataset_path=dataset_path,
    split='train',
    sample_keys=sample_keys
)

# finally, you can iterate over the dataset
for sample in dataset_train:
    print(sample)

# note: for usage along with pytorch, simply change the import above to
from nicr_scene_analysis_datasets.pytorch import Hypersim
```

## Detectron2 usage (experimental support)
The following example shows how a dataset, e.g., Hypersim, can be loaded through the [Detectron2](https://github.com/facebookresearch/detectron2) API.

First, the API must be imported:
```python
# the import automatically registers all datasets to d2
from nicr_scene_analysis_datasets import d2 as nicr_d2
```

This import registers all available datasets to the detectron2's [DatasetCatalog](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.DatasetCatalog) and [MetadataCatalog](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.MetadataCatalog) logic.
Note that the Metadata can already be accessed (see below).
However, the dataset path might be incorrect (i.e., when the dataset is not in detectron2's default directory).
This path can be set by following function:
```python
# set the path for the dataset, so that d2 can use it
# note, dataset_path must point to the actual dataset (e.g. ../datasets/hypersim)
# this limits the API currently because only one dataset can be used at a time
nicr_d2.set_dataset_path(dataset_path)
```
After doing this, the dataset can be used by detectron2:
```python
from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

# get the dataset config
dataset_config = MetadataCatalog.get('hypersim_test').dataset_config

# get the dataset for usage
dataset = DatasetCatalog.get('hypersim_test')
```
Note that the name is always a combination of the dataset's name and the split, which should be used.

### Further Important Remarks
The logic of our dataset implementation is different from the logic of detectron2.
While we use classes that already provide data loaded from file in the correct format, the default [`DatasetMappper`](https://detectron2.readthedocs.io/en/latest/modules/data.html#detectron2.data.DatasetMapper) of detectron2 expects paths to files that should be loaded later on.

To handle this, a special `NICRSceneAnalysisDatasetMapper` is provided to replace the default `DatasetMapper`.
An example for doing that is given below:
```python
# use the mapper
data_mapper = nicr_d2.NICRSceneAnalysisDatasetMapper(dataset_config)

# pass data_mapper (in your custom Trainer class) to
# build_detection_train_loader / build_detection_test_loader
```
In certain situations, multiple mappers are required (e.g., a target generator for panoptic segmentation, which combines semantic and instance to panoptic).
For this use case, we further provide a helper class, which can be used to chain multiple mappers.
```python
chained_mapper = nicr_d2.NICRChainedDatasetMapper(
    [data_mapper, panoptic_mapper]
)
```
For further details, we refer to the usage in our [EMSANet repository](https://github.com/TUI-NICR/EMSANet/blob/main//external).

The dataset can be used as an iterator (detectron2 usually does this) and can then be mapped with the custom mappers to generate the correct layout of the data.

## Changelog
__Version 0.4.0 (July 15, 2022)__
- *no dataset preparation related changes*
- Hypersim: [BREAKING CHANGE TO V030] enable fixed depth stats
- add experimental support for Detectron2
- semantic_use_nyuv2_colors as option in SUNRGBD constructor
- changed license to Apache 2.0
