# NICR Scene Analysis Datasets
This repository contains code to prepare and use common datasets for scene analysis tasks.

> Note that this package is used in ongoing research projects and will be extended and maintained as needed.

Currently, this packages features the following datasets and annotations:

| Dataset                                                               | Updated/Tested |   Type    | Semantic | Instance | Orientations |  Scene   |  Normal  | 3D Boxes | Extrinsics | Intrinsics |
|:----------------------------------------------------------------------|:--------------:|:---------:|:--------:|:--------:|:------------:|:--------:|:--------:|:--------:|:----------:|:----------:|
| [COCO](https://cocodataset.org/#home)                                 | v030/v031      | RGB       | &#10003; | &#10003; |              |          |          |          |            |            |
| [Cityscapes](https://www.cityscapes-dataset.com/)                     | v030/v031      | RGB-D*    | &#10003; |          |              |          |          |          |            |            |
| [Hypersim](https://machinelearning.apple.com/research/hypersim)       | v030/v031      | RGB-D     | &#10003; | &#10003; | (&#10003;)** | &#10003; | &#10003; | &#10003; | &#10003;   | &#10003;   |
| [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)     | v030/v031      | RGB-D     | &#10003; | &#10003; | &#10003;***  | &#10003; | &#10003; |          |            |            |
| [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html)  | v030/v031      | RGB-D     | &#10003; |          |              |          |          |          |            |            |
| [SUNRGB-D](https://rgbd.cs.princeton.edu/)                            | v030/v031      | RGB-D     | &#10003; | &#10003; |   &#10003;   | &#10003; |          | &#10003; | &#10003;   | &#10003;   |

\*Both depth and disparity are available.  
\*\*Orientations are available but not consistent for instances of the same semantic class (see Hypersim).  
\*\*\*Annotated by hand in 3D for instances of some relevant semantic classes.  

## License and Citations
The source code is published under BSD 3-Clause license, see [license file](LICENSE) for details.

If you use the source code, please cite the papers related to this work:

**Efficient and Robust Semantic Mapping for Indoor Environments** ([arXiv](https://arxiv.org/pdf/2203.05836.pdf)):
>Seichter, D., Langer, P., Wengefeld, T., Lewandowski, B., Höchemer, D., Gross, H.-M.
*Efficient and Robust Semantic Mapping for Indoor Environments*
to appear in IEEE International Conference on Robotics and Automation (ICRA), 2022.

```bibtex
@article{semanticndtmapping2022arXiv,
title={Efficient and Robust Semantic Mapping for Indoor Environments},
author={Seichter, Daniel and Langer, Patrick and Wengefeld, Tim and Lewandowski, Benjamin and H{\"o}chemer, Dominik and Gross, Horst-Michael},
journal={arXiv preprint arXiv:2203.05836},
year={2022}
}
```
Note that the preprint was accepted to be published in IEEE International Conference on Robotics and Automation (ICRA).

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
- [Cityscapes](nicr_scene_analysis_datasets/cityscapes)
- [Hypersim](nicr_scene_analysis_datasets/hypersim)
- [NYUv2](nicr_scene_analysis_datasets/nyuv2)
- [SceneNet RGB-D](nicr_scene_analysis_datasets/scenenetrgbd)
- [SUNRGB-D](nicr_scene_analysis_datasets/sunrgbd)


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
