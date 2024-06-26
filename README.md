# NICR Scene Analysis Datasets

This repository contains code to prepare and use common datasets for scene analysis tasks.

> Note that this package is used in ongoing research projects and will be extended and maintained as needed.

Currently, this packages features the following datasets and annotations:

| Dataset                                                               | Updated/Tested |   Type    | Semantic | Instance |  Orientations  |  Scene   |       Normal       | 3D Boxes | Extrinsics | Intrinsics |
|:----------------------------------------------------------------------|:--------------:|:---------:|:--------:|:--------:|:--------------:|:--------:|:------------------:|:--------:|:----------:|:----------:|
| [COCO](https://cocodataset.org/#home)                                 | v030/v070      | RGB       | &#10003; | &#10003; |                |          |                    |          |            |            |
| [Cityscapes](https://www.cityscapes-dataset.com/)                     | v050/v070      | RGB-D\*   | &#10003; | &#10003; |                |          |                    |          |            |            |
| [Hypersim](https://machinelearning.apple.com/research/hypersim)       | v052/v070      | RGB-D     | &#10003; | &#10003; | (&#10003;)\*\* | &#10003; | &#10003;           | &#10003; | &#10003;   | &#10003;   |
| [NYUv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)     | v070/v070      | RGB-D     | &#10003; | &#10003; | &#10003;\*\*\* | &#10003; | (&#10003;)\*\*\*\* |          |            |            |
| [ScanNet](http://www.scan-net.org/)                                   | v051/v070      | RGB-D     | &#10003; | &#10003; |                | &#10003; |                    |          | &#10003;   | &#10003;   |
| [SceneNet RGB-D](https://robotvault.bitbucket.io/scenenet-rgbd.html)  | v054/v070      | RGB-D     | &#10003; | &#10003; |                | &#10003; |                    |          |            |            |
| [SUNRGB-D](https://rgbd.cs.princeton.edu/)                            | v060/v070      | RGB-D     | &#10003; | &#10003; |   &#10003;     | &#10003; |                    | &#10003; | &#10003;   | &#10003;   |

\* Both depth and disparity are available.  
\*\* Orientations are available but not consistent for instances within a semantic class (see [Hypersim](nicr_scene_analysis_datasets/datasets/hypersim)).  
\*\*\* Annotated by hand in 3D for instances of some relevant semantic classes.  
\*\*\*\* As of Nov 2022, [precomputed normals](https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz) are not publicly available any longer. We are trying to reach the authors.

## License and Citations

The source code is published under Apache 2.0 license, see [license file](LICENSE) for details.

If you use the source code, please cite the paper related to your work:

---

**PanopticNDT: Efficient and Robust Panoptic Mapping** ([IEEE Xplore](https://ieeexplore.ieee.org/document/10342137), [arXiv](https://arxiv.org/abs/2309.13635) (with appendix and some minor fixes)):
> Seichter, D., Stephan, B., Fischedick, S. B., Müller, S., Rabes, L., Gross, H.-M.
*PanopticNDT: Efficient and Robust Panoptic Mapping*,
in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2023.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{panopticndt2023iros,
  title     = {{PanopticNDT: Efficient and Robust Panoptic Mapping}},
  author    = {Seichter, Daniel and Stephan, Benedict and Fischedick, S{\"o}hnke Benedikt and  Mueller, Steffen and Rabes, Leonard and Gross, Horst-Michael},
  booktitle = {IEEE/RSJ Int. Conf. on Intelligent Robots and Systems (IROS)},
  year      = {2023}
}
```

</details>

---

**Efficient Multi-Task Scene Analysis with RGB-D Transformers** ([IEEE Xplore](https://ieeexplore.ieee.org/document/10191977), [arXiv](https://arxiv.org/abs/2306.05242)):
> Fischedick, S., Seichter, D., Schmidt, R., Rabes, L., Gross, H.-M.
*Efficient Multi-Task Scene Analysis with RGB-D Transformers*,
in IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1-10, 2023.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{emsaformer2023ijcnn,
  title     = {{Efficient Multi-Task Scene Analysis with RGB-D Transformers}},
  author    = {Fischedick, S{\"o}hnke and Seichter, Daniel and Schmidt, Robin and Rabes, Leonard and Gross, Horst-Michael},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2023},
  pages     = {1-10},
  doi       = {10.1109/IJCNN54540.2023.10191977}
}
```

</details>

> Use `--instances-version emsanet` when preparing the SUNRGB-D dataset with `nicr-scene-analysis-datasets` to reproduce reported results.

---

**Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9892852), [arXiv](https://arxiv.org/abs/2207.04526)):
> Seichter, D., Fischedick, S., Köhler, M., Gross, H.-M.
*Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments*,
in IEEE International Joint Conference on Neural Networks (IJCNN), pp. 1-10, 2022.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{emsanet2022ijcnn,
  title     = {{Efficient Multi-Task RGB-D Scene Analysis for Indoor Environments}},
  author    = {Seichter, Daniel and Fischedick, S{\"o}hnke and K{\"o}hler, Mona and Gross, Horst-Michael},
  booktitle = {IEEE International Joint Conference on Neural Networks (IJCNN)},
  year      = {2022},
  pages     = {1-10},
  doi       = {10.1109/IJCNN55064.2022.9892852}
}
```

</details>

> Use `--instances-version emsanet` when preparing the SUNRGB-D dataset with `nicr-scene-analysis-datasets` to reproduce reported results.

---

**Efficient and Robust Semantic Mapping for Indoor Environments** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9812205), [arXiv](https://arxiv.org/pdf/2203.05836.pdf)):
>Seichter, D., Langer, P., Wengefeld, T., Lewandowski, B., Höchemer, D., Gross, H.-M.
*Efficient and Robust Semantic Mapping for Indoor Environments*
in IEEE International Conference on Robotics and Automation (ICRA), pp. 9221-9227, 2022.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{semanticndtmapping2022icra,
  title     = {{Efficient and Robust Semantic Mapping for Indoor Environments}},
  author    = {Seichter, Daniel and Langer, Patrick and Wengefeld, Tim and Lewandowski, Benjamin and H{\"o}chemer, Dominik and Gross, Horst-Michael},
  booktitle = {2022 International Conference on Robotics and Automation (ICRA)},
  year      = {2022},
  pages     = {9221-9227},
  doi       = {10.1109/ICRA46639.2022.9812205}
}
```

</details>

---

**Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis** ([IEEE Xplore](https://ieeexplore.ieee.org/document/9561675),  [arXiv](https://arxiv.org/pdf/2011.06961.pdf)):
>Seichter, D., Köhler, M., Lewandowski, B., Wengefeld T., Gross, H.-M.
*Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis*
in IEEE International Conference on Robotics and Automation (ICRA), pp. 13525-13531, 2021.

<details>
<summary>BibTeX</summary>

```bibtex
@inproceedings{esanet2021icra,
  title     = {{Efficient RGB-D Semantic Segmentation for Indoor Scene Analysis}},
  author    = {Seichter, Daniel and K{\"o}hler, Mona and Lewandowski, Benjamin and Wengefeld, Tim and Gross, Horst-Michael},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2021},
  pages     = {13525-13531},
  doi       = {10.1109/ICRA48506.2021.9561675}
}
```

</details>

---


## Installation

```bash
git clone https://github.com/TUI-NICR/nicr-scene-analysis-datasets.git
cd /path/to/this/repository

# full installation:
# - withpreparation: requirements for preparing the datasets
# - with3d: requirements for 3D processing (see entry points below)
python -m pip install -e "./[withpreparation,with3d]"

# for usage only
python -m pip install -e "./"
```

## Prepare dataset

Please follow the instructions given in the respective dataset folder to prepare the datasets.

- [Cityscapes](nicr_scene_analysis_datasets/datasets/cityscapes)
- [COCO](nicr_scene_analysis_datasets/datasets/coco)
- [Hypersim](nicr_scene_analysis_datasets/datasets/hypersim)
- [NYUv2](nicr_scene_analysis_datasets/datasets/nyuv2)
- [ScanNet](nicr_scene_analysis_datasets/datasets/scannet)
- [SceneNet RGB-D](nicr_scene_analysis_datasets/datasets/scenenetrgbd)
- [SUNRGB-D](nicr_scene_analysis_datasets/datasets/sunrgbd)


## Usage
### Entry points

We provide several command-line entry points for common tasks:
  - `nicr_sa_prepare_dataset`: prepare a dataset for usage
  - `nicr_sa_prepare_labeled_point_clouds`: create labeled point clouds as ply files similar to ScanNet benchmark
  - `nicr_sa_depth_viewer`: viewer for depth images
  - `nicr_sa_semantic_instance_viewer`: viewer for semantic and instance (and panoptic) annotations
  - `nicr_sa_labeled_pc_viewer`: viewer for labeled point clouds

### How to use a dataset in a pipeline
In the following, an example for Hypersim is given.

First, specify the dataset path:

```python
dataset_path = '/path/to/prepared/hypersim'
```

With `sample_keys` you can specify what a sample of your dataset should contain.

```python
from nicr_scene_analysis_datasets import Hypersim

sample_keys = (
    'identifier',    # helps to know afterwards which sample was loaded
    'rgb', 'depth',    # camera data
    'rgb_intrinsics', 'depth_intrinsics', 'extrinsics',    # camera parameters
    'semantic', 'instance', 'orientations', '3d_boxes', 'scene', 'normal'    # annotations
)

# list available sample keys
print(Hypersim.sample_keys.get_available_sample_keys(split='train'))

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
For further details, we refer to the usage in our [EMSANet repository](https://github.com/TUI-NICR/EMSANet/blob/main/external).

The dataset can be used as an iterator (detectron2 usually does this) and can then be mapped with the custom mappers to generate the correct layout of the data.

## Changelog

**Version 0.7.0 (Jun 26, 2024)**
- allow extracting both instance annotation versions for SUNRGB-D with a 
  single version of the dataset package: 'emsanet' and 'panopticndt', use 
  'emsanet' to reproduce results reported in EMSANet or EMSAFormer paper, and 
  'panopticndt' for follow-up papers
- fix for missing creation meta files 
- NYUv2: do not create outdated `class_names_*.txt` and `class_colors_*.txt`
  files anymore

**Version 0.6.1 (Dec 5, 2023)**
- force 'instance' sample key to always be of dtype uint16
- force 'semantic' sample key to always be of dtype uint8 (i.e., for Cityscapes, 
  COCO, Hypersim, NYUv2 (13+40 classes), ScanNet (20, 40, 200), SceneNet RGB-D 
  and SUNRGB-D) or uint16 (i.e., for NYUv2 (894 classes), ScanNet (549 classes))
- add test to verify the dtypes of each dataset
- remove 'semantic_n_classes' argument from SceneNet RGB-D and set it to '13'
- fix version format and parsing to be PEP440 compliant (required for more 
  recent packaging versions)
- fix `--max-z-value` in `nicr_sa_labeled_pc_viewer` to work with additionally 
  given label files (`*-label-filepath`) as well
- this version was an internal release only

**Version 0.6.0 (Sep 26, 2023)**
- SUNRGB-D:
  - refactor and update instance creation from 3D boxes: annotations for
    instances, boxes, and (instance) orientations have changed:
    - ignore semantic stuff classes and void while matching boxes and point
      clouds
    - enhanced matching for similar classes (e.g., table <-> desk)
    - resulting annotations feature a lot of more instances
    - if you use the new instance annotations, please refer to this version of
      the dataset as *SUNRGB-D (PanopticNDT version)* and to previous versions
      with instance information as *SUNRGB-D (EMSANet version)*
  - **note, version 0.6.0 is NOT compatible with previous versions, you will
    get deviating results when applying EMSANet or EMSAFormer**
- Hypersim:
  - add more notes/comments for blacklisted scenes/camera trajectories
  - do not use orientations by default (annotations provided by the dataset are
    not consistent for instances within a semantic class), i.e., return an
    empty OrientationDict for all samples unless `orientations_use` is enabled
- `nicr_sa_labeled_pc_viewer`: add `--max-z-value` argument to limit the
  maximum z-value for the point cloud viewer
- `nicr_sa_depth_viewer`: add `image_nonzero` mode for scaling depth values
  (`--mode` argument)
- MIRA readers:
  - add instance meta stuff
  - terminate MIRA in a softer way (do not send SIGKILL, send SIGINT instead
    and wait before sending again) to force propper termination (and profile
    creation)
- some test fixes

**Version 0.5.6 (Sep 26, 2023)**
- `ConcatDataset`:
  - add `datasets` property to get the list of currently active datasets
  - implement `load` to load a specific sample key for a given index (e.g.,
    `load('rgb', 0)` loads the rgb image of the main dataset at index 0)
- update citations
- tests: some fixes, skip testing with Python 3.6, add testing with Python 3.11

**Version 0.5.5 (Sep 08, 2023)**
- make `creation_meta.json` optional to enable loading old datasets
- some minor fixes (typos, ...)

**Version 0.5.4 (Jun 07, 2023)**
- SUNRGB-D:
  - fix for `depth_force_mm=True`:
    - divide by 8 (shift by 3 to right) instead of divide by 10
    - updated depth stats
    - for more details, see notes in [nicr_scene_analysis_datasets/datasets/sunrgbd/dataset.py](nicr_scene_analysis_datasets/datasets/sunrgbd/dataset.py#L213)
    - note, for `depth_force_mm=False`, nothing changed, everything is as before (EMSANet / EMSAFormer)
- SceneNet RGB-D: add support for instances and scene classes
- add `identifier2idx()` to base dataset class to search for samples by identifier

**Version 0.5.3 (Mar 31, 2023)**
- *no dataset preparation related changes*
- minor changes to `nicr_sa_prepare_labeled_point_clouds` and `nicr_sa_labeled_pc_viewer`

**Version 0.5.2 (Mar 28, 2023)**
- Hypersim: change instance encoding: do not view G and B channel as uint16 use bit shifting instead
- add new scripts and update entry points:
  - `nicr_sa_prepare_dataset`: prepare a dataset (replaces `python -m ...` calls)
  - `nicr_sa_prepare_labeled_point_clouds`: create labeled point clouds as ply files similar to ScanNet benchmark
  - `nicr_sa_depth_viewer`: viewer for depth images
  - `nicr_sa_semantic_instance_viewer`: viewer for semantic and instance annotations
  - `nicr_sa_labeled_pc_viewer`: viewer for labeled point clouds

**Version 0.5.1 (Mar 01, 2023)**
- refactor MIRA reader to support multiple datasets, create an abstract base class
- ScanNet:
  - blacklist broken frames due to invalid extrinsic parameters (see datasets/scannet/scannet.py)
- Hypersim:
  - IMPORTANT: version 0.5.1 is not compatible with ealier versions of the dataset
  - convert all data to standard pinhole camera projections (without tilt-shift parameters, see datasets/hypersim/prepare_dataset.py for details)
  - convert intrinsic parameters to standard format for usage in MIRA or ROS
  - update depth train stats due to new data

**Version 0.5.0 (Jan 04, 2023)**
- add depth viewer and semantic-instance viewer command-line entrypoints
- add support for ScanNet dataset
- add ScanNet MIRA reader
- add instance support to Hypersim MIRA reader
- add static `get_available_sample_keys` to all datasets
- add `depth_force_mm` to SUNRGB-D dataset class (same depth scale as for Hypersim, NYUv2, ScanNet, and SceneNet RGB-D)
- add `ConcatDataset` and `pytorch.ConcatDataset` to concatenate multiple datasets
- add `cameras` argument to constructors to apply a static camera filter
- add instance support to Cityscapes dataset

**Version 0.4.1 (Nov 12, 2022)**
- *no dataset preparation related changes*
- make normal extraction for NYUv2 dataset optional as the [precomputed normals](https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz) are not publicly available any longer

**Version 0.4.0 (July 15, 2022)**
- *no dataset preparation related changes*
- Hypersim: [BREAKING CHANGE TO V030] enable fixed depth stats
- add experimental support for Detectron2
- semantic_use_nyuv2_colors as option in SUNRGBD constructor
- changed license to Apache 2.0
