# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Type, Union


from .utils.imports import install_nicr_scene_analysis_datasets_dependency_import_hooks

install_nicr_scene_analysis_datasets_dependency_import_hooks()


from .auxiliary_data import wrap_dataset_with_auxiliary_data
from .dataset_base import KNOWN_CLASS_WEIGHTINGS
from .dataset_base import ConcatDataset
from .datasets.ade20k.dataset import ADE20K
from .datasets.cityscapes.dataset import Cityscapes
from .datasets.coco.dataset import COCO
from .datasets.hypersim.dataset import Hypersim
from .datasets.nyuv2.dataset import NYUv2
from .datasets.scannet.dataset import ScanNet
from .datasets.scenenetrgbd.dataset import SceneNetRGBD
from .datasets.sunrgbd.dataset import SUNRGBD


_DATASETS = {
    'ade20k': ADE20K,
    'cityscapes': Cityscapes,
    'coco': COCO,
    'hypersim': Hypersim,
    'nyuv2': NYUv2,
    'scannet': ScanNet,
    'scenenetrgbd': SceneNetRGBD,
    'sunrgbd': SUNRGBD,
}
KNOWN_DATASETS = tuple(_DATASETS.keys())

DatasetType = Union[
    ADE20K,
    Cityscapes,
    COCO,
    Hypersim,
    NYUv2,
    ScanNet,
    SceneNetRGBD,
    SUNRGBD,
    ConcatDataset
]


def get_dataset_class(name: str, with_auxiliary_data: bool = False) -> Type[DatasetType]:
    name = name.lower()
    if name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'")
    original_dataset_class = _DATASETS[name]
    if with_auxiliary_data:
        current_dataset_class = \
            wrap_dataset_with_auxiliary_data(original_dataset_class)
    else:
        current_dataset_class = original_dataset_class

    return current_dataset_class


from .version import __version__
