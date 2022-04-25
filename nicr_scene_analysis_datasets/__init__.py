# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Type, Union

from .dataset_base import KNOWN_CLASS_WEIGHTINGS
from .datasets.cityscapes.dataset import Cityscapes
from .datasets.coco.dataset import COCO
from .datasets.hypersim.dataset import Hypersim
from .datasets.nyuv2.dataset import NYUv2
from .datasets.scenenetrgbd.dataset import SceneNetRGBD
from .datasets.sunrgbd.dataset import SUNRGBD

_DATASETS = {
    'cityscapes': Cityscapes,
    'coco': COCO,
    'hypersim': Hypersim,
    'nyuv2': NYUv2,
    'scenenetrgbd': SceneNetRGBD,
    'sunrgbd': SUNRGBD,
}
KNOWN_DATASETS = tuple(_DATASETS.keys())

DatasetType = Union[Cityscapes, COCO, Hypersim, NYUv2, SceneNetRGBD, SUNRGBD]


def get_dataset_class(name: str) -> Type[DatasetType]:
    name = name.lower()
    if name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'")

    return _DATASETS[name]


from .version import __version__
