# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from ..pytorch import Cityscapes
from ..pytorch import COCO
from ..pytorch import Hypersim
from ..pytorch import NYUv2
from ..pytorch import ScanNet
from ..pytorch import SceneNetRGBD
from ..pytorch import SUNRGBD

from .utils import register_dataset_to_d2


# Automatically register all datasets with some default keys so that they
# are available through Detectron2's DatasetCatalog.
# Note that they are just registered so that the stats can be access.
# For using the dataset, the 'set_dataset_path' function should be called first.
# Moreover, we currently do not load the 'depth' sample key for any dataset.
# If your interested in another sample key, remove the dataset and call
# 'register_dataset_to_d2' yourself.
register_dataset_to_d2(
    name_prefix='cityscapes',
    dataset_class=Cityscapes,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
register_dataset_to_d2(
    name_prefix='coco',
    dataset_class=COCO,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
register_dataset_to_d2(
    name_prefix='hypersim',
    dataset_class=Hypersim,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
register_dataset_to_d2(
    name_prefix='nyuv2',
    dataset_class=NYUv2,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
register_dataset_to_d2(
    name_prefix='scannet',
    dataset_class=ScanNet,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
register_dataset_to_d2(
    name_prefix='scenenetrgbd',
    dataset_class=SceneNetRGBD,
    sample_keys=('identifier', 'rgb', 'semantic')
)
register_dataset_to_d2(
    name_prefix='sunrgbd',
    dataset_class=SUNRGBD,
    sample_keys=('identifier', 'rgb', 'semantic', 'instance')
)
