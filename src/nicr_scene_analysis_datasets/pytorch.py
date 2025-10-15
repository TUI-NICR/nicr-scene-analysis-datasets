# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Callable, Type, Union

from torch.utils.data import Dataset

from .auxiliary_data import wrap_dataset_with_auxiliary_data
from .dataset_base import KNOWN_CLASS_WEIGHTINGS  # noqa: F401
from .dataset_base._base_dataset import DatasetBase


class _PytorchDatasetWrapper(DatasetBase, Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprocessor = None

    @property
    def transform(self) -> Union[Callable, None]:
        # just to be compatible with VisionDataset from torchvision
        return self.preprocessor

    @transform.setter
    def transform(self, value: Union[Callable, None]):
        # just to be compatible with VisionDataset from torchvision
        self.preprocessor = value

    @property
    def preprocessor(self) -> Union[Callable, None]:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Union[Callable, None]):
        self._preprocessor = value

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)

        # apply preprocessing
        if self._preprocessor is not None:
            sample = self._preprocessor(sample)

        return sample


from . import ADE20K as _ADE20K
from . import COCO as _COCO
from . import Cityscapes as _Cityscapes
from . import Hypersim as _Hypersim
from . import NYUv2 as _NYUv2
from . import ScanNet as _ScanNet
from . import SceneNetRGBD as _SceneNetRGBD
from . import SUNRGBD as _SUNRGBD


class Cityscapes(_Cityscapes, _PytorchDatasetWrapper):
    pass


class COCO(_COCO, _PytorchDatasetWrapper):
    pass


class Hypersim(_Hypersim, _PytorchDatasetWrapper):
    pass


class NYUv2(_NYUv2, _PytorchDatasetWrapper):
    pass


class ScanNet(_ScanNet, _PytorchDatasetWrapper):
    pass


class SceneNetRGBD(_SceneNetRGBD, _PytorchDatasetWrapper):
    pass


class SUNRGBD(_SUNRGBD, _PytorchDatasetWrapper):
    pass


class ADE20K(_ADE20K, _PytorchDatasetWrapper):
    pass


from .dataset_base import ConcatDataset as _ConcatDataset


class _PytorchConcatDatasetWrapper:
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprocessor = None

    @property
    def transform(self) -> Union[Callable, None]:
        # just to be compatible with VisionDataset from torchvision
        return self.preprocessor

    @transform.setter
    def transform(self, value: Union[Callable, None]):
        # just to be compatible with VisionDataset from torchvision
        self.preprocessor = value

    @property
    def preprocessor(self) -> Union[Callable, None]:
        return self._preprocessor

    @preprocessor.setter
    def preprocessor(self, value: Union[Callable, None]):
        self._preprocessor = value
        # apply preprocessor to all datasets
        for ds in self._datasets:
            ds.preprocessor = value


class ConcatDataset(_ConcatDataset, _PytorchConcatDatasetWrapper):
    pass


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

    original_dataset = _DATASETS[name]
    if with_auxiliary_data:
        current_dataset = wrap_dataset_with_auxiliary_data(original_dataset)
    else:
        current_dataset = original_dataset

    return current_dataset
