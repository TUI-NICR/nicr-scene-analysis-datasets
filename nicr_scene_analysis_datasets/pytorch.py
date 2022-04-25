# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Callable, Type, Union

from torch.utils.data import Dataset

from .dataset_base._base_dataset import DatasetBase
from .dataset_base import KNOWN_CLASS_WEIGHTINGS    # noqa: F401


class _PytorchDatasetWrapper(DatasetBase, Dataset):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._preprocessor = None

    @property
    def transform(self) -> Union[Callable, None]:
        # just to be compatible with VisionDataset from torchvision
        return self._preprocessor

    @transform.setter
    def transform(self, value: Union[Callable, None]):
        # just to be compatible with VisionDataset from torchvision
        self._preprocessor = value

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


from . import COCO as _COCO
from . import SUNRGBD as _SUNRGBD
from . import Cityscapes as _Cityscapes
from . import Hypersim as _Hypersim
from . import NYUv2 as _NYUv2
from . import SceneNetRGBD as _SceneNetRGBD


class Cityscapes(_Cityscapes, _PytorchDatasetWrapper):
    pass


class COCO(_COCO, _PytorchDatasetWrapper):
    pass


class Hypersim(_Hypersim, _PytorchDatasetWrapper):
    pass


class NYUv2(_NYUv2, _PytorchDatasetWrapper):
    pass


class SceneNetRGBD(_SceneNetRGBD, _PytorchDatasetWrapper):
    pass


class SUNRGBD(_SUNRGBD, _PytorchDatasetWrapper):
    pass


_DATASETS = {
    'cityscapes': Cityscapes,
    'coco': COCO,
    'hypersim': Hypersim,
    'nyuv2': NYUv2,
    'scenenetrgbd': SceneNetRGBD,
    'sunrgbd': SUNRGBD,
}
KNOWN_DATASETS = tuple(_DATASETS.keys())

DatasetType = Union[Cityscapes, COCO, Hypersim, NYUv2, SceneNetRGBD,
                    SUNRGBD]


def get_dataset_class(name: str) -> Type[DatasetType]:
    name = name.lower()
    if name not in KNOWN_DATASETS:
        raise ValueError(f"Unknown dataset: '{name}'")

    return _DATASETS[name]
