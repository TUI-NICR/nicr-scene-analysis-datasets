# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

import abc

import numpy as np

from ._annotation import IntrinsicCameraParametersNormalized
from ._base_dataset import DatasetBase


class RGBDataset(DatasetBase):
    def __init__(
        self,
        sample_keys: Tuple[str] = ('rgb', 'semantic'),
        use_cache: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

    @abc.abstractmethod
    def _load_rgb(self, idx) -> np.array:
        pass

    def _load_rgb_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        # so far, only few datasets support intrinsics, thus, we define a
        # default here
        raise NotImplementedError()
