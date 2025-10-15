# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

from ._depth_dataset import DepthDataset
from ._rgb_dataset import RGBDataset


class RGBDDataset(RGBDataset, DepthDataset):
    def __init__(
        self,
        depth_mode: str = 'raw',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )
