# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Optional, Tuple, Union

import abc
import os

import numpy as np
import torch

DEFAULT_CACHE_BASEPATH = os.getenv(
    "DEFAULT_CACHE_BASEPATH",
    "~/.cache/nicr_scene_analysis_datasets/auxiliary_data"
)


class AuxiliaryDataEstimatorBase(abc.ABC):
    NAME: str

    def __init__(
        self,
        device: Union[str, torch.device] = 'cpu',
        max_pixels: Optional[int] = None,
        input_interpolation: str = 'bilinear',
        auto_set_up: bool = True,
        cache_basepath: Optional[str] = None,  # None -> DEFAULT_CACHE_BASEPATH
    ) -> None:
        self._device = device
        self._max_pixels = max_pixels
        self._input_interpolation = input_interpolation

        # cache path
        self._cache_basepath = cache_basepath
        if self._cache_basepath is None:
            self._cache_basepath = os.path.expanduser(DEFAULT_CACHE_BASEPATH)

        self._cache_path = os.path.join(self._cache_basepath, self.NAME)
        os.makedirs(self._cache_path, exist_ok=True)

        if auto_set_up:
            self.set_up_estimator(self._device)

    @property
    def cache_path(self) -> str:
        return self._cache_path

    @abc.abstractmethod
    def set_up_estimator(
        self,
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        pass

    @staticmethod
    def _get_height_width(
        img: Union[torch.Tensor, np.ndarray]
    ) -> Tuple[int, int]:
        if 2 == img.ndim:
            # assume single channel: (H, W)
            return img.shape[0], img.shape[1]
        elif 3 == img.ndim:
            if isinstance(img, np.ndarray):
                # assume channels last: (H, W, C)
                return img.shape[0], img.shape[1]
            else:
                # assume channels first: (C, H, W)
                return img.shape[1], img.shape[2]
        elif 4 == img.ndim:
            # assume channels first with batch axis (B, C, H, W)
            return img.shape[2], img.shape[3]

    @staticmethod
    def _resize_image(
        img: torch.Tensor,
        height: int,
        width: int,
        mode: str = 'nearest'
    ) -> torch.Tensor:

        if AuxiliaryDataEstimatorBase._get_height_width(img) == (height, width):
            # nothing to do
            return img

        # resize
        return torch.nn.functional.interpolate(
            img, size=(height, width), mode=mode
        )

    def prepare_input(
        self,
        image: Union[torch.Tensor, np.ndarray],
    ) -> torch.Tensor:
        # check input
        assert image.ndim in (3, 4)

        # store input type and original shape for later postprocessing
        is_numpy = isinstance(image, np.ndarray)
        h, w = self._get_height_width(image)

        # ensure torch tensor with channels first
        if is_numpy:
            # assume image is channels last, i.e., (H, W, C)
            assert image.ndim == 3 and \
                (image.shape[-1] == 3 or image.shape[-1] == 1)
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)   # (H, W, C) -> (C, H, W)

        # ensure (B, C, H, W)
        if 3 == image.ndim:
            image = image[None, ...]

        if self._max_pixels is not None:
            # resize image to have at most max_pixels while keeping the aspect
            # ratio
            n_pixels = h * w
            if n_pixels > self._max_pixels:
                image = self._resize_image(
                    image,
                    height=int(np.round(h * np.sqrt(self._max_pixels)/w)),
                    width=int(np.round(w * np.sqrt(self._max_pixels)/h)),
                    mode=self._input_interpolation
                )

        return image

    @abc.abstractmethod
    def predict(
        self,
        rgb_img: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        pass

    @abc.abstractmethod
    def _estimator_predict(self, rgb_image: torch.Tensor) -> torch.Tensor:
        """(B,C,H,W) -> (B,H,W)"""
        pass
