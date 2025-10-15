# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Union


import numpy as np
import torch

from .._base import AuxiliaryDataEstimatorBase


UINT16_MAX = np.iinfo('uint16').max


class DepthEstimatorBase(AuxiliaryDataEstimatorBase):
    NAME: str

    def predict(
        self,
        rgb_img: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        # store input type and original shape for later postprocessing
        is_numpy = isinstance(rgb_img, np.ndarray)
        h, w = self._get_height_width(rgb_img)

        # prepare the input to have the correct shape
        rgb_img = self.prepare_input(rgb_img)

        # apply estimator
        rgb_img = rgb_img.to(self._device).to(torch.float32)

        predicted_depth = self._estimator_predict(rgb_img).cpu()

        # resize to original shape
        predicted_depth = self._resize_image(
            predicted_depth[:, None, ...],   # (B, H, W) -> (B, C, H, W)
            height=h, width=w, mode='nearest'
        )

        # convert to numpy 2d array if input was numpy
        if is_numpy:
            predicted_depth = predicted_depth.numpy()[0, 0]
            n_above_max = (predicted_depth > UINT16_MAX).sum()
            if n_above_max > 0:
                print(
                    f"Warning: Detected {n_above_max} values above "
                    f"{UINT16_MAX} in predicted depth."
                )
                predicted_depth = np.clip(predicted_depth, 0, UINT16_MAX)
            predicted_depth = np.asarray(predicted_depth, dtype='uint16')

            assert 2 == predicted_depth.ndim

        return predicted_depth
