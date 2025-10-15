# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Union

import numpy as np
import torch

from .._base import AuxiliaryDataEstimatorBase


UINT16_MAX = np.iinfo('uint16').max


class EmbeddingEstimatorBase(AuxiliaryDataEstimatorBase):
    NAME: str

    def predict(
        self,
        rgb_img: Union[torch.Tensor, np.ndarray],
        mask_img: Union[torch.Tensor, np.ndarray],
    ) -> Union[torch.Tensor, np.ndarray]:
        # store input type and original shape for later postprocessing
        rgb_is_numpy = isinstance(rgb_img, np.ndarray)
        rgb_h, rgb_w = self._get_height_width(rgb_img)
        # Ensure that mask only has 0 and 1 values
        assert np.all(np.isin(mask_img, [0, 1]))

        mask_h, mask_w = self._get_height_width(mask_img)
        assert rgb_h == mask_h and rgb_w == mask_w, \
            f"Input image and mask must have the same shape. " \
            f"Got '{rgb_h}x{rgb_w}' and '{mask_h}x{mask_w}'."

        # prepare the input to have the correct shape
        rgb_img = self.prepare_input(rgb_img)
        mask_img = self.prepare_input(mask_img)

        # apply estimator
        rgb_img = rgb_img.to(self._device).to(torch.float32)
        mask_img = mask_img.to(self._device).to(torch.float32)

        predicted_embeddings = self._estimator_predict(rgb_img, mask_img).cpu()

        # convert to numpy 2d array if input was numpy
        if rgb_is_numpy:
            predicted_embeddings = predicted_embeddings.numpy()

        return predicted_embeddings
