# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Optional, Union

import os

import numpy as np
import torch
from transformers import AutoImageProcessor
from transformers import AutoModelForDepthEstimation
from transformers.utils import cached_file

from ._base import DepthEstimatorBase
from ...utils.io import get_sha256_hash


class _HuggingFaceDepthEstimator(DepthEstimatorBase):
    DEPTH_SCALE: float
    MODEL_FILE_NAME: str
    MODEL_LOOKUP_DICT: Dict[str, str]
    MODEL_COMMIT_HASH_DICT: Dict[str, str]
    MODEL_HASH_DICT: Dict[str, str]

    def __init__(
        self,
        model_name: str,
        device: Union[str, torch.device] = 'cpu',
        max_pixels: Optional[int] = None,
        auto_set_up: bool = True,
        cache_basepath: Optional[str] = None,  # None -> DEFAULT_CACHE_BASEPATH
    ) -> None:
        assert model_name in self.MODEL_LOOKUP_DICT.keys(), (
            f"Unknown model: '{model_name}', "
            f"Available models: {list(self.MODEL_LOOKUP_DICT.keys())}"
        )
        self._model_name = model_name

        super().__init__(
            device=device,
            max_pixels=max_pixels,
            auto_set_up=auto_set_up,
            cache_basepath=cache_basepath
        )

    def set_up_estimator(
        self,
        device: Union[str, torch.device] = 'cpu'
    ) -> None:
        # get preprocessor
        self._image_processor = AutoImageProcessor.from_pretrained(
            self.MODEL_LOOKUP_DICT[self._model_name],
            revision=self.MODEL_COMMIT_HASH_DICT[self._model_name],
            cache_dir=self.cache_path,
            # For backwards compatibility as the new default behavior changed
            # starting from transformers v4.48.0.
            use_fast=False,
        )

        # get model
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self.MODEL_LOOKUP_DICT[self._model_name],
            revision=self.MODEL_COMMIT_HASH_DICT[self._model_name],
            cache_dir=self.cache_path,
        )

        model_path = cached_file(
            self.MODEL_LOOKUP_DICT[self._model_name],
            filename=self.MODEL_FILE_NAME,
            revision=self.MODEL_COMMIT_HASH_DICT[self._model_name],
            cache_dir=self.cache_path,
            local_files_only=True,
        )
        assert os.path.exists(model_path), (
            f"Model file '{model_path}' does not exist. "
            f"Please check the model name '{self._model_name}' and "
            f"the cache path '{self.cache_path}'"
        )
        # TODO: Hash check is applied to the original model weights.
        # However the transformers library always tries to load a safetensors
        # model, even if the original weights are not safetensors.
        # Here we check the hash of the original model weights, but the
        # safetensors model is not guaranteed to be the same.
        # Because of this we might check the hash of the wrong file.
        # For us, only the DINOv2 model is affected, as the others are already
        # safetensors. See comments at DINOv2DPTDepthEstimator class for more
        # details.
        model_hash = get_sha256_hash(model_path)
        assert model_hash == self.MODEL_HASH_DICT[self._model_name], (
            f"Model file '{model_path}' has wrong hash '{model_hash}' "
            f"should be '{self.MODEL_HASH_DICT[self._model_name]}'"
        )
        self._model.eval()
        self._model.to(device)

    @torch.inference_mode()
    def _estimator_predict(self, rgb_image: torch.Tensor) -> torch.Tensor:
        # apply preprocessing
        inputs = self._image_processor(images=rgb_image, return_tensors='pt')

        # apply model
        inputs = inputs.to(self._model.device)
        outputs = self._model(**inputs)

        # apply postprocessing
        predicted_depth = self._estimator_postprocess(
            rgb_image=rgb_image,
            predicted_depth=outputs.predicted_depth
        )

        return predicted_depth

    def _estimator_postprocess(
        self,
        rgb_image: torch.Tensor,
        predicted_depth: torch.Tensor
    ) -> torch.Tensor:
        # default implementation, simply scale the depth
        predicted_depth *= self.DEPTH_SCALE

        return predicted_depth


class DepthAnythingV2DepthEstimator(_HuggingFaceDepthEstimator):
    NAME = 'depthanything_v2'
    MODEL_FILE_NAME = 'model.safetensors'
    MODEL_LOOKUP_DICT = {
        # indoor = trained on hypersim
        'indoor_small': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf',
        'indoor_base': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Base-hf',
        'indoor_large': 'depth-anything/Depth-Anything-V2-Metric-Indoor-Large-hf',
        # indoor = trained on vkitti
        'outdoor_small': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf',
        'outdoor_base': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Base-hf',
        'outdoor_large': 'depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf',
    }
    MODEL_COMMIT_HASH_DICT = {
        'indoor_small': '8078d68a9c75a972131914f6afd0c1723be0da7f',
        'indoor_base': 'c6d9784685727bfc6d0a7b5452ce94afaee1e7f5',
        'indoor_large': 'd2fc6a93601aabb1139a3bf0ebfcb4e89c67817f',

        'outdoor_small': 'fd2c22027eaf20374204f14099b8341e1925ad39',
        'outdoor_base': '5b575fded0346068343a6d96d0ce3ab6dbc9fb49',
        'outdoor_large': '4eab4cf1983c2801c515804005214de56a4b67cc',
    }
    MODEL_HASH_DICT = {
        'indoor_small': 'e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3',
        'indoor_base': '8b902f5d9a8c8a9520f3a8c6a00afe442b464eff5aaf0cb405d1da721cd9f79f',
        'indoor_large': '04d3295ff4f9cbee72b8998a85930fe870b45ba4431e7af47415fbded02388ca',

        'outdoor_small': 'ad065c77a7421ca55159a1f0db9433397a607690f2d76bb8a6fc54b1be7a3124',
        'outdoor_base': '1f2e9ed13ee39126686994669f8d1bfd9cd8f093a4d2a0891e5a0e5263b480c9',
        'outdoor_large': '15f7de4b22f1b1194847ea694ee9ed6395563ba5cd52b70b87901f183c232c9a',
    }
    DEPTH_SCALE = 1000     # to convert from meters to millimeters


class ZoeDepthDepthEstimator(_HuggingFaceDepthEstimator):
    NAME = 'zoedepth'
    MODEL_FILE_NAME = 'model.safetensors'
    MODEL_LOOKUP_DICT = {
        'indoor': 'Intel/zoedepth-nyu',
        'outdoor': 'Intel/zoedepth-kitti',
        'indoor_outdoor': 'Intel/zoedepth-nyu-kitti',
    }
    MODEL_COMMIT_HASH_DICT = {
        'indoor': '52ae69caf7896c927909b3ceab4249c235e16dc1',
        'outdoor': '0347db85b019f309d5544c70ad2ac998ad570899',
        'indoor_outdoor': 'f364d4c7936e91f465abba182208dd68142bf0ca',
    }
    MODEL_HASH_DICT = {
        'indoor': 'b616e347efc30e64822be28d2464c2fcd0b665b5fdbc9a135d53d168c31c77ab',
        'outdoor': '8427d1e398736f909cc7432276981417b2e7ec698b4d70786022142def78fe38',
        'indoor_outdoor': 'c5494fa0938f18d71e215e245472470c3aefebd7b434abd89750e5ae4008e2dc',
    }
    DEPTH_SCALE = 1000     # to convert from meters to millimeters

    def _estimator_postprocess(
        self,
        rgb_image: torch.Tensor,
        predicted_depth: torch.Tensor
    ) -> torch.Tensor:
        # there is some strange reflection padding applied to the input to fix
        # boundary artifacts, to align input and output, we must to crop the
        # output accordingly
        # see: https://github.com/huggingface/transformers/blob/edeca4387c527c3f8d35c1b941ebe3ecad9cd798/src/transformers/models/zoedepth/image_processing_zoedepth.py#L282

        _, _, height, width = rgb_image.shape
        pad_height = int(np.sqrt(height / 2) * 3)
        pad_width = int(np.sqrt(width / 2) * 3)

        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),  # add channel axis
            size=(height + 2*pad_height, width + 2*pad_width),
            mode='nearest'
        ).squeeze(1)  # remove channel axis

        predicted_depth = predicted_depth[:, pad_height:-pad_height, pad_width:-pad_width]

        return super()._estimator_postprocess(
            rgb_image=rgb_image,
            predicted_depth=predicted_depth
        )


class DinoV2DPTDepthEstimator(_HuggingFaceDepthEstimator):
    # TODO: THE CURRENT HASH CHECK DOES NOT ENSURE INTEGRITY OF THE FINAL MODEL.
    # This is due to a quirk in how model weights are handled.
    # Facebook provides them as pickle files, but the Transformers
    # library prefers safetensors format (to prevent code execution).
    # Since the commit hash doesn't include safetensors, Hugging Face falls
    # back to locating a pull request with converted weights. This happens
    # asynchronously in another thread. Because of this, the hash check happens
    # on the original (unconverted) weights (as the .safetensor model is not
    # downloaded yet), but the actual weights loaded are the converted one.
    # As a result, the hash check for DINOv2 does not guarantee integrity of
    # the final `safetensors` model.
    # See: https://github.com/huggingface/transformers/blob/v4.48.0/src/transformers/safetensors_conversion.py
    NAME = 'dino_v2_dpt'
    MODEL_FILE_NAME = 'pytorch_model.bin'
    MODEL_LOOKUP_DICT = {
        'indoor_small': 'facebook/dpt-dinov2-small-nyu',
        'indoor_base': 'facebook/dpt-dinov2-base-nyu',
        'indoor_large': 'facebook/dpt-dinov2-large-nyu',
        'indoor_giant': 'facebook/dpt-dinov2-giant-nyu',
        'outdoor_small': 'facebook/dpt-dinov2-small-kitti',
        'outdoor_base': 'facebook/dpt-dinov2-base-kitti',
        'outdoor_large': 'facebook/dpt-dinov2-large-kitti',
        'outdoor_giant': 'facebook/dpt-dinov2-giant-kitti',
    }
    MODEL_COMMIT_HASH_DICT = {
        'indoor_small': '21a14468819a6aa4a1bbe56d872ba8be23184b0f',
        'indoor_base': '5776fdeb81586a1563c9a4414d47e4156b3ea9b1',
        'indoor_large': 'ef81b2d74ef33a42107b498206e889d3b2a10c27',
        'indoor_giant': 'c18b63c9ecc1c5e868d240dc8f58ab5e56e8e1a1',

        'outdoor_small': 'c0d86b8c609b96365202ba99a2065e2b549ad782',
        'outdoor_base': 'e388c90f97fd01c1e3fa393fad136b3be1dd71d0',
        'outdoor_large': '8c762dd3921556d73ba9f4c6af96665feb5195c7',
        'outdoor_giant': 'a2ad1f6f221e58dc561e2f8bc98cd705f6bde7fb',
    }
    MODEL_HASH_DICT = {
        'indoor_small': '16f263f1793476fe83a10c02da0897573935b4ddb640123c213b39f672649e2a',
        'indoor_base': '6f09db8fd5a5368934ac7eb0ab9881b1fb059d9b85c08ce916ada16f030b22c1',
        'indoor_large': 'd4dedbe0ca27012862c1f32128d4f7db9b528dc831e28dbee9d7abe10ba48705',
        'indoor_giant': '2850413a96d137e46a4f75563ecbedf47387bb7fca8fd9a078b00420e340e829',

        'outdoor_small': 'd75e96bdb1355fb9075aa3ecd727daa78fbd2887f6c02dbf410aa4b0237a6ba8',
        'outdoor_base': '85f648db0567d70d2b9434559d1a79c0bdb1d64aca37ae8e59820256d17ff3d1',
        'outdoor_large': '27b17137a546badf93285fba5d5a4cb6bd09974601a5477848a602a271a07c41',
        'outdoor_giant': '71bb74fe8c73bbb078da13e12a1efe67667b018a015b02708f6fa444d745e444',
    }
    DEPTH_SCALE = 1000     # to convert from meters to millimeters

    def _estimator_postprocess(
        self,
        rgb_image: torch.Tensor,
        predicted_depth: torch.Tensor
    ) -> torch.Tensor:

        # DinoV2 uses the DPTImageProcessor image preprocessor, which pads the
        # image image dimensions to a multiple of 14:
        # see: https://github.com/facebookresearch/dinov2/blob/dc1d2cbcc8204b1f2f988e31aa8fc710e65e2d8d/notebooks/depth_estimation.ipynb
        # see: https://github.com/huggingface/transformers/blob/edeca4387c527c3f8d35c1b941ebe3ecad9cd798/src/transformers/models/dpt/image_processing_dpt.py#L95
        # see: https://github.com/huggingface/transformers/blob/edeca4387c527c3f8d35c1b941ebe3ecad9cd798/src/transformers/models/dpt/image_processing_dpt.py#L223
        # to align input and output, we must to crop the output accordingly

        # however, according to:
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/eval/depth/models/depther/encoder_decoder.py#L69
        # simple resizing is used (which, in our case, is done in the base
        # class)

        # therefore, the code below is disabled
        # moreover, note that the output is of different shape than the input, so
        # the code below would not work anyway

        # def _get_pad(size, size_divisor):
        #     # copied from links above
        #     new_size = np.ceil(size / size_divisor) * size_divisor
        #     pad_size = new_size - size
        #     pad_size_left = pad_size // 2
        #     pad_size_right = pad_size - pad_size_left
        #     return pad_size_left, pad_size_right

        # _, _, height, width = rgb_image.shape
        # size_divisor = self._image_processor.size_divisor
        # pad_left, pad_right = _get_pad(height, size_divisor=size_divisor)
        # pad_top, pad_bottom = _get_pad(width, size_divisor=size_divisor)

        # # crop the output
        # predicted_depth = predicted_depth[:, :, pad_top:-pad_bottom, pad_left:-pad_right]
        # assert predicted_depth.shape[-2:] == (height, width)

        return super()._estimator_postprocess(
            rgb_image=rgb_image,
            predicted_depth=predicted_depth
        )
