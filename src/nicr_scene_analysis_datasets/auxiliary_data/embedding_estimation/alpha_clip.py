
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, List, Optional, Union

import os

import alpha_clip
import gdown
import torch
from PIL import Image
from torchvision import transforms

from ._base import EmbeddingEstimatorBase
from ...utils.io import get_sha256_hash


class AlphaCLIPEmbeddingEstimator(EmbeddingEstimatorBase):
    DEPTH_SCALE: float
    # Links to models are taken from:
    # https://github.com/SunzeY/AlphaCLIP/blob/main/model-zoo.md
    # The dict always contains the model name and the google drive id
    MODEL_LOOKUP_DICT: Dict[str, str] = {
        # models trained on grit-1m
        'b16-grit-1m': '16fHEXZ-7bgzcSBHzEz1wXRIZTjQIXm_2',
        'l14-grit-1m': '1PIhplBnsmSWiJN--TXCCSsiaV6bY9koq',
        'l14-336-grit-1m': '1DeNbUv0lraDxJZItb7shTlvGW6z_Z9Si',
        # models trained on grit-20m
        'b16-grit-20m': '1cj3cYwrzBivx0h0NzSjlCg9HAd5aTkDW',
        'l14-grit-20m': '1WykuBYWePriCVeW5lOwBsgxgeBMzb1nd',
        'l14-336-grit-20m': '1dUq1deeLcou26RuxZbBG57m2ALPWev6-',
        # models trained on mimagenet_top+grit-1m
        'b16-combined': '11iDlSAYI_BAi1A_Qz6LTWYHNgPe-UY7I',
        'l14-combined': '1JfzOTvjf0tqBtKWwpBJtjYxdHi-06dbk',
    }

    # Sha256 hashes for the models so we can verify the downloaded files
    MODEL_HASHES = {
        'b16-grit-1m':
            '9f6ffac9044f641b09975f9c782a34a02a2b52ffd8de4a3d3394404b38794c16',
        'l14-grit-1m':
            '05254743d5fc12e24f947fb92ba3737e3bc86a9c190dc6608fc9926a0eb64b91',
        'l14-336-grit-1m':
            'fa701cf18a3e19a1e5e81668885b15bf75337e59b0788405999e889b7aa775d0',
        'b16-grit-20m':
            '6ff9e3c5735aa34bcbd386ba4d1b99e5efc5deffab077d3d2f2911d1f1c43a57',
        'l14-grit-20m':
            '42c621bb5bac89a511ab625a878f026c11de4de8cf7abf52db5dfd11862bdb8e',
        'l14-336-grit-20m':
            'b0e9c4d5d33bfa2724a0864f8e4b2b92f99aba3a4daf43084114c8caecfbadd9',
        'b16-combined':
            'a00f7bfd80944dacbb77e238eb2c0f055ec3274201023b2e781d9f0d180d9630',
        'l14-combined':
            'a5f3f2e24459e9764d9f4b4c053fb354dc9d508bd8f647b952402d6860bc9c3d'
    }

    NAME = 'alpha_clip'

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
        model_ckpt_path = os.path.join(
            self.cache_path, f'{self._model_name}.pth'
        )
        if not os.path.exists(model_ckpt_path):
            gdown.download(
                id=self.MODEL_LOOKUP_DICT[self._model_name],
                output=model_ckpt_path,
            )
        # Check the hash of the downloaded file
        # to ensure that the file was downloaded correctly
        # and is not corrupted
        file_hash = get_sha256_hash(model_ckpt_path)
        assert file_hash == self.MODEL_HASHES[self._model_name], \
            f"Model file '{model_ckpt_path}' has wrong hash '{file_hash}' " \
            f"should be '{self.MODEL_HASHES[self._model_name]}'"

        base_model_name = None
        input_size = None
        if 'l14-336' in self._model_name:
            base_model_name = 'ViT-L/14@336px'
            input_size = 336
        elif 'l14' in self._model_name:
            base_model_name = 'ViT-L/14'
            input_size = 224
        elif 'b16' in self._model_name:
            base_model_name = 'ViT-B/16'
            input_size = 224
        else:
            raise ValueError(f"Unknown model name: '{self._model_name}'")

        assert base_model_name is not None, "base_model_name is None"
        assert input_size is not None, "input_size is None"

        # create preprocessor for input mask
        self._mask_transform = transforms.Compose([
            transforms.Resize((input_size, input_size), Image.NEAREST),
            transforms.Normalize(0.5, 0.26)
        ])

        # get model and image processor
        self._model, self._image_processor = alpha_clip.load(
            base_model_name,
            alpha_vision_ckpt_pth=model_ckpt_path,
            device=device
        )
        self._device = device

        # Model by default is in float16. It is converted to float32
        # so we can use float32 inputs.
        self._model = self._model.float()

        # In the original alpha clip implementation the image gets loaded
        # as pillow image. As we already have a tensor we can skip this step.
        # This is why we slightly adjusted the preprocessing.
        assert isinstance(self._image_processor.transforms[-1], transforms.Normalize)
        original_normalize = self._image_processor.transforms[-1]
        # As we don't need the ToTensor which scales from [0, 255] to [0, 1]
        # we adjust the normalization to the range [0, 255]
        adjusted_normalize = transforms.Normalize(
            (
                original_normalize.mean[0] * 255,
                original_normalize.mean[1] * 255,
                original_normalize.mean[2] * 255
            ),
            (
                original_normalize.std[0] * 255,
                original_normalize.std[1] * 255,
                original_normalize.std[2] * 255
            )
        )

        self._image_processor = transforms.Compose([
            # Resize bicubic interpolation
            transforms.Resize((input_size, input_size), Image.BICUBIC, antialias=True),
            transforms.CenterCrop((input_size, input_size)),
            adjusted_normalize
        ])

        self._model.eval()

    @torch.inference_mode()
    def _estimator_predict(
        self,
        rgb_image: torch.Tensor,
        mask_image: torch.Tensor
    ) -> torch.Tensor:
        # apply preprocessing
        rgb_input = self._image_processor(rgb_image)
        mask_input = self._mask_transform(mask_image).float()

        # apply model
        rgb_input = rgb_input.to(self._device)
        mask_input = mask_input.to(self._device)

        outputs = self._model.visual(rgb_input, mask_input)

        return outputs

    def _estimator_postprocess(
        self,
        rgb_image: torch.Tensor,
        predicted_depth: torch.Tensor
    ) -> torch.Tensor:
        # default implementation, simply scale the depth
        predicted_depth *= self.DEPTH_SCALE

        return predicted_depth

    @torch.inference_mode()
    def _get_text_embedding(self, text_list: List[str]) -> torch.Tensor:
        tokenized_text = alpha_clip.tokenize(text_list).to(self._device)
        text_features = self._model.encode_text(tokenized_text)
        return text_features
