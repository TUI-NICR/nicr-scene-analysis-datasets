# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple

import os

import cv2
import numpy as np

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import RGBDataset
from ...dataset_base import SampleIdentifier
from .coco import COCOMeta


class COCO(COCOMeta, RGBDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'semantic'),
        use_cache: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        assert split in self.SPLITS
        self._split = split

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load filenames
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)

            # COCO is comprised of images of various cameras and spatial
            # dimensions, so we do not know the actual cameras, however, in the
            # dataset class, we use the camera property to split the dataset
            # in virtual cameras with images of same spatial dimensions
            # get filelist for each camera
            self._filenames_per_camera = {}
            for fn in self._filenames:
                camera = os.path.dirname(fn)
                if camera not in self._filenames_per_camera:
                    self._filenames_per_camera[camera] = []
                self._filenames_per_camera[camera].append(fn)

            self._cameras = tuple(self._filenames_per_camera.keys())
        elif not self._disable_prints:
            print(f"Loaded COCO dataset without files")
            self._cameras = self.CAMERAS

        # build config object
        self._config = build_dataset_config(
            semantic_label_list=self.SEMANTIC_LABEL_LIST,
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    @property
    def cameras(self) -> Tuple[str]:
        return self._cameras

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def split(self) -> str:
        return self._split

    def __len__(self) -> int:
        if self.camera is None or self.CAMERAS[0] == self.camera:
            return len(self._filenames)
        return len(self._filenames_per_camera[self.camera])

    def _get_filename(self, idx: int) -> str:
        if self.camera is None or self.CAMERAS[0] == self.camera:
            return self._filenames[idx]
        else:
            return self._filenames_per_camera[self.camera][idx]

    def _load(
        self,
        directory: str,
        idx: int,
        ext: str = '.png'
    ) -> np.array:
        # get filename depending on current camera
        filename = self._get_filename(idx)

        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          f'{filename}{ext}')
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if 3 == img.ndim:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.array:
        img = self._load(self.IMAGE_DIR, idx, '.jpg')

        # force RGB if the image is grayscale
        if 2 == img.ndim:
            img = img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        return img

    def _load_identifier(self, idx: int) -> Tuple[str]:
        # get filename depending on current camera
        filename = self._get_filename(idx)
        return SampleIdentifier(os.path.normpath(filename).split(os.sep))

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(self.SEMANTIC_DIR, idx)

    def _load_instance(self, idx: int) -> np.array:
        instance = self._load(self.INSTANCES_DIR, idx)
        return instance.astype('int32')
