# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
from typing import Any
from typing import Optional
from typing import Tuple

import cv2
import numpy as np

from ...dataset_base import DatasetConfig
from ...dataset_base import RGBDataset
from ...dataset_base import SampleIdentifier
from ...dataset_base import build_dataset_config
from .ade20k import ADE20KMeta


class ADE20K(ADE20KMeta, RGBDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train_panoptic_2017',
        sample_keys: Tuple[str] = ('rgb', 'semantic', 'instance', 'scene'),
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        assert split in self.SPLITS
        assert all(
            sk in self.get_available_sample_keys(split) for sk in sample_keys
        )

        self._split = split

        if dataset_path is not None:
            # load filenames
            fp = os.path.join(self.dataset_path,
                              self.get_split_filelist_filename(split))
            self._filenames = list(np.loadtxt(fp, dtype=str))

            # ADE20K is comprised of images of various cameras and spatial
            # dimensions, so we do not know the actual cameras, however, in the
            # dataset class, we use the camera property to split the dataset
            # in virtual cameras with images of same spatial dimensions, thus
            # we parse the cameras from the filenames

            # get filelist for each camera
            self._filenames_per_camera = {}
            for fn in self._filenames:
                camera = os.path.dirname(fn)
                if camera not in self._filenames_per_camera:
                    self._filenames_per_camera[camera] = []
                self._filenames_per_camera[camera].append(fn)

            available_cameras = tuple(self._filenames_per_camera.keys())

            if cameras is None:
                # use all available cameras
                self._cameras = available_cameras
            else:
                # use subset of cameras
                assert all(c in available_cameras for c in cameras)
                self._cameras = cameras

                # filter dict
                for camera in list(self._filenames_per_camera.keys()):
                    if camera not in self._cameras:
                        # remove from dict
                        del self._filenames_per_camera[camera]
                # recreate filelist
                self._filenames = []
                for camera, filenames in self._filenames_per_camera.items():
                    self._filenames.extend(
                        os.path.join(fn) for fn in filenames
                    )
        else:
            self.debug_print("Loaded ADE20K dataset without files")
            self._cameras = self.CAMERAS	 # single dummy camera
            self._filenames = []  # no files
            self._filenames_per_camera = {k: 0 for k in self.CAMERAS}

        # build config object
        semantic_label_list = self.SEMANTIC_LABEL_LIST_CHALLENGE_150
        scene_label_list = self.SCENE_LABEL_LIST_CHALLENGE_1055

        self._config = build_dataset_config(
            semantic_label_list=semantic_label_list,
            scene_label_list=scene_label_list
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

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return ADE20KMeta.SPLIT_SAMPLE_KEYS[split]

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
    ) -> np.ndarray:
        # get filename depending on current camera
        filename = self._get_filename(idx)
        fp = os.path.join(
            self.dataset_path,
            self.split,
            directory,
            f'{filename}{ext}'
        )

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if 3 == img.ndim:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.ndarray:
        img = self._load(self.IMAGE_DIR, idx, '.jpg')
        # force RGB if the image is grayscale
        if 2 == img.ndim:
            img = img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        return img

    def _load_identifier(self, idx: int) -> SampleIdentifier:
        filename = self._get_filename(idx)
        return SampleIdentifier(os.path.normpath(filename).split(os.sep))

    def _load_semantic(self, idx: int) -> np.ndarray:
        semantic = self._load(
            self.SEMANTIC_DIR,
            idx
        )
        semantic = semantic.astype('uint8')
        return semantic

    def _load_instance(self, idx: int) -> np.ndarray:
        instance = self._load(self.INSTANCES_DIR, idx)
        instance = instance.astype('uint16')
        return instance

    def _load_scene(self, idx: int) -> int:
        fp = os.path.join(self.dataset_path,
                          self.split,
                          self.SCENE_CLASS_DIR,
                          f'{self._filenames[idx]}.txt')
        with open(fp, 'r') as f:
            class_str = f.readline()

        class_idx = self.config.scene_label_list.index(class_str)
        return class_idx
