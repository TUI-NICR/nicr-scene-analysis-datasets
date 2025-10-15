# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Union

import json
import os

import cv2
import numpy as np

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from .scenenetrgbd import SceneNetRGBDMeta


class SceneNetRGBD(SceneNetRGBDMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        depth_mode: str = 'refined',
        scene_use_indoor_domestic_labels: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        assert split in self.SPLITS
        assert depth_mode in self.DEPTH_MODES
        assert all(sk in self.get_available_sample_keys(split) for sk in sample_keys)
        self._semantic_n_classes = 13
        self._split = split
        self._depth_mode = depth_mode
        self._cameras = self.CAMERAS
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

        # cameras
        if cameras is None:
            # use all available cameras (=default dummy camera)
            self._cameras = self.CAMERAS
        else:
            # use subset of cameras (does not really apply to this dataset)
            assert all(c in self.CAMERAS for c in cameras)
            self._cameras = cameras

        # load file list
        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load file list
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            with open(fp, 'r') as f:
                self._files = f.read().splitlines()

        elif not self._disable_prints:
            print(f"Loaded SceneNetRGBD dataset without files")

        if self._scene_use_indoor_domestic_labels:
            # use remapped scene labels
            scene_label_list = self.SCENE_LABEL_LIST_INDOOR_DOMESTIC
        else:
            # use original scene labels
            scene_label_list = self.SCENE_LABEL_LIST

        # build config object
        self._config = build_dataset_config(
            semantic_label_list=self.SEMANTIC_LABEL_LIST,
            scene_label_list=scene_label_list,
            depth_stats=self.TRAIN_SPLIT_DEPTH_STATS
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

    @property
    def depth_mode(self) -> str:
        return self._depth_mode

    def __len__(self) -> int:
        return len(self._files)

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return SceneNetRGBDMeta.SPLIT_SAMPLE_KEYS[split]

    def _load(
        self,
        directory: str,
        idx: int,
        extension: str = '.png'
    ) -> Union[str, np.ndarray]:
        # determine filepath
        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          f'{self._files[idx]}{extension}')

        # load data
        if '.txt' == extension:
            with open(fp, 'r') as f:
                data = f.readline()
        else:
            # default load using OpenCV
            data = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if data is None:
                raise IOError(f"Unable to load image: '{fp}'")
            if data.ndim == 3:
                data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)

        return data

    def _load_rgb(self, idx: int) -> np.ndarray:
        return self._load(self.RGB_DIR, idx, '.jpg')

    def _load_depth(self, idx: int) -> np.ndarray:
        return self._load(self.DEPTH_DIR, idx)

    def _load_identifier(self, idx: int) -> Tuple[str]:
        fn = self._files[idx]
        return SampleIdentifier(os.path.normpath(fn).split(os.sep))

    def _load_semantic(self, idx: int) -> np.ndarray:
        return self._load(self.SEMANTIC_13_DIR, idx).astype('uint8')

    def _load_instance(self, idx: int) -> np.ndarray:
        return self._load(self.INSTANCES_DIR, idx).astype('uint16')

    def _load_scene(self, idx: int) -> int:
        class_str = self._load(self.SCENE_CLASS_DIR, idx, '.txt')

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx
