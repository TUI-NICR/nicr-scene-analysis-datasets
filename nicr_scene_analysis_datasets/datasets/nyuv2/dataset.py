# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple

import json
import os

import cv2
import numpy as np

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import OrientationDict
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from .nyuv2 import NYUv2Meta


class NYUv2(NYUv2Meta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        depth_mode: str = 'refined',
        semantic_n_classes: int = 40,
        scene_use_indoor_domestic_labels: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        assert split in self.SPLITS
        assert depth_mode in self.DEPTH_MODES
        self._semantic_n_classes = semantic_n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._cameras = self.CAMERAS
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load filenames
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
        elif not self._disable_prints:
            print(f"Loaded NYUv2 dataset without files")

        # build config object
        semantic_label_list = getattr(
            self,
            f'SEMANTIC_LABEL_LIST_{self._semantic_n_classes}'
        )
        if self._scene_use_indoor_domestic_labels:
            # use remapped scene labels
            scene_label_list = self.SCENE_LABEL_LIST_INDOOR_DOMESTIC
        else:
            # use original scene labels
            scene_label_list = self.SCENE_LABEL_LIST
        self._config = build_dataset_config(
            semantic_label_list=semantic_label_list,
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
        return len(self._filenames)

    def _load(self, directory: str, filename: str) -> np.array:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          f'{filename}.png')

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.array:
        return self._load(self.RGB_DIR, self._filenames[idx])

    def _load_depth(self, idx) -> np.array:
        if self._depth_mode == 'raw':
            return self._load(self.DEPTH_RAW_DIR, self._filenames[idx])
        else:
            return self._load(self.DEPTH_DIR, self._filenames[idx])

    def _load_identifier(self, idx: int) -> Tuple[str]:
        return SampleIdentifier((self._filenames[idx],))

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(
            self.SEMANTIC_DIR_FMT.format(self._semantic_n_classes),
            self._filenames[idx]
        )

    def _load_instance(self, idx: int) -> np.array:
        instance = self._load(self.INSTANCES_DIR, self._filenames[idx])
        return instance.astype('int32')

    def _load_orientations(self, idx: int) -> Dict[int, float]:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          self.ORIENTATIONS_DIR,
                          f'{self._filenames[idx]}.json')
        with open(fp, 'r') as f:
            orientations = json.load(f)

        orientations = {int(k): v for k, v in orientations.items()}
        return OrientationDict(orientations)

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def _load_scene(self, idx: int) -> int:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          self.SCENE_CLASS_DIR,
                          f'{self._filenames[idx]}.txt')
        if not os.path.isfile(fp):
            # catch common error
            raise FileNotFoundError(
                "Scene class file not found. Maybe the SUNRGBD matching was "
                "not done yet."
            )
        with open(fp, "r") as f:
            class_str = f.readline()

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx

    def _load_normal(self, idx: int) -> np.array:
        # format is xyz with invalid values (127, 127, 127)
        normal = self._load(self.NORMAL_DIR, self._filenames[idx])
        # convert to float, thus, invalid values are (0., 0., 0.)
        normal = normal.astype('float32')
        normal /= 127
        normal -= 1
        # ensure unit length for valid normals (add 1e-7 to circumvent /0)
        norm = np.linalg.norm(normal, ord=2, axis=-1, keepdims=True) + 1e-7
        return normal/norm
