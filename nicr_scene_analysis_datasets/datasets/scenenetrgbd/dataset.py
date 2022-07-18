# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple

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
        depth_mode: str = 'refined',
        semantic_n_classes: int = 13,
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

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load file lists
            def _loadtxt(fn):
                return np.loadtxt(os.path.join(self._dataset_path, fn),
                                  dtype=str)

            self._files = {
                'rgb': _loadtxt(f'{self._split}_rgb.txt'),
                'depth': _loadtxt(f'{self._split}_depth.txt'),
                'label': _loadtxt(f'{self._split}_labels_{self._semantic_n_classes}.txt')
            }
            assert all(len(li) == len(self._files['rgb'])
                       for li in self._files.values())
        elif not self._disable_prints:
            print(f"Loaded SceneNetRGBD dataset without files")

        # build config object
        self._config = build_dataset_config(
            semantic_label_list=self.SEMANTIC_LABEL_LIST,
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
        return len(self._files['rgb'])

    def _load(self, directory, filename) -> np.array:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          filename)
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx: int) -> np.array:
        return self._load(self.RGB_DIR, self._files['rgb'][idx])

    def _load_depth(self, idx: int) -> np.array:
        return self._load(self.DEPTH_DIR, self._files['depth'][idx])

    def _load_identifier(self, idx: int) -> Tuple[str]:
        fn, _ = os.path.splitext(self._files['rgb'][idx])
        return SampleIdentifier(os.path.normpath(fn).split(os.sep))

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(self.SEMANTIC_13_DIR, self._files['label'][idx])
