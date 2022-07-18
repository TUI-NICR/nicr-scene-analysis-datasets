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
from .cityscapes import CityscapesMeta


class Cityscapes(CityscapesMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        depth_mode: str = 'raw',
        disparity_instead_of_depth: bool = False,
        semantic_n_classes: int = 19,
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
        self._disparity_instead_of_depth = disparity_instead_of_depth
        self._cameras = self.CAMERAS

        if self._disparity_instead_of_depth:
            self._depth_dir = self.DISPARITY_RAW_DIR
        else:
            self._depth_dir = self.DEPTH_RAW_DIR

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
                self._depth_dir: _loadtxt(f'{self._split}_{self._depth_dir}.txt'),
                'semantic': _loadtxt(f'{self._split}_labels_{self._semantic_n_classes}.txt'),
            }
            assert all(len(li) == len(self._files['rgb'])
                       for li in self._files.values())
        elif not self._disable_prints:
            print(f"Loaded Cityscapes dataset without files")

        # class names, class colors, and semantic directory
        if self._semantic_n_classes == 19:
            semantic_label_list = self.SEMANTIC_LABEL_LIST_REDUCED
            self._semantic_dir = self.SEMANTIC_REDUCED_DIR
        else:
            semantic_label_list = self.SEMANTIC_LABEL_LIST_FULL
            self._semantic_dir = self.SEMANTIC_FULL_DIR

        # build config object
        if disparity_instead_of_depth:
            depth_stats = self.TRAIN_SPLIT_DEPTH_STATS_DISPARITY
        else:
            depth_stats = self.TRAIN_SPLIT_DEPTH_STATS

        self._config = build_dataset_config(
            semantic_label_list=semantic_label_list,
            depth_stats=depth_stats
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

    def _load(self, directory: str, filename: str) -> np.array:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          filename)
        if os.path.splitext(fp)[-1] == '.npy':
            # depth files as numpy files
            return np.load(fp)
        else:
            # all the other files are pngs
            img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"Unable to load image: '{fp}'")
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img

    def _load_rgb(self, idx) -> np.array:
        return self._load(self.RGB_DIR, self._files['rgb'][idx])

    def _load_depth(self, idx) -> np.array:
        depth = self._load(self._depth_dir,
                           self._files[self._depth_dir][idx])
        if depth.dtype == 'float16':
            # precomputed depth values are stored as float16 -> cast to float32
            depth = depth.astype('float32')
            # set values larger than 300 to zero as they are most likely not
            # valid
            depth[depth > 300] = 0
        return depth

    def _load_identifier(self, idx: int) -> Tuple[str]:
        # filenames for all sample keys contain some suffix and an extention,
        # we use the filename of the rgb images and remove both
        fn, _ = os.path.splitext(self._files['rgb'][idx])
        fn = fn.replace('_leftImg8bit', '')
        return SampleIdentifier(os.path.normpath(fn).split(os.sep))

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(self._semantic_dir, self._files['semantic'][idx])
