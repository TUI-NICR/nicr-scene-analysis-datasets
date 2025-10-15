# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Leonard Rabes <leonard.rabes@tu-ilmenau.de>
"""

from typing import Any, Optional, Tuple

import json
import os

import cv2
import numpy as np

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import ExtrinsicCameraParametersNormalized
from ...dataset_base import IntrinsicCameraParametersNormalized
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from .scannet import ScanNetMeta


class ScanNet(ScanNetMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        subsample: Optional[int] = 50,
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        depth_mode: str = 'raw',
        semantic_n_classes: int = 40,
        instance_semantic_mode: str = 'refined',
        scene_use_indoor_domestic_labels: bool = False,
        semantic_use_nyuv2_colors: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            dataset_path=dataset_path,
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        # catch common error
        if 'test' == split:
            # annotations for test split are not publicly available
            available_keys_test = ScanNetMeta.SPLIT_SAMPLE_KEYS['test']
            for s in sample_keys:
                if s not in available_keys_test:
                    raise ValueError(
                        f"Sample key: '{s}' is not available for test split. "
                        f"Available keys: {available_keys_test}."
                    )

        assert split in self.SPLITS
        assert depth_mode in self.DEPTH_MODES
        assert semantic_n_classes in self.SEMANTIC_N_CLASSES
        assert instance_semantic_mode in self.INSTANCE_SEMANTIC_MODES
        # only semantic_n_classes=40 or 20 can have nyuv2 colors
        assert any(((semantic_n_classes in (40, 20)) and semantic_use_nyuv2_colors,
                    not semantic_use_nyuv2_colors))
        assert all(sk in self.get_available_sample_keys(split)
                   for sk in sample_keys)

        self._instance_semantic_mode = instance_semantic_mode
        self._semantic_n_classes = semantic_n_classes
        self._split = split
        self._depth_mode = depth_mode
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels
        self._subsample = subsample

        if subsample is not None:
            self.debug_print(
                f"ScanNet: using subsample: '{subsample}' for '{split}'"
            )

        # cameras
        if cameras is None:
            # use all available cameras
            self._cameras = self.CAMERAS
        else:
            # use subset of cameras
            assert all(c in self.CAMERAS for c in cameras)
            self._cameras = cameras

        # load file list
        if dataset_path is not None:
            # load whole file list for correct split and subsample
            fp = os.path.join(
                self.dataset_path,
                self.get_split_filelist_filenames(subsample)[self._split]
            )
            with open(fp, 'r') as f:
                file_list = f.read().splitlines()

            # filter and split for cameras
            self._files = {'list': [], 'dict': {c: [] for c in self._cameras}}
            for fn in file_list:
                # get camera, e.g. structureio_968x1296/scene0000_00/04000
                # -> structureio_968x1296
                camera = os.path.normpath(fn).split(os.sep)[0]

                if camera in self._cameras:
                    self._files['list'].append(fn)
                    self._files['dict'][camera].append(fn)
        else:
            self.debug_print("Loaded ScanNet dataset without files")

        # build config object
        semantic_label_list = getattr(
            self,
            f'SEMANTIC_LABEL_LIST_{self._semantic_n_classes}'
        )
        if semantic_use_nyuv2_colors:
            # force NYUv2 colors (only 20/40 classes)
            if self._semantic_n_classes == 40:
                semantic_label_list = self.SEMANTIC_LABEL_LIST_NYUV2_COLORS_40
            elif self._semantic_n_classes == 20:
                semantic_label_list = self.SEMANTIC_LABEL_LIST_NYUV2_COLORS_20
            else:
                raise ValueError()

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

    def __len__(self) -> int:
        if self.camera is None:
            return len(self._files['list'])
        return len(self._files['dict'][self.camera])

    @property
    def cameras(self) -> Tuple[str]:
        return self._cameras

    @property
    def config(self) -> DatasetConfig:
        return self._config

    @property
    def split(self) -> str:
        return self._split

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return ScanNetMeta.SPLIT_SAMPLE_KEYS[split]

    def _get_filename(self, idx: int) -> str:
        if self.camera is None:
            return self._files['list'][idx]
        else:
            return self._files['dict'][self.camera][idx]

    def _get_scene_name(self, idx: int) -> str:
        # get scene name, e.g., scene0000_00/00000 -> scene0000_00
        if self.camera is None:
            return os.path.dirname(self._files['list'][idx])
        return os.path.dirname(self._files['dict'][self.camera][idx])

    def _load(
        self,
        directory: str,
        index: int,
        extension: str = '.png'
    ) -> np.ndarray:
        filename = self._get_filename(index)
        fp = os.path.join(self.dataset_path,
                          self.SPLIT_DIRS[self.split],
                          directory,
                          f'{filename}{extension}')

        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_extrinsics(
        self,
        idx: int
    ) -> ExtrinsicCameraParametersNormalized:
        fp = os.path.join(self.dataset_path,
                          self.SPLIT_DIRS[self.split],
                          self.EXTRINSICS_DIR,
                          f'{self._get_filename(idx)}.json')
        with open(fp, 'r') as f:
            extrinsics = json.load(f)

        return ExtrinsicCameraParametersNormalized(extrinsics)

    def _load_rgb(self, idx: int) -> np.ndarray:
        # note that the provided raw color images are already compressed using
        # jpg, we directly write the compressed bytestream to disk and do not
        # apply another encoding
        # further note that neither RGB nor any spatial annotation is
        # registered to depth, however, as the shift between both is minimal,
        # simple resizing during preprocessing is fine, for more details, see:
        # https://github.com/ScanNet/ScanNet/issues/109
        return self._load(self.RGB_DIR, idx, '.jpg')

    def _load_rgb_intrinsics(
        self,
        idx: int
    ) -> IntrinsicCameraParametersNormalized:
        fp = os.path.join(self.dataset_path,
                          self.SPLIT_DIRS[self.split],
                          self.INTRINSICS_RGB_DIR,
                          f'{self._get_scene_name(idx)}.json')
        with open(fp, 'r') as f:
            intrinsic = json.load(f)

        return IntrinsicCameraParametersNormalized({
            **intrinsic,
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
        })

    def _load_depth(self, idx: int) -> np.ndarray:
        # note that depth is not registered to RGB or any spatial annotation,
        # however, as the shift between both is minimal, simple resizing
        # during preprocessing is fine, for more details, see:
        # https://github.com/ScanNet/ScanNet/issues/109
        return self._load(self.DEPTH_DIR, idx, '.png')

    def _load_depth_intrinsics(
        self,
        idx: int
    ) -> IntrinsicCameraParametersNormalized:
        fp = os.path.join(self.dataset_path,
                          self.SPLIT_DIRS[self.split],
                          self.INTRINSICS_DEPTH_DIR,
                          f'{self._get_scene_name(idx)}.json')
        with open(fp, 'r') as f:
            intrinsic = json.load(f)

        return IntrinsicCameraParametersNormalized({
            **intrinsic,
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
            # depth specific parameters (1m = 1000)
            'a': 0.001, 'b': -1
        })

    def _load_identifier(self, idx: int) -> Tuple[str]:
        ident = self._files['list'][idx]
        if self.camera is not None:
            ident = self._files['dict'][self.camera][idx]

        return SampleIdentifier(os.path.normpath(ident).split(os.sep))

    def _load_semantic(self, idx: int) -> np.ndarray:
        semantic = self._load(
            self.SEMANTIC_DIR_FMT.format(self._instance_semantic_mode,
                                         self._semantic_n_classes),
            idx,
            '.png'
        )
        if self._semantic_n_classes <= 255:
            # convert to uint8
            semantic = semantic.astype('uint8')
        else:
            # convert to uint16
            semantic = semantic.astype('uint16')
        return semantic

    def _load_instance(self, idx: int) -> np.ndarray:
        # max train: 132, max valid: 107
        # instance images are using uint8
        instance = self._load(
            self.INSTANCES_DIR_FMT.format(self._instance_semantic_mode),
            idx,
            '.png'
        )
        return instance.astype('uint16')

    def _load_scene(self, idx: int) -> int:
        fp = os.path.join(self.dataset_path,
                          self.SPLIT_DIRS[self.split],
                          self.SCENE_CLASS_DIR,
                          f'{self._get_scene_name(idx)}.txt')
        with open(fp, 'r') as f:
            class_str = f.readline()

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx
