# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Union

from dataclasses import asdict
import os

import numpy as np
import cv2
import json

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import DepthStats
from ...dataset_base import OrientationDict
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from ...dataset_base import ExtrinsicCameraParametersNormalized
from ...dataset_base import IntrinsicCameraParametersNormalized


from .sunrgbd import SUNRGBDMeta


class SUNRGBD(SUNRGBDMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        depth_mode: str = 'refined',
        depth_force_mm: bool = False,
        semantic_use_nyuv2_colors: bool = False,
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
        self._split = split

        self._depth_mode = depth_mode
        self._depth_force_mm = depth_force_mm
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

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
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load whole file list
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            with open(fp, 'r') as f:
                file_list = f.read().splitlines()

            # filter and split for cameras
            self._files = {'list': [], 'dict': {c: [] for c in self._cameras}}
            for fn in file_list:
                # get camera, e.g. kv2/kinect2data/00012 -> kv2
                camera = os.path.normpath(fn).split(os.sep)[0]
                if camera in self._cameras:
                    self._files['list'].append(fn)
                    self._files['dict'][camera].append(fn)

        elif not self._disable_prints:
            print(f"Loaded SUNRGBD dataset without files")

        # build config object
        depth_stats = self.TRAIN_SPLIT_DEPTH_STATS
        if self._depth_force_mm:
            depth_stats = DepthStats(
                **{k: v/10 for k, v in asdict(depth_stats).items()}
            )

        if self._scene_use_indoor_domestic_labels:
            # use remapped scene labels
            scene_label_list = self.SCENE_LABEL_LIST_INDOOR_DOMESTIC
        else:
            # use original scene labels
            scene_label_list = self.SCENE_LABEL_LIST
        if semantic_use_nyuv2_colors:
            semantic_label_list = self.SEMANTIC_LABEL_LIST_NYUV2_COLORS
        else:
            semantic_label_list = self.SEMANTIC_LABEL_LIST
        self._config = build_dataset_config(
            semantic_label_list=semantic_label_list,
            scene_label_list=scene_label_list,
            depth_stats=depth_stats
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    def __len__(self) -> int:
        if self.camera is None:
            return len(self._files['list'])
        return len(self._files['dict'][self.camera])

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return SUNRGBDMeta.SPLIT_SAMPLE_KEYS[split]

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

    @property
    def depth_force_mm(self) -> bool:
        return self._depth_force_mm

    def _get_filename(self, idx: int) -> str:
        if self.camera is None:
            return self._files['list'][idx]
        else:
            return self._files['dict'][self.camera][idx]

    def _load(
        self,
        directory: str,
        idx: int,
        extension: str = '.png'
    ) -> Union[Dict, np.ndarray]:
        # get filename depending on current camera
        filename = self._get_filename(idx)

        # determine filepath
        fp = os.path.join(self._dataset_path,
                          self.split,
                          directory,
                          f'{filename}{extension}')

        # load data
        if '.json' == extension:
            with open(fp, 'r') as f:
                data = json.load(f)
        elif '.txt' == extension:
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
        return self._load(self.IMAGE_DIR, idx, '.jpg')

    def _load_rgb_intrinsics(
        self,
        idx: int
    ) -> IntrinsicCameraParametersNormalized:
        return IntrinsicCameraParametersNormalized({
            # load fx, fy, cx, and cy from file
            **self._load(self.INTRINSICS_DIR, idx, '.json'),
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
        })

    def _load_depth(self, idx: int) -> np.ndarray:
        if self._depth_mode == 'raw':
            depth = self._load(self.DEPTH_DIR_RAW, idx)
        else:
            depth = self._load(self.DEPTH_DIR, idx)

        if self._depth_force_mm:
            # depth is in 1/10 mm, convert to mm
            depth //= 10

        return depth

    def _load_depth_intrinsics(
        self,
        idx: int
    ) -> IntrinsicCameraParametersNormalized:
        a = 0.0001 if not self._depth_force_mm else 0.001   # depth to meters
        return IntrinsicCameraParametersNormalized({
            # load fx, fy, cx, and cy
            **self._load(self.INTRINSICS_DIR, idx, '.json'),
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
            # depth specific parameters (1m = 10000)
            'a': a, 'b': -1
        })

    def _load_identifier(self, idx: int) -> Tuple[str]:
        # get filename depending on current camera
        fn = self._get_filename(idx)
        return SampleIdentifier(os.path.normpath(fn).split(os.sep))

    def _load_semantic(self, idx: int) -> np.ndarray:
        return self._load(self.SEMANTIC_DIR, idx)

    def _load_instance(self, idx: int) -> np.ndarray:
        return self._load(self.INSTANCES_DIR, idx).astype('int32')

    def _load_orientations(self, idx: int) -> Dict[int, float]:
        orientations = self._load(self.ORIENTATIONS_DIR, idx, '.json')
        orientations = {int(k): v for k, v in orientations.items()}
        return OrientationDict(orientations)

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        return self._load(self.BOX_DIR, idx, '.json')

    def _load_scene(self, idx: int) -> int:
        class_str = self._load(self.SCENE_CLASS_DIR, idx, '.txt')

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx

    def _load_extrinsics(
        self,
        idx: int
    ) -> ExtrinsicCameraParametersNormalized:
        return ExtrinsicCameraParametersNormalized(
            self._load(self.EXTRINSICS_DIR, idx, '.json')
        )
