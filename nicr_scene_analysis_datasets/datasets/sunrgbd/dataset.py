# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Union

import os

import numpy as np
import cv2
import json

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
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
        depth_mode: str = 'refined',
        semantic_use_nyuv2_colors: bool = False,
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
        self._split = split

        self._cameras = self.CAMERAS
        self._depth_mode = depth_mode
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path
            self._filenames_dict = {}
            # load filenames
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames_dict['list'], self._filenames_dict['dict'] = \
                self._list_and_dict_from_file(fp)
        elif not self._disable_prints:
            print(f"Loaded SUNRGBD dataset without files")

        # build config object
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
            depth_stats=self.TRAIN_SPLIT_DEPTH_STATS
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    def _list_and_dict_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        dictionary = dict()
        for cam in self.cameras:
            dictionary[cam] = [i for i in file_list if cam in i]

        return file_list, dictionary

    def __len__(self) -> int:
        if self.camera is None:
            return len(self._filenames_dict['list'])
        return len(self._filenames_dict['dict'][self.camera])

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

    def _get_filename(self, idx: int) -> str:
        if self.camera is None:
            return self._filenames_dict['list'][idx]
        else:
            return self._filenames_dict['dict'][self.camera][idx]

    def _load(
        self,
        directory: str,
        idx: int,
        extension: str = '.png'
    ) -> Union[Dict, np.array]:
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

    def _load_rgb(self, idx: int) -> np.array:
        return self._load(self.IMAGE_DIR, idx, '.jpg')

    def _load_rgb_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        return IntrinsicCameraParametersNormalized({
            # load fx, fy, cx, and cy from file
            **self._load(self.INTRINSICS_DIR, idx, '.json'),
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
        })

    def _load_depth(self, idx: int) -> np.array:
        if self._depth_mode == 'raw':
            return self._load(self.DEPTH_DIR_RAW, idx)
        else:
            return self._load(self.DEPTH_DIR, idx)

    def _load_depth_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        return IntrinsicCameraParametersNormalized({
            # load fx, fy, cx, and cy
            **self._load(self.INTRINSICS_DIR, idx, '.json'),
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
            # depth specific parameters (1m = 10000)
            'a': 0.0001, 'b': -1
        })

    def _load_identifier(self, idx: int) -> Tuple[str]:
        # get filename depending on current camera
        fn = self._get_filename(idx)
        return SampleIdentifier(os.path.normpath(fn).split(os.sep))

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(self.SEMANTIC_DIR, idx)

    def _load_instance(self, idx: int) -> np.array:
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

    def _load_extrinsics(self, idx: int) -> ExtrinsicCameraParametersNormalized:
        return ExtrinsicCameraParametersNormalized(
            self._load(self.EXTRINSICS_DIR, idx, '.json')
        )
