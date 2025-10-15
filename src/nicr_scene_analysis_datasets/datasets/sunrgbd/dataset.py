# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Dict, Optional, Tuple, Union

import json
import os

import cv2
import numpy as np

from ...dataset_base import DatasetConfig
from ...dataset_base import ExtrinsicCameraParametersNormalized
from ...dataset_base import IntrinsicCameraParametersNormalized
from ...dataset_base import OrientationDict
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from ...dataset_base import build_dataset_config
from .sunrgbd import SUNRGBDMeta


class SUNRGBD(SUNRGBDMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        instances_version: str = 'panopticndt',  # see notes below
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

        # we created two versions of SUNRGB-D with instance annotations
        # extracted from existing 3d-box annotations:
        # - 'emsanet': this initial version was created for training EMSANet
        #   (efficient panoptic segmentation) - see IJCNN 2022 paper - and was
        #   also used for EMSAFormer (efficient panoptic segmentation) - see
        #   IJCNN 2023 paper
        # - 'panopticndt': this revised version was created along with the work
        #   for PanopticNDT (panoptic mapping) - see IROS 2023 paper, it
        #   refines large parts of the instance extraction (see changelog for
        #   v0.6.0 of this package)
        # - 'anyold': this value can be used as workaround to load any dataset
        #   prepared with a package version < v0.7.0 - use this value only if
        #   you know what you are doing!
        assert instances_version in (self.INSTANCE_VERSIONS + ('anyold',))
        self._instances_version = instances_version

        # try to load annotation version from creation meta, if not available
        # use the passed value
        if (
            self.creation_meta is not None and
            'additional_meta' in self.creation_meta
        ):
            self._instances_version_meta = \
                self.creation_meta['additional_meta'].get(
                    'instances_version',
                    self._instances_version
                )
        else:
            self._instances_version_meta = self._instances_version

        # determine paths based on annotation version
        if 'emsanet' == self._instances_version:
            assert self._instances_version_meta == self._instances_version

            self.INSTANCES_DIR = self.INSTANCES_EMSANET_DIR
            self.ORIENTATIONS_DIR = self.ORIENTATIONS_EMSANET_DIR
            self.BOXES_DIR = self.BOXES_EMSANET_DIR
        elif 'panopticndt' == self._instances_version:
            assert self._instances_version_meta == self._instances_version

            self.INSTANCES_DIR = self.INSTANCES_PANOPTICNDT_DIR
            self.ORIENTATIONS_DIR = self.ORIENTATIONS_PANOPTICNDT_DIR
            self.BOXES_DIR = self.BOXES_PANOPTICNDT_DIR
        elif 'anyold' == self._instances_version:
            self.INSTANCES_DIR = self.INSTANCES_LEGACY_DIR
            self.ORIENTATIONS_DIR = self.ORIENTATIONS_LEGACY_DIR
            self.BOXES_DIR = self.BOXES_LEGACY_DIR

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
            # load whole file list
            fp = os.path.join(self.dataset_path,
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
        else:
            self.debug_print("Loaded SUNRGBD dataset without files")

        # build config object
        if not self._depth_force_mm:
            depth_stats = self.TRAIN_SPLIT_DEPTH_STATS
        else:
            depth_stats = self.TRAIN_SPLIT_DEPTH_STATS_MM

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
    ) -> Union[Dict, str, np.ndarray]:
        # get filename depending on current camera
        filename = self._get_filename(idx)

        # determine filepath
        fp = os.path.join(self.dataset_path,
                          self.split,
                          directory,
                          f'{filename}{extension}')

        # load data
        try:
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
        except (FileNotFoundError, IOError) as e:
            # handle common errors caused by changes introduced in v0.7.0
            raise FileNotFoundError(
                f"Cannot load file: '{fp}'. \n"
                "It is likely that your are trying to load files from a "
                "SUNRGB-D dataset prepared with another version of this "
                "dataset package. We recommend re-preparing the SUNRGB-D "
                "dataset with the current version of the dataset package. "
                "Otherwise - and only if you know what you are doing - you "
                "might consider the `instances_version` parameter of this "
                "SUNRGB-D dataset class to force loading anyway."
            ) from e

        return data

    def _load_rgb(self, idx: int) -> np.ndarray:
        return self._load(self.RGB_DIR, idx, '.jpg')

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
        # load depth image
        if 'raw' == self._depth_mode:
            depth = self._load(self.DEPTH_DIR_RAW, idx)
        else:
            depth = self._load(self.DEPTH_DIR, idx)

        if not self._depth_force_mm:
            # nothing to do, return raw depth (use this for benchmarking)
            return depth

        # convert to mm (use this for applications)

        # depth is encoded in only 13 of the 16 bits, i.e., bits 3-15 store the
        # actual depth information
        # see: http://velastin.dynu.com/G3D/G3D.html:
        #    The depth information was also mapped to the colour coordinate
        #    space and stored in a 16-bit greyscale. The 16-bits of depth
        #    data contains 13 bits for depth data and 3 bits to identify
        #    the player.

        # see: https://social.msdn.microsoft.com/Forums/en-US/3fe21ce5-4b75-4b31-b73d-2ff48adfdf52/kinect-uses-12-bits-or-13-bits-for-depth-data?forum=kinectsdk

        # original code from toolbox:
        # -> SUNRGBDtoolbox/SUNRGBDtoolbox/readData/read3dPoints.m:
        #     function [rgb,points3d,depthInpaint,imsize]=read3dPoints(data)
        #             depthVis = imread(data.depthpath);
        #             imsize = size(depthVis);
        #             depthInpaint = bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3));
        #             depthInpaint = single(depthInpaint)/1000;
        #             depthInpaint(depthInpaint >8)=8;
        #             [rgb,points3d]=read_3d_pts_general(depthInpaint,data.K,size(depthInpaint),data.rgbpath);
        #             points3d = (data.Rtilt*points3d')';

        # NOTE:
        # we only apply the shift to the right by 3 bits to get rid of the
        # lowest three bits and then clip to 8m; in the toolbox code above, the
        # lowest three bits are added again to the highest bits, and then the
        # depth values are clipped to 8000 (=8m); we do not know the exact
        # reason for the first step as subsequent clipping to 8000 again
        # excludes the highest 3 bits
        depth = np.right_shift(depth, 3)

        # clip to 8m (note, number of pixels affected is small, i.e., ~0.03%)
        if 'raw' == self._depth_mode:
            # as depth mode is 'raw', we set the values to zero to
            # indicate invalid
            depth[depth > 8000] = 0
        else:
            # for the refined depth images, we follow the toolbox code
            depth[depth > 8000] = 8000

        return depth

    def _load_depth_intrinsics(
        self,
        idx: int
    ) -> IntrinsicCameraParametersNormalized:
        if not self._depth_force_mm:
            # see above: d_mm = d >> 3 --> /8 --> *0.125 * 0.001
            # note that this is only an approximation, as the lower 3 bits
            # are not removed
            a = 0.000125
        else:
            a = 0.001   # depth to meters

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
        return self._load(self.SEMANTIC_DIR, idx).astype('uint8')

    def _load_instance(self, idx: int) -> np.ndarray:
        return self._load(self.INSTANCES_DIR, idx).astype('uint16')

    def _load_orientations(self, idx: int) -> Dict[int, float]:
        orientations = self._load(self.ORIENTATIONS_DIR, idx, '.json')
        orientations = {int(k): v for k, v in orientations.items()}
        return OrientationDict(orientations)

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        return self._load(self.BOXES_DIR, idx, '.json')

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
