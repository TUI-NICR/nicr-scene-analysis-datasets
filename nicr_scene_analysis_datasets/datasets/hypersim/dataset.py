# -*- coding: utf-8 -*-
"""
.. codeauthor:: Marius Engelhardt <marius.engelhardt@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de

"""
from typing import Any, Dict, Optional, Tuple

import json
import os

import cv2
import numpy as np

from ...dataset_base import build_dataset_config
from ...dataset_base import DatasetConfig
from ...dataset_base import ExtrinsicCameraParametersNormalized
from ...dataset_base import IntrinsicCameraParametersNormalized
from ...dataset_base import OrientationDict
from ...dataset_base import RGBDDataset
from ...dataset_base import SampleIdentifier
from .hypersim import HypersimMeta


class Hypersim(HypersimMeta, RGBDDataset):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        split: str = 'train',
        subsample: Optional[int] = None,
        sample_keys: Tuple[str] = ('rgb', 'depth', 'semantic'),
        depth_mode: str = 'raw',
        scene_use_indoor_domestic_labels: bool = False,
        use_cache: bool = False,
        **kwargs: Any
    ) -> None:
        super().__init__(
            depth_mode=depth_mode,
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        # note: depth mode is 'raw' since we have to exclude the zero depth
        # values that come from clipping the depth to uint16
        assert depth_mode in self.DEPTH_MODES
        assert split in self.SPLITS

        self._split = split
        self._depth_mode = depth_mode
        self._cameras = self.CAMERAS
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

        self._subsample = subsample
        if subsample:
            print(f"Using subsampling: '{subsample}' for split '{split}'")

        if dataset_path is not None:
            dataset_path = os.path.expanduser(dataset_path)
            assert os.path.exists(dataset_path), dataset_path
            self._dataset_path = dataset_path

            # load filenames
            # set constant here to have the same identifier across all datasets
            self.SPLIT_FILELIST_FILENAMES = self.get_split_filelist_filenames(
                subsample=subsample
            )
            fp = os.path.join(self._dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
        elif not self._disable_prints:
            print(f"Loaded Hypersim dataset without files")

        # build config object
        if self._scene_use_indoor_domestic_labels:
            # use remapped scene labels
            scene_label_list = self.SCENE_LABEL_LIST_INDOOR_DOMESTIC
        else:
            # use original scene labels
            scene_label_list = self.SCENE_LABEL_LIST

        self._config = build_dataset_config(
            semantic_label_list=self.SEMANTIC_LABEL_LIST,
            scene_label_list=scene_label_list,
            depth_stats=self.TRAIN_SPLIT_DEPTH_STATS
        )

        # register loader functions
        self.auto_register_sample_key_loaders()

    def __len__(self) -> int:
        return len(self._filenames)

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
    def instance_max_instances_per_image(self) -> int:
        return self.MAX_INSTANCES_PER_IMAGE

    def _load(
        self,
        directory: str,
        filename: str,
    ) -> np.array:
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

    def _load_rgb(self, idx) -> np.array:
        return self._load(self.RGB_DIR, f'{self._filenames[idx]}.png')

    def _load_rgb_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        return IntrinsicCameraParametersNormalized(
            self.RGB_INTRINSICS_NORMALIZED[self.cameras[0]]    # single camera
        )

    def _load_depth(self, idx) -> np.array:
        return self._load(self.DEPTH_DIR, f'{self._filenames[idx]}.png')

    def _load_depth_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        return IntrinsicCameraParametersNormalized(
            self.DEPTH_INTRINSICS_NORMALIZED[self.cameras[0]]    # single camera
        )

    def _load_identifier(self, idx: int) -> Tuple[str]:
        return SampleIdentifier(
            os.path.normpath(self._filenames[idx]).split(os.sep)
        )

    def _load_extrinsics(self, idx: int) -> ExtrinsicCameraParametersNormalized:
        fp_extrinsics = os.path.join(self._dataset_path,
                                     self.split,
                                     self.EXTRINSICS_DIR,
                                     f'{self._filenames[idx]}.json')
        with open(fp_extrinsics, 'r') as f:
            extrinsics = json.load(f)

        return ExtrinsicCameraParametersNormalized(extrinsics)

    def _load_semantic(self, idx: int) -> np.array:
        return self._load(self.SEMANTIC_DIR, f'{self._filenames[idx]}.png')

    def _load_instance(self, idx: int) -> np.array:
        # Notes:
        # - actually, only channel 1 and 2 hold the instance id
        # - channel 0 holds the semantic id
        # - channel 1 and 2 together encode a uint16 for the real instance id
        #   (note that this instance id can correspond to multiple semantic
        #   labels, to encode unique instance ids, use the semantic channel as
        #   well)
        # - hewever, since HyperSim is only used for pretraining and we need
        #   this id for getting the orientation, only the instance id without
        #   the semantic label is used
        im = self._load(self.INSTANCES_DIR, f'{self._filenames[idx]}.png')
        im_sliced = im[:, :, 1:3]
        # array must be contiguous for uint16 view
        im_sliced = np.ascontiguousarray(im_sliced).view(np.uint16)
        # remove channel axis
        instance_im = im_sliced[..., 0]

        return instance_im.astype('int32')

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
        fp_box = os.path.join(self._dataset_path,
                              self.split,
                              self.BOXES_3D_DIR,
                              f'{self._filenames[idx]}.json')
        with open(fp_box, 'r') as f:
            boxes_dict = json.load(f)

        return boxes_dict

    def _load_scene(self, idx: int) -> int:
        fp = os.path.join(self._dataset_path,
                          self.split,
                          self.SCENE_CLASS_DIR,
                          f'{self._filenames[idx]}.txt')
        with open(fp, "r") as f:
            class_str = f.read().splitlines()[0].lower()

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx

    def _load_normal(self, idx: int) -> np.array:
        # format is xyz with invalid values (127, 127, 127)
        normal = self._load(self.NORMAL_DIR, f'{self._filenames[idx]}.png')
        # convert to float, thus, invalid values are (0., 0., 0.)
        normal = normal.astype('float32')
        normal /= 127
        normal -= 1
        # ensure unit length for valid normals (add 1e-7 to circumvent /0)
        norm = np.linalg.norm(normal, ord=2, axis=-1, keepdims=True) + 1e-7
        return normal/norm
