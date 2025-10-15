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
        use_cache: bool = False,
        cameras: Optional[Tuple[str]] = None,
        depth_mode: str = 'raw',
        orientations_use: bool = False,
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

        # note: depth mode is 'raw' since we have to exclude the zero depth
        # values that come from clipping the depth to uint16
        assert depth_mode in self.DEPTH_MODES
        assert split in self.SPLITS
        assert all(sk in self.get_available_sample_keys(split) for sk in sample_keys)

        self._split = split
        self._depth_mode = depth_mode
        self._orientations_use = orientations_use
        self._scene_use_indoor_domestic_labels = scene_use_indoor_domestic_labels

        # cameras
        if cameras is None:
            # use all available cameras (=default dummy camera)
            self._cameras = self.CAMERAS
        else:
            # use subset of cameras (does not really apply to this dataset)
            assert all(c in self.CAMERAS for c in cameras)
            self._cameras = cameras

        self._subsample = subsample
        if subsample is not None:
            self.debug_print(
                f"Hypersim: using subsample: '{subsample}' for '{split}'"
            )

        if dataset_path is not None:
            # load filenames
            # set constant here to have the same identifier across all datasets
            self.SPLIT_FILELIST_FILENAMES = self.get_split_filelist_filenames(
                subsample=subsample
            )
            fp = os.path.join(self.dataset_path,
                              self.SPLIT_FILELIST_FILENAMES[self._split])
            self._filenames = np.loadtxt(fp, dtype=str)
        else:
            self.debug_print("Loaded Hypersim dataset without files")

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

    def _get_filename(self, idx: int) -> str:
        return self._filenames[idx]

    def __len__(self) -> int:
        return len(self._filenames)

    @staticmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        return HypersimMeta.SPLIT_SAMPLE_KEYS[split]

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
    def instance_max_instances_per_image(self) -> int:
        return self.MAX_INSTANCES_PER_IMAGE

    def _load(
        self,
        directory: str,
        filename: str,
    ) -> np.ndarray:
        fp = os.path.join(self.dataset_path,
                          self.split,
                          directory,
                          filename)
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img

    def _load_rgb(self, idx) -> np.ndarray:
        return self._load(self.RGB_DIR, f'{self._filenames[idx]}.png')

    def _load_rgb_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        scene, cam, _ = self._load_identifier(idx)
        fp = os.path.join(self.dataset_path,
                          self.split,
                          self.RGB_INTRINSICS_DIR,
                          scene,
                          f'{cam}.json')
        with open(fp, 'r') as f:
            intrinsics = json.load(f)

        return IntrinsicCameraParametersNormalized({
            **intrinsics,
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
        })

    def _load_depth(self, idx) -> np.ndarray:
        return self._load(self.DEPTH_DIR, f'{self._filenames[idx]}.png')

    def _load_depth_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        scene, cam, _ = self._load_identifier(idx)
        fp = os.path.join(self.dataset_path,
                          self.split,
                          self.DEPTH_INTRINSICS_DIR,
                          scene,
                          f'{cam}.json')
        with open(fp, 'r') as f:
            intrinsics = json.load(f)

        return IntrinsicCameraParametersNormalized({
            **intrinsics,
            # use defaults for remaining parameters
            'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
            'p1': 0, 'p2': 0,
        })

    def _load_identifier(self, idx: int) -> Tuple[str]:
        return SampleIdentifier(
            os.path.normpath(self._filenames[idx]).split(os.sep)
        )

    def _load_extrinsics(self, idx: int) -> ExtrinsicCameraParametersNormalized:
        fp_extrinsics = os.path.join(self.dataset_path,
                                     self.split,
                                     self.EXTRINSICS_DIR,
                                     f'{self._filenames[idx]}.json')
        with open(fp_extrinsics, 'r') as f:
            extrinsics = json.load(f)

        return ExtrinsicCameraParametersNormalized(extrinsics)

    def _load_semantic(self, idx: int) -> np.ndarray:
        return self._load(
            self.SEMANTIC_DIR,
            f'{self._filenames[idx]}.png'
        ).astype('uint8')

    def _load_instance(self, idx: int) -> np.ndarray:
        # Notes:
        # - channel idx=0 holds the semantic class
        # - channel idx=1 and idx=2 hold the instance id encoded as an uint16
        # - combining semantic and instance might be useful for some tasks as an
        #   instance id may correspond to multiple semantic classes:
        #   - note this only affects few instances
        #   - most overlaps are with void (unlabeled textures -> void label)
        #    - ai_017_004: semantic classes 35 + 40
        #        -> lamp + otherprop: some small stuff in the background
        #    - ai_021_008: semantic classes 12 + 35
        #        -> kitchen counter + lamp belong to same instance -> might be
        #           an annotation fault
        #    - ai_022_009: semantic classes 1 + 8:
        #        -> door frame labeled as wall, but door instance contains both
        #           the door frame and the door
        im = self._load(self.INSTANCES_DIR, f'{self._filenames[idx]}.png')
        instance = im[:, :, 1].astype('uint16') << 8
        instance += im[:, :, 2].astype('uint16')

        return instance.astype('uint16')

    def _load_orientations(self, idx: int) -> Dict[int, float]:
        # be aware that provided dataset orientations might not be consistent
        # for instances within the same semantic class, thus, we disable
        # instances by default
        if not self._orientations_use:
            return OrientationDict({})

        # use orientations provided by the dataset
        fp = os.path.join(self.dataset_path,
                          self.split,
                          self.ORIENTATIONS_DIR,
                          f'{self._filenames[idx]}.json')

        with open(fp, 'r') as f:
            orientations = json.load(f)

        orientations = {int(k): v for k, v in orientations.items()}
        return OrientationDict(orientations)

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        fp_box = os.path.join(self.dataset_path,
                              self.split,
                              self.BOXES_3D_DIR,
                              f'{self._filenames[idx]}.json')
        with open(fp_box, 'r') as f:
            boxes_dict = json.load(f)

        return boxes_dict

    def _load_scene(self, idx: int) -> int:
        fp = os.path.join(self.dataset_path,
                          self.split,
                          self.SCENE_CLASS_DIR,
                          f'{self._filenames[idx]}.txt')
        with open(fp, 'r') as f:
            class_str = f.read().splitlines()[0].lower()

        class_idx = self.SCENE_LABEL_LIST.index(class_str)

        if self._scene_use_indoor_domestic_labels:
            # map class to indoor domestic environment labels
            mapping = self.SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX
            class_idx = mapping[class_idx]

        return class_idx

    def _load_normal(self, idx: int) -> np.ndarray:
        # format is xyz with invalid values (127, 127, 127)
        normal = self._load(self.NORMAL_DIR, f'{self._filenames[idx]}.png')
        # convert to float, thus, invalid values are (0., 0., 0.)
        normal = normal.astype('float32')
        normal /= 127
        normal -= 1
        # ensure unit length for valid normals (add 1e-7 to circumvent /0)
        norm = np.linalg.norm(normal, ord=2, axis=-1, keepdims=True) + 1e-7
        return normal/norm
