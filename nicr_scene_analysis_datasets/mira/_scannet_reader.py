# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import cv2
import numpy as np

import mirapy
from PythonCameraIntrinsicWrapper import PinholeCameraIntrinsicNormalized
from PythonCameraIntrinsicWrapper import DepthCameraIntrinsicNormalized

from .. import ScanNet
from ._base_reader import MIRAReaderBase
from .utils import to_mira_img
from .utils import to_mira_img8u1


class ScanNetReaderBase(MIRAReaderBase):
    def __init__(
        self,
        feedback_channel_type=None,
        sample_keys=('extrinsics',
                     'rgb', 'rgb_intrinsics',
                     'depth', 'depth_intrinsics',
                     'semantic',
                     'instance',
                     'scene')
    ):
        super().__init__(feedback_channel_type)

        self._sample_keys = sample_keys

        # reflected members ----------------------------------------------------
        self._dataset_semantic_n_classes = None
        self._dataset_semantic_map_to_benchmark = None
        self._dataset_use_domestic_scene_classes = None

        self._last_color_intrinsics = None
        self._last_depth_intrinsics = None

    def reflect(self, r):
        super().reflect(r)

        def add_member_roproperty(name, type_, getter, setter, desc, *args):
            # use args to pass additional arguments such as a default
            r.member(name, type_, getter, setter, desc, *args)
            r.roproperty(name, type_, getter, desc)

        # dataset-related properties -------------------------------------------
        add_member_roproperty(    # no default, must be set by user
            'DatasetSemanticNClasses',
            int,
            self._rget_dataset_semantic_n_classes,
            self._rset_dataset_semantic_n_classes,
            "Number of classes to use (ground-truth semantic only).",
        )
        add_member_roproperty(    # no default, must be set by user
            'DatasetSemanticMapToBenckmark',
            bool,
            self._rget_dataset_semantic_map_to_benchmark,
            self._rset_dataset_semantic_map_to_benchmark,
            "Map to benchmark classes (ground-truth semantic only).",
        )
        add_member_roproperty(
            'DatasetUseDomesticSceneClasses',
            bool,
            self._rget_dataset_use_domestic_scene_classes,
            self._rset_dataset_use_domestic_scene_classes,
            "Use domestic scene classes (ground-truth scene only).",
            False
        )

    def parse_dataset(self):
        # sanity checks
        if self._dataset_semantic_map_to_benchmark:
            # NOTE: we currently only support the ScanNet benchmark, ScanNet200
            # benchmark would mean to map from 200 to tsv ids, which exceeds the
            # current semantic class limit of 254 (uint8)
            assert self._dataset_semantic_n_classes == 20

            mapping = ScanNet.SEMANTIC_CLASSES_20_MAPPING_TO_BENCHMARK  # with void
            mapping = np.array(list(mapping.values()), dtype=np.uint8)
            self._semantic_class_mapper = lambda x: mapping[x]
        else:
            self._semantic_class_mapper = lambda x: x

        # simply retrieve all sample identifiers
        identifier_dataset = ScanNet(
            dataset_path=self._dataset_basepath,
            split=self._dataset_split,
            subsample=self._dataset_subsample,
            semantic_n_classes=self._dataset_semantic_n_classes,
            scene_use_indoor_domestic_labels=self._dataset_use_domestic_scene_classes,
            sample_keys=('identifier',)
        )
        for idx, s in enumerate(identifier_dataset):
            # unpack identifier
            # e.g., (structureio_968x1296, scene0707_00, 00000)
            _, scene, id_ = s['identifier']
            scene, camera = scene.split('_')  # scene0707_00 -> scene0707, 00

            # apply identifier filter
            if self._dataset_filter_str:
                patterns = self._dataset_filter_str.split(',')
                if not any(p.strip() in f'{scene}/{camera}/{id_}'
                           for p in patterns):
                    continue

            if scene not in self._dataset_meta:
                self._dataset_meta[scene] = {camera: [(idx, id_)]}
            elif camera not in self._dataset_meta[scene]:
                self._dataset_meta[scene][camera] = [(idx, id_)]
            else:
                self._dataset_meta[scene][camera].append((idx, id_))

        # load full dataset
        sample_keys = self._sample_keys
        if self._dataset_split == 'test':
            # test split does not have annotations
            sample_keys = [
                k
                for k in sample_keys
                if k not in ('semantic', 'instance', 'scene')
            ]

        self._dataset = ScanNet(
            dataset_path=self._dataset_basepath,
            split=self._dataset_split,
            subsample=self._dataset_subsample,
            semantic_n_classes=self._dataset_semantic_n_classes,
            scene_use_indoor_domestic_labels=self._dataset_use_domestic_scene_classes,
            sample_keys=(
                'identifier',
                *sample_keys,
            )
        )

    def process_sample(self, sample):
        sample_mira = {}

        # small sanity check
        assert sample['identifier'][1] == f'{self.cur_scene}_{self.cur_camera}'

        # extrinsic ------------------------------------------------------------
        if 'extrinsics' in sample:
            ext = sample['extrinsics']

            if any(not np.isfinite(v) for v in ext.values()):
                # ScanNet seems to contain some broken samples (invalid extrinsic
                # parameters, e.g. structureio_968x1296/scene0015_00/01610), we
                # return None to indicate that this frame should be skipped
                return None

            translation = mirapy.Point3f(ext['x'], ext['y'], ext['z'])
            rotation_quat = mirapy.Quaternionf()
            rotation_quat.x = ext['quat_x']
            rotation_quat.y = ext['quat_y']
            rotation_quat.z = ext['quat_z']
            rotation_quat.w = ext['quat_w']

            transform = mirapy.Pose3(translation, rotation_quat)
            sample_mira['extrinsic'] = transform

        # intrinsic ------------------------------------------------------------
        # note, both do not change over time
        if 'rgb_intrinsics' in sample:
            color_intrinsics = sample['rgb_intrinsics']
            if self._last_color_intrinsics != color_intrinsics:
                # add only if changed to avoid recreating LUTs in MIRA
                self._last_color_intrinsics = color_intrinsics
                sample_mira['color_intrinsic'] = PinholeCameraIntrinsicNormalized(
                    color_intrinsics['fx'], color_intrinsics['fy'],
                    color_intrinsics['cx'], color_intrinsics['cy'],
                    color_intrinsics['k1'], color_intrinsics['k2'],
                    color_intrinsics['p1'], color_intrinsics['p2']
                )

        if 'depth_intrinsics' in sample:
            depth_intrinsics = sample['depth_intrinsics']
            if self._last_depth_intrinsics != depth_intrinsics:
                # add only if changed to avoid recreating LUTs in MIRA
                self._last_depth_intrinsics = depth_intrinsics
                sample_mira['depth_intrinsic'] = DepthCameraIntrinsicNormalized(
                    PinholeCameraIntrinsicNormalized(
                        depth_intrinsics['fx'], depth_intrinsics['fy'],
                        depth_intrinsics['cx'], depth_intrinsics['cy'],
                        depth_intrinsics['k1'], depth_intrinsics['k2'],
                        depth_intrinsics['p1'], depth_intrinsics['p2']
                    ),
                    depth_intrinsics['a'], depth_intrinsics['b']
                )

        # images ---------------------------------------------------------------
        # note that neither RGB nor any spatial annotation are registered to
        # depth, however, as the shift between both is minimal, simple resizing
        # during preprocessing is fine, for more details, see:
        # https://github.com/ScanNet/ScanNet/issues/109
        # TODO: what about the depth intrinsic? should we use color instead?
        if 'rgb' in sample:
            sample_mira['color_img'] = to_mira_img(
               sample['rgb'],
               rgb2bgr=True    # BGR for MIRA!
            )

        if 'depth' in sample:
            if 'rgb' in sample:
                h, w, _ = sample['rgb'].shape
                depth_img = cv2.resize(sample['depth'], (w, h),
                                       interpolation=cv2.INTER_NEAREST)
            else:
                depth_img = sample['depth']

            sample_mira['depth_img'] = to_mira_img(depth_img)

        # ground-truth semantic segmentation -----------------------------------
        if 'semantic' in sample:
            semantic = self._semantic_class_mapper(sample['semantic'])
            # we use a score of 0.999 for all pixels
            sample_mira['semantic_gt'] = to_mira_img(
                semantic.astype('float32') + 0.999
            )
            sample_mira['semantic_gt_classes'] = to_mira_img8u1(
                semantic    # < 255 classes !!!
            )

        # predicted semantic segmentation --------------------------------------
        if self._load_predicted_semantic:
            sem, sem_classes = self.load_predicted_semantic(
                sample['identifier']
            )
            sem = sem.transpose(1, 2, 0)    # convert to channels last
            sem = np.ascontiguousarray(sem)    # <- important !
            sample_mira['semantic'] = to_mira_img(sem)

            sample_mira['semantic_classes'] = [
                to_mira_img8u1(s) for s in sem_classes
            ]

        # ground-truth instance segmentation -----------------------------------
        if 'instance' in sample:
            # we use a score of 0.999 for all pixels
            sample_mira['instance_gt'] = to_mira_img(
                sample['instance'].astype('float32') + 0.999
            )
            sample_mira['instance_gt_meta'] = \
                self.create_instance_meta_from_semantic_instance(
                    sample['semantic'], sample['instance']
                )
            sample_mira['instance_gt_ids'] = to_mira_img(
                sample['instance'].astype('uint16')    # < 65535 ids
            )

        # predicted instance segmentation --------------------------------------
        if self._load_predicted_instance:
            ins, ins_ids, ins_meta = self.load_predicted_instance(
                sample['identifier']
            )
            sample_mira['instance'] = to_mira_img(ins)
            sample_mira['instance_meta'] = ins_meta
            sample_mira['instance_ids'] = to_mira_img(ins_ids)

        # ground-truth scene class ---------------------------------------------
        if 'scene' in sample:
            sample_mira['scene_gt'] = float(sample['scene']) + 0.999
            sample_mira['scene_gt_class'] = sample['scene']

        # predicted scene class ------------------------------------------------
        if self._load_predicted_scene:
            scene, scene_classes = self.load_predicted_scene(
                sample['identifier']
            )
            sample_mira['scene'] = scene.item()
            sample_mira['scene_class'] = scene_classes.item()

        return sample_mira
