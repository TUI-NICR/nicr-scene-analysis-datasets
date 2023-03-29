# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from scipy.spatial.transform import Rotation

import mirapy
from PythonCameraIntrinsicWrapper import PinholeCameraIntrinsicNormalized
from PythonCameraIntrinsicWrapper import DepthCameraIntrinsicNormalized
from OrientedBoundingBoxWrapper import OrientedBoundingBox3f, VectorOrientedBoundingBox3f

from .. import Hypersim
from ._base_reader import MIRAReaderBase
from .utils import to_mira_img
from .utils import to_mira_img8u1


class HypersimReaderBase(MIRAReaderBase):
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
        add_member_roproperty(
            'DatasetUseDomesticSceneClasses',
            bool,
            self._rget_dataset_use_domestic_scene_classes,
            self._rset_dataset_use_domestic_scene_classes,
            "Use domestic scene classes (ground-truth scene only).",
            False
        )

    def parse_dataset(self):
        # simply retrieve all sample identifiers
        identifier_dataset = Hypersim(
            dataset_path=self._dataset_basepath,
            split=self._dataset_split,
            subsample=self._dataset_subsample,
            scene_use_indoor_domestic_labels=self._dataset_use_domestic_scene_classes,
            sample_keys=('identifier',)
        )
        for idx, s in enumerate(identifier_dataset):
            # unpack identifier, e.g., (ai_001_010, cam_00, 0000)
            scene, camera, id_ = s['identifier']

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
        self._dataset = Hypersim(
            dataset_path=self._dataset_basepath,
            split=self._dataset_split,
            subsample=self._dataset_subsample,
            scene_use_indoor_domestic_labels=self._dataset_use_domestic_scene_classes,
            sample_keys=(
                'identifier',
                *self._sample_keys
            )
        )

    def process_sample(self, sample):
        sample_mira = {}

        # small sanity check
        assert sample['identifier'][0] == self.cur_scene
        assert sample['identifier'][1] == self.cur_camera

        # extrinsic ------------------------------------------------------------
        if 'extrinsics' in sample:
            ext = sample['extrinsics']
            translation = mirapy.Point3f(ext['x'], ext['y'], ext['z'])
            rotation_quat = mirapy.Quaternionf()
            rotation_quat.x = ext['quat_x']
            rotation_quat.y = ext['quat_y']
            rotation_quat.z = ext['quat_z']
            rotation_quat.w = ext['quat_w']

            transform = mirapy.Pose3(translation, rotation_quat)
            # done in prepare_dataset.py as of > v051
            # transform *= mirapy.Pose3(0, 0, 0, 0, 0, np.deg2rad(180))
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
        if 'rgb' in sample:
            sample_mira['color_img'] = to_mira_img(
                sample['rgb'],
                rgb2bgr=True    # BGR for MIRA!
            )
        if 'depth' in sample:
            sample_mira['depth_img'] = to_mira_img(sample['depth'])

        # ground-truth semantic segmentation -----------------------------------
        if 'semantic' in sample:
            # we use a score of 0.999 for all pixels
            sample_mira['semantic_gt'] = to_mira_img(
                sample['semantic'].astype('float32') + 0.999
            )
            sample_mira['semantic_gt_classes'] = to_mira_img8u1(
                sample['semantic']    # < 255 classes
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
            sample_mira['instance_gt_ids'] = to_mira_img(
                sample['instance'].astype('uint16')    # < 65535 ids
            )

        # predicted instance segmentation --------------------------------------
        if self._load_predicted_instance:
            ins, ins_ids = self.load_predicted_instance(sample['identifier'])
            sample_mira['instance'] = to_mira_img(ins)
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

        # ground-truth 3d bounding boxes ---------------------------------------
        if '3d_boxes' in sample:
            box_vector = VectorOrientedBoundingBox3f()
            boxes = sample['3d_boxes']
            instance = sample['instance'].astype('uint16')
            semantic = sample['semantic']

            # For center transform of the boxes
            transform = mirapy.Pose3()

            # The box_name is equal to the instance id
            for box_name, box in boxes.items():
                box_name = int(box_name)

                instance_mask = instance == box_name
                assert np.sum(instance_mask) > 0
                semantic_classes, counts = np.unique(semantic[instance_mask], return_counts=True)
                max_count = counts.argmax()
                semantic_class = int(semantic_classes[max_count])

                extents = np.array(box["extents"])
                positions = np.array(box["positions"])
                orientation = Rotation.from_matrix(box["orientations"])

                origin = mirapy.Point3f(positions[0] - extents[0]/2,
                                        positions[1] - extents[1]/2,
                                        positions[2] - extents[2]/2)

                x_axis = mirapy.Point3f(positions[0] + extents[0]/2,
                                        positions[1] - extents[1]/2,
                                        positions[2] - extents[2]/2)

                y_axis = mirapy.Point3f(positions[0] - extents[0]/2,
                                        positions[1] + extents[1]/2,
                                        positions[2] - extents[2]/2)

                z_axis = mirapy.Point3f(positions[0] - extents[0]/2,
                                        positions[1] - extents[1]/2,
                                        positions[2] + extents[2]/2)

                rot = orientation.as_quat()

                transform.r.x = rot[0]
                transform.r.y = rot[1]
                transform.r.z = rot[2]
                transform.r.w = rot[3]

                box = OrientedBoundingBox3f(origin, x_axis, y_axis, z_axis)
                box.id = semantic_class
                box.center_transform(transform)
                box_vector.append(box)

            sample_mira['boxes'] = box_vector

        return sample_mira
