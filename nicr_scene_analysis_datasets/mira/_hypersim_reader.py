# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import signal
from time import time

import numpy as np

import mirapy
from PythonCameraIntrinsicWrapper import PinholeCameraIntrinsicNormalized
from PythonDepthCameraIntrinsicWrapper import DepthCameraIntrinsicNormalized
from PythonImageWrapper import Img
from PythonImageWrapper import Img8U1

from .. import Hypersim
from .utils import AutoGetterSetter
from .utils import to_mira_img


LOG_LEVEL = mirapy.WARNING


class HypersimReaderBase(mirapy.Unit, AutoGetterSetter):
    def __init__(self, feedback_channel_type=None):
        super().__init__()

        # use None if type does not matter (void channel)
        self._feedback_channel_type = feedback_channel_type

        # member ---------------------------------------------------------------
        self._dataset = None
        self._dataset_meta = {}
        self._scenes = None
        self._cur_scene_idx = -1
        self._cameras_in_scene = None
        self._cur_camera_idx = -1
        self._cur_idx = -1

        self._frame = None

        self._done = False
        self._start_time = -1
        self._iterating = False
        self._last_time = -1

        self._kill_time = -1

        self._ignore_next_feedback = False

        # reflected members ----------------------------------------------------
        self._dataset_basepath = None
        self._split = 'test'
        self._subsample = 1
        self._scene_filter_str = ''
        self._zero_depth_for_void = False
        self._start_delay = 5    # seconds
        self._paused = False
        self._watchdog_trigger_delay = 5
        self._feedback_channel = None
        self._reference_frame = '/GlobalFrame'
        self._load_predicted_segmentation = False
        self._predicted_segmentation_path = None
        self._predicted_segmentation_dir = 'semantic_40_topk'
        self._predicted_segmentation_topk = 3
        self._kill_when_done = False

        # subscribed channels --------------------------------------------------
        self._ch_feedback = None

        # published channels ---------------------------------------------------
        self._ch_color_instrinsic = None
        self._ch_depth_instrinsic = None
        self._ch_color_img = None
        self._ch_depth_img = None
        self._ch_segmentation = None
        self._chs_segmentation_classes = []

    @property
    def split(self):
        return self._split

    @property
    def cur_scene(self):
        if self._dataset is None:
            # dataset not parsed so far
            return ''
        return self._scenes[self._cur_scene_idx]

    @property
    def cur_camera(self):
        if self._dataset is None:
            # dataset not parsed so far
            return ''
        return self._cameras_in_scene[self._cur_camera_idx]

    @property
    def cur_basename(self):
        if self._dataset is None:
            # dataset not parsed so far
            return ''
        return self._dataset_meta[self.cur_scene][self.cur_camera][self.cur_idx][1]

    @property
    def cur_idx(self):
        if self._dataset is None:
            # dataset not parsed so far
            return -1
        return self._cur_idx

    @property
    def cur_n(self):
        if self._dataset is None:
            # dataset not parsed so far
            return -1
        return len(self._dataset_meta[self.cur_scene][self.cur_camera])

    @property
    def reference_frame(self):
        return self._reference_frame

    def reflect(self, r):
        r.member(
            'DatasetBasepath',
            str,
            self._rget_dataset_basepath,
            self._rset_dataset_basepath,
            "Path to nicr-scene-analysis-datasets folder for hypersim."
        )
        r.member(
            'Split',
            str,
            self._rget_split,
            self._rset_split,
            "Dataset split to use 'train', 'valid' or 'test'."
        )
        r.member(
            'Subsample',
            int,
            self._rget_subsample,
            self._rset_subsample,
            "Dataset subsampling to use: '1' (no subsampling), '2', '5', '10', "
            "or '20'"
        )
        r.member(
            'SceneFilter',
            str,
            self._rget_scene_filter_str,
            self._rset_scene_filter_str,
            "/-separated list of scenes to filter or empty for all scenes."
        )
        r.member(
            'ZeroDepthForVoid',
            bool,
            self._rget_zero_depth_for_void,
            self._rset_zero_depth_for_void,
            "Set depth value for void pixels (class index 0) to zero. This "
            "might be useful for subsequent processing steps such as mapping."
        )
        r.member(
            'StartDelay',
            float,
            self._rget_start_delay,
            self._rset_start_delay,
            "Number of seconds to wait before posting the first data."
        )
        r.property(
            'Paused',
            bool,
            self._rget_paused,
            self._set_paused,
            "Pause reader."
        )
        r.property(
            'WatchdogTriggerDelay',
            float,
            self._rget_watchdog_trigger_delay,
            self._rset_watchdog_trigger_delay,
            "Trigger next step if there was no new data on the feedback "
            "channel after a certain amount of seconds. '-1' disables the "
            "watchdog trigger."
        )
        r.member(
            'FeedbackChannel',
            str,
            self._rget_feedback_channel,
            self._rset_feedback_channel,
            "Channel to wait for data before posting the next frame."
        )
        r.member(
            'ReferenceFrame',
            str,
            self._rget_reference_frame,
            self._rset_reference_frame,
            "Reference frame for posted data."
        )
        r.member(
            'LoadPredictedSegmentation',
            bool,
            self._rget_load_predicted_segmentation,
            self._rset_load_predicted_segmentation,
            "Whether to load predicted segmentation as well. Note that the "
            "segmentation is not part of the actual dataset, see "
            "`PredictedSegmentationPath`."
        )
        r.member(
            'PredictedSegmentationPath',
            str,
            self._rget_predicted_segmentation_path,
            self._rset_predicted_segmentation_path,
            "Path to stored predicted segmentation."
        )
        r.member(
            'PredictedSegmentationDir',
            str,
            self._rget_predicted_segmentation_dir,
            self._rset_predicted_segmentation_dir,
            "Name of directory containing the predicted segmentation."
        )
        r.member(
            'PredictedSegmentationTopK',
            int,
            self._rget_predicted_segmentation_topk,
            self._rset_predicted_segmentation_topk,
            "TopK to use for loading predicted segmentation"
        )
        r.member(
            'KillMIRAWhenDone',
            bool,
            self._rget_kill_when_done,
            self._rset_kill_when_done,
            "Kill MIRA 5 seconds after dataset processing is done."
        )

        # for reporting current status
        r.roproperty(
            'CurrentScene',
            str,
            self._get_scene,
            "Current scene"
        )
        r.roproperty(
            'CurrentCamera',
            str,
            self._get_camera,
            "Current camera in scene"
        )
        r.roproperty(
            'Progress',
            str,
            self._get_progress,
            "Progress in current camera of current scene"
        )
        r.roproperty(
            'ProgressCameras',
            str,
            self._get_progress_cameras,
            "Progress in cameras of current scene"
        )
        r.roproperty(
            'ProgressScenes', str,
            self._get_progress_scenes,
            "Progress in scenes"
        )

    def _get_scene(self):
        return self.cur_scene

    def _get_camera(self):
        return self.cur_camera

    @staticmethod
    def _format_progress(n, total):
        return f'{n}/{total} ({(n)/total*100: 3.0f}%)'

    def _get_progress(self):
        if self._done:
            "Done"
        if not self._iterating:
            return ''
        return HypersimReaderBase._format_progress(self.cur_idx+1, self.cur_n)

    def _get_progress_cameras(self):
        if not self._iterating:
            return ''
        return HypersimReaderBase._format_progress(self._cur_camera_idx+1,
                                                   len(self._cameras_in_scene))

    def _get_progress_scenes(self):
        if not self._iterating:
            return ''
        return HypersimReaderBase._format_progress(self._cur_scene_idx+1,
                                                   len(self._dataset_meta))

    def _set_paused(self, value):
        self._paused = value
        if not self._paused and self._iterating:
            self.start()

    def initialize(self):
        # publish frames -------------------------------------------------------
        parent_frame = self.resolveName(self._reference_frame)

        self._frame = self.resolveName('ImageFrame')
        self.addTransformLink(self._frame, parent_frame)

        # publish channels -----------------------------------------------------
        self.bootup("Publishing channels")
        self._ch_color_instrinsic = self.publish(
            'ColorIntrinsic',
            PinholeCameraIntrinsicNormalized
        )
        self._ch_depth_instrinsic = self.publish(
            'DepthIntrinsic',
            DepthCameraIntrinsicNormalized
        )
        self._ch_color_img = self.publish('ColorImage', Img)
        self._ch_depth_img = self.publish('DepthImage', Img)
        self._ch_gt = self.publish('GroundTruth', Img)
        self._ch_gt_classes = self.publish('GroundTruthClasses', Img8U1)
        if self._load_predicted_segmentation:
            self._ch_segmentation = self.publish('Segmentation', Img)
            for i in range(self._predicted_segmentation_topk):
                ch = self.publish(f'SegmentationClasses_{i}', Img8U1)
                self._chs_segmentation_classes.append(ch)

        # subscribe channels ---------------------------------------------------
        self.bootup("Subscribing channels")
        self._ch_feedback = self.subscribe(
            self._feedback_channel,
            self._feedback_channel_type,
            self.cb_feedback
        )

        # post intrinsics ------------------------------------------------------
        self.bootup("Posting rgb and depth intrinsic")

        # there is only one camera, thus, intrinsics do not change
        # note: both are already normalized
        camera = Hypersim.CAMERAS[0]
        color_intrinsics = Hypersim.RGB_INTRINSICS_NORMALIZED[camera]
        depth_intrinsics = Hypersim.DEPTH_INTRINSICS_NORMALIZED[camera]

        color_intrinsic = PinholeCameraIntrinsicNormalized(
            color_intrinsics['fx'], color_intrinsics['fy'],
            color_intrinsics['cx'], color_intrinsics['cy'],
            color_intrinsics['k1'], color_intrinsics['k2'],
            color_intrinsics['p1'], color_intrinsics['p2']
        )
        self._ch_color_instrinsic.post(color_intrinsic)

        # depth camera intrinsic (same as rgb but with parameter a)
        depth_intrinsic = DepthCameraIntrinsicNormalized(
            color_intrinsic, depth_intrinsics['a']
        )
        self._ch_depth_instrinsic.post(depth_intrinsic)

        # parse dataset --------------------------------------------------------
        self.bootup("Parsing dataset")

        # just retrieve all sample identifiers and parse dataset
        dataset = Hypersim(
            dataset_path=self._dataset_basepath,
            split=self._split,
            subsample=self._subsample,
            sample_keys=('identifier',)
        )
        for idx, s in enumerate(dataset):
            scene, camera, id_ = s['identifier']

            # apply scene filter
            if self._scene_filter_str:
                if scene not in self._scene_filter_str:
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
            split=self._split,
            subsample=self._subsample,
            sample_keys=('identifier', 'extrinsics',
                         'rgb', 'depth',
                         'semantic')
        )

        mirapy.log(LOG_LEVEL, f"{len(self._dataset_meta)} scenes found.")

    def reset(self, scene_idx=0, camera_idx=0, idx=0):
        # set up scenes and select first
        self._scenes = list(self._dataset_meta.keys())
        self._cur_scene_idx = scene_idx

        # set up cameras is current scene and select first
        self._cameras_in_scene = list(self._dataset_meta[self.cur_scene].keys())
        self._cur_camera_idx = camera_idx

        # set up current images in camera
        self._cur_idx = idx-1    # -1 since index is first incremented

        self._start_time = -1
        self._last_time = -1

        self._iterating = False
        self._done = False

    def start(self):
        if not self._iterating:
            # we start iterating
            self.cb_dataset_start()

            self.cb_scene_start()
            self.cb_camera_start()

        # start / resume
        self._iterating = True

        # self.process_next_frame()
        # let the watchdog start iterating to ensure everything is loaded
        mirapy.log(LOG_LEVEL, f"Watchdog will start iterating soon.")
        self._last_time = time()

    def pause_resume(self, pause=None):
        if pause is None:
            self._paused = not self._paused
        else:
            self._paused = pause

    def ignore_next_feedback(self):
        self._ignore_next_feedback = True

    def cb_scene_start(self):
        pass

    def cb_scene_end(self):
        pass

    def cb_camera_start(self):
        pass

    def cb_camera_end(self):
        pass

    def cb_dataset_start(self):
        pass

    def cb_dataset_end(self):
        mirapy.log(LOG_LEVEL, f"Done")
        if self._kill_when_done:
            self._kill_time = time() + 5

    def cb_watchdog_triggered(self):
        pass

    def cb_feedback(self, channel_read):
        if self._ignore_next_feedback:
            self._ignore_next_feedback = False
            mirapy.log(LOG_LEVEL, f"Feedback skipped")
            return
        self.process_next_frame()

    def load_predicted_segmentation(self, identifier):
        fp = os.path.join(self._predicted_segmentation_path,
                          self._split,
                          self._predicted_segmentation_dir,
                          *identifier)
        fp += '.npy'

        # segmentation is of shape (topk, h, w) with each element equal to
        # class+score, while it is ensured that score < 1
        segmentation = np.load(fp)

        # limit to topk
        if segmentation.shape[0] < self._predicted_segmentation_topk:
            mirapy.log(LOG_LEVEL,
                       "`PredictedSegmentationTopK` is larger than the number "
                       f"of channels in the loaded segmentaton: '{fp}'.")
        segmentation = segmentation[:self._predicted_segmentation_topk, ...]

        # convert to channels last
        segmentation = segmentation.transpose(1, 2, 0)
        segmentation = np.ascontiguousarray(segmentation)    # <- important

        # note, we do not load the images for the classes!
        segmentation_classes = segmentation.astype('uint8')    # < 256 classes

        return segmentation, segmentation_classes

    def process_next_frame(self):
        if self._done:
            return

        if self._paused:
            return

        if not self._iterating:
            return

        if self._cur_idx+1 == self.cur_n:
            # go to next camera / scene
            if self._cur_camera_idx < len(self._cameras_in_scene)-1:
                # end of camera reached
                self.cb_camera_end()

                if not self._iterating:
                    # reset was called in callback
                    return

                # go to next camera
                self._cur_camera_idx += 1
                self._cur_idx = -1
                self.cb_camera_start()

                if self._watchdog_trigger_delay == -1:
                    # there is no watchdog, so trigger next frame
                    return self.process_next_frame()
                return
            else:
                # go to next scene
                if self._cur_scene_idx < len(self._scenes)-1:
                    # end of camera and scene reached
                    self.cb_camera_end()
                    self.cb_scene_end()

                    if not self._iterating:
                        # reset was called in callback
                        return

                    self._cur_scene_idx += 1
                    self._cameras_in_scene = \
                        list(self._dataset_meta[self.cur_scene].keys())
                    self.cb_scene_start()
                    self._cur_camera_idx = 0
                    self._cur_idx = -1
                    self.cb_camera_start()

                    if self._watchdog_trigger_delay == -1:
                        # there is no watchdog, so trigger next frame
                        return self.process_next_frame()
                    return

                else:
                    # end of camera, scene, and dataset reached
                    self._done = True
                    self.cb_camera_end()
                    self.cb_scene_end()

                    if not self._iterating:
                        # reset was called in callback
                        return

                    self.cb_dataset_end()
                    self._iterating = False
                    return

        # get new sample
        self._cur_idx += 1

        time_now = mirapy.now()
        dataset_idx = self._dataset_meta[self.cur_scene][self.cur_camera][self._cur_idx][0]
        sample = self._dataset[dataset_idx]

        assert sample['identifier'][0] == self.cur_scene
        assert sample['identifier'][1] == self.cur_camera

        sample_identifier = '/'.join(sample['identifier'])
        mirapy.log(LOG_LEVEL, f"Processing {sample_identifier}")

        # extrinsic
        ext = sample['extrinsics']
        translation = mirapy.Point3f(ext['x'], ext['y'], ext['z'])
        rotation_quat = mirapy.Quaternionf()
        rotation_quat.x = ext['quat_x']
        rotation_quat.y = ext['quat_y']
        rotation_quat.z = ext['quat_z']
        rotation_quat.w = ext['quat_w']

        transform = mirapy.Pose3(translation, rotation_quat)
        transform *= mirapy.Pose3(0, 0, 0, 0, 0, np.deg2rad(180))
        self.publishTransform3(self._frame, transform)

        # images
        color_img = sample['rgb']
        depth_img = sample['depth'].astype('float32')

        # ground truth segmentation (use score of 0.999 for all pixels)
        gt = sample['semantic'].astype('float32') + 0.999
        gt_classes = sample['semantic']

        # predicted segmentation
        if self._load_predicted_segmentation:
            segmentation, segmentation_classes = \
                self.load_predicted_segmentation(sample['identifier'])

        # set depth to zero for void pixels
        if self._zero_depth_for_void:
            depth_img[gt_classes == 0] = 0

        # wrap using MIRA Img
        color_img_mira = to_mira_img(color_img, rgb2bgr=True)    # BGR for MIRA!
        depth_img_mira = to_mira_img(depth_img)
        gt_mira = to_mira_img(gt)
        gt_classes_mira = Img8U1(gt_classes.shape[1], gt_classes.shape[0])
        gt_classes_mira.setMat(gt_classes)
        if self._load_predicted_segmentation:
            segmentation_mira = to_mira_img(segmentation)
            segmentation_classes_mira = []
            for i in range(segmentation.shape[-1]):
                img_mira = Img8U1(segmentation_classes.shape[1],
                                  segmentation_classes.shape[0])
                img_mira.setMat(
                    np.ascontiguousarray(segmentation_classes[..., i])
                )
                segmentation_classes_mira.append(img_mira)

        # post images
        self._ch_color_img.post(color_img_mira, time_now, self._frame)
        self._ch_depth_img.post(depth_img_mira, time_now, self._frame)
        self._ch_gt.post(gt_mira, time_now, self._frame)
        self._ch_gt_classes.post(gt_classes_mira, time_now, self._frame)
        if self._load_predicted_segmentation:
            self._ch_segmentation.post(segmentation_mira, time_now, self._frame)
            for ch, img_mira in zip(self._chs_segmentation_classes,
                                    segmentation_classes_mira):
                ch.post(img_mira, time_now, self._frame)

        self._last_time = time()

    def process(self):
        if self._done:
            # iterating done, check whether to kill mira process
            if self._kill_when_done:
                if self._kill_time > time():
                    time_left = self._kill_time - time()
                    mirapy.log(LOG_LEVEL,
                               f"Killing MIRA in {time_left:.1f} second(s)")
                else:
                    # kill MIRA
                    os.kill(os.getpid(), signal.SIGKILL)
            return

        if self._paused:
            # nothing to do for now since reader is paused
            return

        if not self._iterating:
            # we are not iterating, check whether to start
            if self._start_time == -1:
                # start time not set or resetted
                self._start_time = time()

            # start processing after some initial delay
            if self._start_time+self._start_delay > time():
                # we still have to wait
                time_left = self._start_time+self._start_delay - time()
                mirapy.log(LOG_LEVEL,
                           f"Waiting ({time_left:.1f} second(s) left)")
                return
            # start processing
            self.reset()
            self.start()
        else:
            # reader is running, check if we have to trigger the next frame
            # based on watchdog
            if self._watchdog_trigger_delay != -1:
                # watchdog trigger is enabled
                if self._last_time+self._watchdog_trigger_delay < time():
                    mirapy.log(LOG_LEVEL,
                               "Triggering next frame using watchdog since no "
                               "feedback was received")
                    self.cb_watchdog_triggered()
                    self.process_next_frame()

    def finalize(self):
        pass
