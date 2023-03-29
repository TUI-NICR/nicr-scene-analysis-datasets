# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import datetime
import os
import signal
from time import time

import numpy as np

import mirapy
from PythonCameraIntrinsicWrapper import PinholeCameraIntrinsicNormalized
from PythonCameraIntrinsicWrapper import DepthCameraIntrinsicNormalized
from PythonImageWrapper import Img
from PythonImageWrapper import Img8U1
from OrientedBoundingBoxWrapper import VectorOrientedBoundingBox3f

from .utils import AutoGetterSetter


LOG_LEVEL = mirapy.WARNING


class MIRAReaderBase(mirapy.Unit, AutoGetterSetter):
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

        self._total_cur_idx = -1
        self._total_n = None

        self._frame = None
        self._global_frame = None

        self._done = False
        self._start_time = -1
        self._iterating = False
        self._iterating_start_time = -1
        self._last_time = -1

        self._kill_time = -1

        self._ignore_next_feedback = False

        # reflected members ----------------------------------------------------
        self._dataset_basepath = None
        self._dataset_split = None
        self._dataset_subsample = None
        self._dataset_filter_str = None
        self._start_delay = None
        self._paused = None
        self._watchdog_trigger_delay = None
        self._feedback_channel = None
        self._reference_frame = None
        self._register_rgb_to_depth = None
        self._load_predicted_semantic = None
        self._predicted_semantic_path = None
        self._predicted_semantic_topk = None
        self._load_predicted_instance = None
        self._predicted_instance_path = None
        self._load_predicted_scene = None
        self._predicted_scene_path = None
        self._kill_when_done = None

        # subscribed channels --------------------------------------------------
        self._ch_feedback = None

        # published channels ---------------------------------------------------
        self._ch_color_instrinsic = None
        self._ch_depth_instrinsic = None
        self._ch_color_img = None
        self._ch_depth_img = None
        self._ch_sem = None
        self._chs_sem_classes = []
        self._ch_ins_gt = None
        self._ch_ins_gt_ids = None
        self._ch_ins = None
        self._ch_ins_ids = None
        self._ch_scene_gt = None
        self._ch_scene_gt_class = None
        self._ch_scene = None
        self._ch_scene_class = None
        self._ch_boxes_gt = None

    @property
    def split(self):
        return self._dataset_split

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
    def total_cur_idx(self):
        if self._dataset is None:
            # dataset not parsed so far
            return -1
        return self._total_cur_idx

    @property
    def total_n(self):
        if self._dataset is None:
            # dataset not parsed so far
            return -1

        if self._total_n is None:
            # get total_n lazily
            self._total_n = 0
            for scene in self._dataset_meta:
                for camera in self._dataset_meta[scene]:
                    self._total_n += len(self._dataset_meta[scene][camera])

        return self._total_n

    @property
    def reference_frame(self):
        return self._reference_frame

    def reflect(self, r):
        def add_member_roproperty(name, type_, getter, setter, desc, *args):
            # use args to pass additional arguments such as a default
            r.member(name, type_, getter, setter, desc, *args)
            r.roproperty(name, type_, getter, desc)

        # dataset-related properties -------------------------------------------
        add_member_roproperty(
            'DatasetBasepath',
            str,
            self._rget_dataset_basepath,
            self._rset_dataset_basepath,
            "Path to nicr-scene-analysis-datasets-* folder."
        )
        add_member_roproperty(
            'DatasetSplit',
            str,
            self._rget_dataset_split,
            self._rset_dataset_split,
            "Dataset split to use 'train', 'valid' or 'test'.",
            'test'
        )
        add_member_roproperty(
            'DatasetSubsample',
            int,
            self._rget_dataset_subsample,
            self._rset_dataset_subsample,
            "Dataset subsample to use.",
            10
        )
        add_member_roproperty(
            'DatasetFilter',
            str,
            self._rget_dataset_filter_str,
            self._rset_dataset_filter_str,
            "Comma-separated list of identifier filters or empty for all "
            "scenes, e.g., 'ai_001_010,ai_003_007', 'ai_001_010/cam_02', "
            "'ai_001_010/cam_02/0000', or 'scene0707/00'",
        )

        # predicted semantic ---------------------------------------------------
        add_member_roproperty(
            'LoadPredictedSemantic',
            bool,
            self._rget_load_predicted_semantic,
            self._rset_load_predicted_semantic,
            "Whether to load predicted semantic segmentation as well. Note "
            "that the segmentation is not part of the actual dataset, see "
            "`PredictedSemanticPath`.",
            False
        )
        add_member_roproperty(
            'PredictedSemanticPath',
            str,
            self._rget_predicted_semantic_path,
            self._rset_predicted_semantic_path,
            "Path to stored predicted semantic segmentation.",
            ''
        )
        add_member_roproperty(
            'PredictedSemanticTopK',
            int,
            self._rget_predicted_semantic_topk,
            self._rset_predicted_semantic_topk,
            "TopK to use for loading predicted semantic segmentation",
            1
        )
        # predicted instance ---------------------------------------------------
        add_member_roproperty(
            'LoadPredictedInstance',
            bool,
            self._rget_load_predicted_instance,
            self._rset_load_predicted_instance,
            "Whether to load predicted instance segmentation as well. Note "
            "that the segmentation is not part of the actual dataset, see "
            "`PredictedInstancePath`.",
            False
        )
        add_member_roproperty(
            'PredictedInstancePath',
            str,
            self._rget_predicted_instance_path,
            self._rset_predicted_instance_path,
            "Path to stored predicted instance segmentation.",
            ''
        )
        # predicted instance ---------------------------------------------------
        add_member_roproperty(
            'LoadPredictedScene',
            bool,
            self._rget_load_predicted_scene,
            self._rset_load_predicted_scene,
            "Whether to load predicted scene classes as well. Note "
            "that the predictions is not part of the actual dataset, see "
            "`PredictedScenePath`.",
            False
        )
        add_member_roproperty(
            'PredictedScenePath',
            str,
            self._rget_predicted_scene_path,
            self._rset_predicted_scene_path,
            "Path to stored predicted scene classes.",
            ''
        )

        # other ----------------------------------------------------------------
        add_member_roproperty(
            'StartDelay',
            float,
            self._rget_start_delay,
            self._rset_start_delay,
            "Number of seconds to wait before posting the first data.",
            5
        )
        r.property(
            'Paused',
            bool,
            self._rget_paused,
            self._set_paused,
            "Pause reader.",
            False
        )
        r.property(
            'WatchdogTriggerDelay',
            float,
            self._rget_watchdog_trigger_delay,
            self._rset_watchdog_trigger_delay,
            "Trigger next step if there was no new data on the feedback "
            "channel after a certain amount of seconds. '-1' disables the "
            "watchdog trigger.",
            5
        )
        add_member_roproperty(
            'FeedbackChannel',
            str,
            self._rget_feedback_channel,
            self._rset_feedback_channel,
            "Channel to wait for data before posting the next frame."
        )
        add_member_roproperty(
            'ReferenceFrame',
            str,
            self._rget_reference_frame,
            self._rset_reference_frame,
            "Reference frame for posted data.",
            '/GlobalFrame'
        )
        add_member_roproperty(
            'KillMIRAWhenDone',
            bool,
            self._rget_kill_when_done,
            self._rset_kill_when_done,
            "Kill MIRA 5 seconds after dataset processing is done.",
            False
        )

        # for reporting current status -----------------------------------------
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
        r.roproperty(
            'ProgressTotal', str,
            self._get_progress_total,
            "Total progress across all scenes and cameras"
        )

        r.roproperty(
            'ETA', str,
            self._get_progress_total_eta,
            "Total time left."
        )

    def _get_scene(self):
        return self.cur_scene

    def _get_camera(self):
        return self.cur_camera

    @staticmethod
    def _format_progress(n, total):
        return f'{n}/{total} ({(n)/total*100: 3.0f}%)'

    def _format_eta(self, idx, total, start_time=None):
        if start_time is None:
            # get global start time
            start_time = self._iterating_start_time

        if start_time is None or start_time == -1:
            # we do not have a start time
            return "N/A"
        avg_time_sample = (time() - start_time) / idx
        time_left = (total - idx) * avg_time_sample

        return str(datetime.timedelta(seconds=time_left))

    def _get_progress(self):
        if self._done:
            "Done"
        if not self._iterating:
            return ''
        return MIRAReaderBase._format_progress(self.cur_idx+1, self.cur_n)

    def _get_progress_cameras(self):
        if not self._iterating:
            return ''
        return MIRAReaderBase._format_progress(self._cur_camera_idx+1,
                                               len(self._cameras_in_scene))

    def _get_progress_scenes(self):
        if not self._iterating:
            return ''
        return MIRAReaderBase._format_progress(self._cur_scene_idx+1,
                                               len(self._dataset_meta))

    def _get_progress_total(self):
        if not self._iterating:
            return ''
        return MIRAReaderBase._format_progress(self.total_cur_idx+1,
                                               self.total_n)

    def _get_progress_total_eta(self):
        if not self._iterating:
            return ''
        return self._format_eta(self.total_cur_idx+1, self.total_n)

    def _set_paused(self, value):
        self._paused = value
        if not self._paused and self._iterating:
            self.start()

    # @abc.abstractmethod
    def parse_dataset(self):
        # needs to be implemented in derived class
        # must set self._dataset_meta
        # must set self._dataset

        # as abc meta stuff does not work with mirapy/boost, we simply raise an
        # error here
        raise NotImplementedError("Needs to be implemented in derived class")

    def initialize(self):
        # publish frames -------------------------------------------------------
        parent_frame = self.resolveName(self._reference_frame)

        self._frame = self.resolveName('ImageFrame')
        self._global_frame = self.resolveName('/GlobalFrame')
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

        # semantic
        self._ch_sem_gt = self.publish('SemanticGT', Img)
        self._ch_sem_gt_classes = self.publish('SemanticGTClasses', Img8U1)

        # predicted semantic
        if self._load_predicted_semantic:
            self._ch_sem = self.publish('Semantic', Img)
            for i in range(self._predicted_semantic_topk):
                ch = self.publish(f'SemanticClasses_{i}', Img8U1)
                self._chs_sem_classes.append(ch)

        # instance
        self._ch_ins_gt = self.publish('InstanceGT', Img)
        self._ch_ins_gt_ids = self.publish('InstacneGTIds', Img)

        # predicted instance
        if self._load_predicted_instance:
            self._ch_ins = self.publish('Instance', Img)
            self._ch_ins_ids = self.publish('InstanceIds', Img)

        # scene
        self._ch_scene_gt = self.publish('SceneGT', float)
        self._ch_scene_gt_class = self.publish('SceneGTClass', int)

        # boxes
        self._ch_boxes_gt = self.publish('BoxesGT', VectorOrientedBoundingBox3f)

        # predicted instance
        if self._load_predicted_scene:
            self._ch_scene = self.publish('Scene', float)
            self._ch_scene_class = self.publish('SceneClass', int)

        # subscribe channels ---------------------------------------------------
        self.bootup("Subscribing channels")
        self._ch_feedback = self.subscribe(
            self._feedback_channel,
            self._feedback_channel_type,
            self.cb_feedback
        )

        # parse dataset --------------------------------------------------------
        self.bootup("Parsing dataset")
        self.parse_dataset()

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

        self._total_cur_idx = -1
        self._start_time = -1
        self._last_time = -1

        self._iterating = False
        self._iterating_start_time = -1
        self._done = False

    def start(self):
        if not self._iterating:
            # we start iterating
            self._iterating_start_time = time()
            self.cb_dataset_start()

            self.cb_scene_start()
            self.cb_camera_start()

        # start / resume
        self._iterating = True

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
        # might be implemented in derived class
        pass

    def cb_scene_end(self):
        # might be implemented in derived class
        pass

    def cb_camera_start(self):
        # might be implemented in derived class
        pass

    def cb_camera_end(self):
        # might be implemented in derived class
        pass

    def cb_dataset_start(self):
        # might be implemented in derived class
        pass

    def cb_dataset_end(self):
        # might be extended in derived class

        mirapy.log(LOG_LEVEL, f"Done")
        if self._kill_when_done:
            self._kill_time = time() + 5

    def cb_watchdog_triggered(self):
        # might be implemented in derived class
        pass

    def cb_feedback(self, channel_read):
        # might be extended in derived class

        if self._ignore_next_feedback:
            self._ignore_next_feedback = False
            mirapy.log(LOG_LEVEL, f"Feedback skipped")
            return
        self.process_next_frame()

    def load_predicted_semantic(self, identifier):
        fp = os.path.join(self._predicted_semantic_path, *identifier)
        fp += '.npz'

        # semantic is of shape (topk, h, w) with each element equal to
        # class+score with score in [0, 0.999]
        semantic = np.load(fp)['arr_0']

        if semantic.ndim == 2:
            # add topk dimension
            semantic = semantic[None, ...]

        # limit to topk
        if semantic.shape[0] < self._predicted_semantic_topk:
            mirapy.log(LOG_LEVEL,
                       "`PredictedSemanticTopK` is larger than the number "
                       f"of channels in the loaded semantic: '{fp}'.")
        semantic = semantic[:self._predicted_semantic_topk, ...]

        # derive classes
        semantic_classes = semantic.astype('uint8')    # < 256 classes

        return semantic, semantic_classes

    def load_predicted_instance(self, identifier):
        fp = os.path.join(self._predicted_instance_path, *identifier)
        fp += '.npz'

        # segmentation is of shape (h, w) with each element equal to
        # instance_id+score with score in [0, 0.999], 0 mean 'no instance'/bg
        instance = np.load(fp)['arr_0']

        # derive ids
        instance_ids = instance.astype('uint16')    # < 65535 classes

        return instance, instance_ids

    def load_predicted_scene(self, identifier):
        fp = os.path.join(self._predicted_scene_path, *identifier)
        fp += '.npz'

        # scene is encoded as class+score with score in [0, 0.999]
        scene = np.load(fp)['arr_0']

        # derive ids
        scene_class = scene.astype('uint8')    # < 256 classes

        return scene, scene_class

    # @abc.abstractmethod
    def process_sample(self, sample):
        # needs to be implemented in derived class

        # as abc meta stuff does not work with mirapy/boost, we simply raise an
        # error here
        raise NotImplementedError("Needs to be implemented in derived class")

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
        self._total_cur_idx += 1
        self._last_time = time()
        time_now = mirapy.now()

        dataset_idx = self._dataset_meta[self.cur_scene][self.cur_camera][self._cur_idx][0]
        sample = self._dataset[dataset_idx]

        # process sample
        sample_identifier = '/'.join(sample['identifier'])
        mirapy.log(
            LOG_LEVEL,
            f"[{self._get_progress_total()}, "
            f"eta: {self._get_progress_total_eta()}, "
            f"cur: {self._get_progress()}, "
            f"cameras: {self._get_progress_cameras()}, "
            f"scenes: {self._get_progress_scenes()}] "
            f"Processing {sample_identifier}")

        sample_mira = self.process_sample(sample)

        if sample_mira is None:
            # something went wrong while loading, stop here, try next sample
            mirapy.log(
                mirapy.ERROR,
                f"Skipped {sample_identifier} (process_sample failed)"
            )
            return self.process_next_frame()

        # post extrinsic
        if 'extrinsic' in sample_mira:
            self.publishTransform3(self._frame, sample_mira['extrinsic'],
                                   time_now)

        # post intrinsic (only if changed to avoid recreating LUTs in MIRA)
        if 'color_intrinsic' in sample_mira:
            self._ch_color_instrinsic.post(sample_mira['color_intrinsic'],
                                           time_now)
        if 'depth_intrinsic' in sample_mira:
            self._ch_depth_instrinsic.post(sample_mira['depth_intrinsic'],
                                           time_now)

        # post images
        def _post(channel, data):
            channel.post(data, time_now, self._frame)

        if 'color_img' in sample_mira:
            _post(self._ch_color_img, sample_mira['color_img'])
        if 'depth_img' in sample_mira:
            _post(self._ch_depth_img, sample_mira['depth_img'])

        if 'semantic_gt' in sample_mira:
            _post(self._ch_sem_gt, sample_mira['semantic_gt'])
            _post(self._ch_sem_gt_classes, sample_mira['semantic_gt_classes'])
        if self._load_predicted_semantic:
            _post(self._ch_sem, sample_mira['semantic'])
            for ch, img_mira in zip(self._chs_sem_classes,
                                    sample_mira['semantic_classes']):
                _post(ch, img_mira)

        if 'instance_gt' in sample_mira:
            _post(self._ch_ins_gt, sample_mira['instance_gt'])
            _post(self._ch_ins_gt_ids, sample_mira['instance_gt_ids'])
        if self._load_predicted_instance:
            _post(self._ch_ins, sample_mira['instance'])
            _post(self._ch_ins_ids, sample_mira['instance_ids'])

        if 'scene_gt' in sample_mira:
            _post(self._ch_scene_gt, sample_mira['scene_gt'])
            _post(self._ch_scene_gt_class, sample_mira['scene_gt_class'])

        if 'boxes_gt' in sample_mira:
            self._ch_boxes_gt.post(sample_mira['boxes_gt'], time_now,
                                   self._global_frame)

        if self._load_predicted_scene:
            _post(self._ch_scene, sample_mira['scene'])
            _post(self._ch_scene_class, sample_mira['scene_class'])

    def process(self):
        cur_time = time()

        if self._done:
            # iterating done, check whether to kill mira process
            if self._kill_when_done:
                if self._kill_time > cur_time:
                    time_left = self._kill_time - cur_time
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
                # start time not set or reset
                self._start_time = cur_time

            # start processing after some initial delay
            if self._start_time+self._start_delay > cur_time:
                # we still have to wait
                time_left = self._start_time+self._start_delay - cur_time
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
                if (self._last_time+self._watchdog_trigger_delay) < cur_time:
                    mirapy.log(LOG_LEVEL,
                               "Triggering next frame using watchdog since no "
                               "feedback was received")
                    self.cb_watchdog_triggered()
                    self.process_next_frame()

    def finalize(self):
        pass
