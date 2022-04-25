# -*- coding: utf-8 -*-
"""
Some common dataset tests

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import time

from nicr_scene_analysis_datasets import NYUv2
from nicr_scene_analysis_datasets import SUNRGBD
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


def test_caching():
    # as NYUv2 is quite small, we additionally test some functions
    dataset = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        sample_keys=('rgb', 'depth', 'semantic'),
        use_cache=True
    )

    simple_sums = []
    durations = []
    for _ in range(5):
        start = time.time()
        sum_ = 0
        for sample in dataset:
            sum_ += sample['rgb'].sum()
        end = time.time()

        simple_sums.append(sum_)
        durations.append(end-start)

        # print(simple_sums[-1], durations[-1])

    assert all(d < durations[0] for d in durations[1:])
    assert all(s == simple_sums[0] for s in simple_sums[1:])


def test_filter_camera_with_caching():
    # we use SunRGBD as it contains multiple cameras with samples of different
    # spatial resolution which may affect caching in a bad way

    # create dataset without caching and get shapes for each camera
    dataset_no_cache = SUNRGBD(
        dataset_path=DATASET_PATH_DICT['sunrgbd'],
        sample_keys=('rgb',),
        use_cache=False
    )

    shapes_no_cache = {}
    for camera in dataset_no_cache.cameras:
        with dataset_no_cache.filter_camera(camera):
            # get shape of first sample
            sample = dataset_no_cache[0]
            shapes_no_cache[camera] = sample['rgb'].shape

    # create dataset WITH caching and get shapes for each camera
    dataset_cache = SUNRGBD(
        dataset_path=DATASET_PATH_DICT['sunrgbd'],
        sample_keys=('rgb',),
        use_cache=True
    )

    shapes_cache = {}
    for camera in dataset_cache.cameras:
        with dataset_cache.filter_camera(camera):
            # get shape of first sample
            sample = dataset_cache[0]
            shapes_cache[camera] = sample['rgb'].shape

    assert shapes_no_cache == shapes_cache
