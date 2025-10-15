# -*- coding: utf-8 -*-
"""
Some common dataset tests

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import time

import numpy as np
import pytest

from nicr_scene_analysis_datasets import KNOWN_DATASETS
from nicr_scene_analysis_datasets import SUNRGBD
from nicr_scene_analysis_datasets import NYUv2
from nicr_scene_analysis_datasets import get_dataset_class
from nicr_scene_analysis_datasets.scripts.generate_auxiliary_data \
    import SCENE_INDOOR_DOMESTIC_DATASETS
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


def test_filter_camera():
    # we use SunRGBD as it contains multiple cameras with samples of different
    # spatial resolution which may affect caching in a bad way

    # create dataset
    dataset = SUNRGBD(
        dataset_path=DATASET_PATH_DICT['sunrgbd'],
        sample_keys=('rgb',),
        use_cache=False
    )
    n_samples = len(dataset)

    # test filter_camera with context manager
    camera = 'kv1'
    with dataset.filter_camera(camera):
        assert dataset.camera == camera
        assert len(dataset) == 1073     # yes, this number is fixed ;)
    # everything should be back to normal
    assert dataset.camera is None
    assert len(dataset) == n_samples

    # test filter_camera without context manager
    camera = 'kv1'
    dataset.filter_camera(camera)
    assert dataset.camera == camera
    assert len(dataset) == 1073     # yes, this number is fixed ;)
    dataset.filter_camera(None)
    # everything should be back to normal
    assert dataset.camera is None
    assert len(dataset) == n_samples


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


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
def test_datatypes(dataset_name):
    # In this test case we check if the datatypes of the samples are correct
    # for all datasets.

    # Get the dataset class and prepare the arguments
    dataset = get_dataset_class(dataset_name)
    train_split_key = 'train'
    if dataset_name == 'ade20k':
        train_split_key = 'train_panoptic_2017'
    dataset_args = {
        'dataset_path': DATASET_PATH_DICT[dataset_name],
        'sample_keys': dataset.get_available_sample_keys(train_split_key),
        'use_cache': False
    }

    # Get the number of semantic classes.
    semantic_n_classes = dataset.SEMANTIC_N_CLASSES
    # Check if there are multiple semantic classes, as we want to test all
    # of them.
    has_multiple_semantic_classes = \
        isinstance(semantic_n_classes, (list, tuple))
    # If semantic_n_classes is not iterable, make it iterable, so we can
    # we can use it in the for loop.
    if not has_multiple_semantic_classes:
        semantic_n_classes = (semantic_n_classes,)

    for n_classes in semantic_n_classes:
        if has_multiple_semantic_classes:
            # If we have multiple semantic classes, we need to set the
            # semantic_n_classes argument for each iteration.
            dataset_args['semantic_n_classes'] = n_classes

        # Create the dataset instance.
        dataset_instance = dataset(**dataset_args)

        for idx, sample in enumerate(dataset_instance):
            if idx >= 9:
                break

            # Ensure that the rgb, depth, semantic and instance samples
            # have the correct datatype.
            rgb = sample['rgb']
            assert rgb.dtype == np.uint8

            # Not all datasets have depth
            if 'depth' in sample:
                depth = sample['depth']
                # The Cityscapes dataset has float32 depth values, while
                # all other datasets have uint16 depth values.
                if dataset_name == 'cityscapes':
                    assert depth.dtype == np.float32
                else:
                    assert depth.dtype == np.uint16

            # If the number of semantic classes is less than 256, the
            # semantic values should be uint8, otherwise uint16.
            semantic = sample['semantic']
            if n_classes <= 255:
                assert semantic.dtype == np.uint8
            else:
                assert semantic.dtype == np.uint16

            instance = sample['instance']
            assert instance.dtype == np.uint16


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
@pytest.mark.parametrize('with_auxiliary_data', [True, False])
def test_dataset_config(dataset_name, with_auxiliary_data):
    Dataset = get_dataset_class(
        dataset_name, with_auxiliary_data=with_auxiliary_data
    )
    kwargs = {}
    kwargs['dataset_path'] = DATASET_PATH_DICT[dataset_name]
    dataset_n_classes = Dataset.SEMANTIC_N_CLASSES
    dataset_n_classes_was_int = False
    if isinstance(dataset_n_classes, int):
        dataset_n_classes = (dataset_n_classes,)
        dataset_n_classes_was_int = True
    for n_classes in dataset_n_classes:
        if not dataset_n_classes_was_int:
            kwargs['semantic_n_classes'] = n_classes

        # Prepare configurations to test - start with default config
        configs = [kwargs.copy()]

        # Add indoor domestic configuration if supported
        if dataset_name in SCENE_INDOOR_DOMESTIC_DATASETS:
            domestic_kwargs = kwargs.copy()
            domestic_kwargs['scene_use_indoor_domestic_labels'] = True
            configs.append(domestic_kwargs)

        # Test all configurations
        for config in configs:
            # Create dataset with current configuration
            dataset = Dataset(**config)
            available_sample_keys = dataset.get_available_sample_keys(dataset.split)

            # Test semantic labels if available
            if 'semantic' in available_sample_keys:
                assert len(dataset.config.semantic_label_list_without_void) == n_classes
                assert len(dataset.config.semantic_label_list) == n_classes + 1

            # Test scene labels if available
            if 'scene' in available_sample_keys:
                # Get label list information
                scene_label_list = dataset.config.scene_label_list
                # Verify label list exists
                assert scene_label_list is not None
                # Get class names from label list
                class_names = scene_label_list.class_names
                # Verify class names exist
                assert len(class_names) > 0
