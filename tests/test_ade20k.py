# -*- coding: utf-8 -*-
"""
Simple (interface) tests for ADE20k dataset

.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""

import numpy as np
import pytest

from nicr_scene_analysis_datasets import ADE20K
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

# Constants based on ADE20K dataset details
SPLITS = ADE20K.SPLITS
# SEMANTIC_N_CLASSES = (150, 3688)
CAMERAS = ('683x512', '674x512')


@pytest.mark.parametrize('split', SPLITS)
def test_dataset_initialization(split):
    # Initialize dataset
    dataset = ADE20K(
        dataset_path=DATASET_PATH_DICT['ade20k'],
        split=split,
        sample_keys=ADE20K.get_available_sample_keys(split),
    )

    # Check basic properties
    assert dataset.split == split
    # +1 because of void class
    assert dataset.semantic_n_classes == 151
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes

    # Check config based on semantic_n_classes
    assert dataset.config.semantic_label_list == ADE20K.SEMANTIC_LABEL_LIST_CHALLENGE_150

    # Check sample keys
    available_keys = ADE20K.get_available_sample_keys(split)
    for key in available_keys:
        assert key in dataset.sample_keys

    # Test loading first few samples
    for i in range(min(10, len(dataset))):
        sample = dataset[i]
        assert isinstance(sample, dict)
        assert isinstance(sample['identifier'], SampleIdentifier)

        if 'rgb' in sample:
            assert sample['rgb'].ndim == 3
            assert sample['rgb'].shape[2] == 3  # RGB channels

        if 'semantic' in sample:
            assert sample['semantic'].ndim == 2
            assert sample['semantic'].dtype == np.uint8

        if 'instance' in sample:
            assert sample['instance'].ndim == 2
            assert sample['instance'].dtype == np.uint16

        if 'scene' in sample:
            assert isinstance(sample['scene'], int)
            # +1 because of void class
            assert 0 <= sample['scene'] < len(dataset.scene_class_names) + 1


@pytest.mark.parametrize('split', SPLITS)
def test_scene_class_loading(split):
    if 'scene' not in ADE20K.get_available_sample_keys(split):
        pytest.skip(f"Split {split} does not contain scene labels")

    dataset = ADE20K(
        dataset_path=DATASET_PATH_DICT['ade20k'],
        split=split,
        sample_keys=('scene',)
    )

    for i in range(min(10, len(dataset))):
        scene_class = dataset[i]['scene']
        assert isinstance(scene_class, int)
        assert 0 <= scene_class < len(dataset.scene_class_names)


@pytest.mark.parametrize('split', SPLITS)
def test_camera_filtering(split):
    # Test with first camera (assuming multiple exist)
    test_cameras = (CAMERAS[0],)
    dataset = ADE20K(
        dataset_path=DATASET_PATH_DICT['ade20k'],
        split=split,
        cameras=test_cameras,
        sample_keys=ADE20K.get_available_sample_keys(split)
    )

    # Check all samples are from specified cameras
    for i in range(min(10, len(dataset))):
        identifier = dataset[i]['identifier']
        assert identifier[0] in test_cameras

    # Test filtering after initialization
    if len(dataset.cameras) > 1:
        dataset.filter_camera(CAMERAS[1])
        assert len(dataset) <= len(dataset._filenames_per_camera[CAMERAS[1]])
        for i in range(min(10, len(dataset))):
            assert dataset[i]['identifier'][0] == CAMERAS[1]


def test_debug_mode():
    # Test dataset without dataset_path
    dataset = ADE20K(dataset_path=None)
    assert len(dataset) == 0
    assert dataset.cameras == ADE20K.CAMERAS  # Single dummy camera
