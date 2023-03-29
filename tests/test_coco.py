# -*- coding: utf-8 -*-
"""
Simple (interface) tests for COCO dataset

.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest

from nicr_scene_analysis_datasets import COCO
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

N_SAMPLES = {'train': 118287, 'valid': 5000}
N_CLASSES_WITH_VOID = 133 + 1
N_CAMERAS = {'train': 2477, 'valid': 603}


@pytest.mark.parametrize('split', ('train', 'valid'))
def test_dataset(split):
    dataset = COCO(
        dataset_path=DATASET_PATH_DICT['coco'],
        split=split,
        sample_keys=COCO.get_available_sample_keys(split)
    )
    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[split]

    assert dataset.semantic_n_classes == N_CLASSES_WITH_VOID
    assert dataset.semantic_n_classes_without_void == N_CLASSES_WITH_VOID - 1
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_colors_without_void) == dataset.semantic_n_classes_without_void

    # test first 10 samples sample
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert isinstance(sample['identifier'], SampleIdentifier)
        # inputs: rgb
        assert sample['rgb'].ndim == 3
        # semantic
        assert sample['semantic'].ndim == 2
        # instance
        assert sample['instance'].ndim == 2

        if i >= 9:
            break

    # test camera filtering
    assert len(dataset.cameras) == N_CAMERAS[split], len(dataset.cameras)
    for camera in dataset.cameras[::10]:    # test only every 10th camera
        with dataset.filter_camera(camera):
            # get shape of first sample
            h, w, _ = dataset[0]['rgb'].shape

            assert f'{w}x{h}' == camera


@pytest.mark.parametrize('split', ('train', 'valid'))
def test_filter_camera(split):
    # just some random cameras and counts that we know
    sample_cameras = {
        'train': {'480x640': 8411, '426x640': 1660},
        'valid': {'640x480': 1061, '480x640': 336, '500x335': 9}
    }

    cameras = tuple(sample_cameras[split].keys())
    n_samples = tuple(sample_cameras[split].values())

    # create dataset with specified cameras
    dataset = COCO(
        dataset_path=DATASET_PATH_DICT['coco'],
        split=split,
        sample_keys=COCO.get_available_sample_keys(split),
        cameras=cameras
    )

    assert dataset.cameras == cameras
    assert len(dataset) == sum(n_samples)

    # test filtering
    dataset.filter_camera(cameras[0])
    assert dataset.camera == cameras[0]
    assert len(dataset) == n_samples[0]

    # reset filtering
    dataset.filter_camera(None)
    assert dataset.camera is None
    assert len(dataset) == sum(n_samples)
