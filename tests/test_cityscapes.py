# -*- coding: utf-8 -*-
"""
Simple (interface) tests for Cityscapes dataset

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest

from nicr_scene_analysis_datasets import Cityscapes
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

N_SAMPLES = {'train': 2975, 'valid': 500, 'test': 1525}


@pytest.mark.parametrize('split', ('train', 'valid', 'test'))
@pytest.mark.parametrize('semantic_n_classes', (19, 33))
@pytest.mark.parametrize('disparity_instead_of_depth', (False, True))
def test_dataset(split,
                 semantic_n_classes,
                 disparity_instead_of_depth):
    sample_keys = (
        'identifier',
        'rgb', 'depth',
        'semantic'
    )
    dataset = Cityscapes(
        dataset_path=DATASET_PATH_DICT['cityscapes'],
        split=split,
        sample_keys=sample_keys,
        depth_mode='raw',
        disparity_instead_of_depth=disparity_instead_of_depth,
        semantic_n_classes=semantic_n_classes
    )

    assert dataset.depth_mode == 'raw'
    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[split]

    assert dataset.semantic_n_classes == semantic_n_classes + 1
    assert dataset.semantic_n_classes_without_void == semantic_n_classes
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void

    assert len(dataset.cameras) == 1

    assert isinstance(dataset.depth_min, float)
    assert isinstance(dataset.depth_max, float)
    assert isinstance(dataset.depth_mean, float)
    assert isinstance(dataset.depth_std, float)
    assert isinstance(dataset.depth_stats, dict)

    # test first 10 samples
    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert isinstance(sample['identifier'], SampleIdentifier)
        # inputs: rgb and depth
        assert sample['rgb'].ndim == 3
        assert sample['depth'].ndim == 2
        # semantic
        assert sample['semantic'].ndim == 2

        if i >= 9:
            break
