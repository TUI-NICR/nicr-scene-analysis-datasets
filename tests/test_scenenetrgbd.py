# -*- coding: utf-8 -*-
"""
Simple (interface) tests for SceneNetRGBD dataset

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest

from nicr_scene_analysis_datasets import SceneNetRGBD
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT

N_CLASSES_WITH_VOID = 13 + 1
N_SAMPLES = {'train': 50595, 'valid': 6000}


@pytest.mark.parametrize('split', ('train', 'valid'))
def test_dataset(split):
    sample_keys = (
        'identifier',
        'rgb', 'depth',
        'semantic'
    )
    dataset = SceneNetRGBD(
        dataset_path=DATASET_PATH_DICT['scenenetrgbd'],
        split=split,
        sample_keys=sample_keys,
        depth_mode='refined',
        semantic_n_classes=N_CLASSES_WITH_VOID - 1
    )

    assert dataset.depth_mode == 'refined'
    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[split]

    assert dataset.semantic_n_classes == N_CLASSES_WITH_VOID
    assert dataset.semantic_n_classes_without_void == N_CLASSES_WITH_VOID - 1
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_colors_without_void) == dataset.semantic_n_classes_without_void

    assert len(dataset.cameras) == 1

    assert isinstance(dataset.depth_min, float)
    assert isinstance(dataset.depth_max, float)
    assert isinstance(dataset.depth_mean, float)
    assert isinstance(dataset.depth_std, float)
    assert isinstance(dataset.depth_stats, dict)

    # test first 10 samples sample
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
