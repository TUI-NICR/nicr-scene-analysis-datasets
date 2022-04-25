# -*- coding: utf-8 -*-
"""
Simple (interface) tests for NYUv2 dataset

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from nicr_scene_analysis_datasets import NYUv2
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


N_SAMPLES = {'train': 795, 'test': 654}
N_SCENE_CLASSES = 27


@pytest.mark.parametrize('split', ('train', 'test'))
@pytest.mark.parametrize('semantic_n_classes', (894, 40, 13))
@pytest.mark.parametrize('depth_mode', ('refined', 'raw'))
def test_dataset(split, semantic_n_classes, depth_mode):
    sample_keys = (
        'identifier',
        'rgb', 'depth',
        'semantic', 'instance', 'orientations', 'scene', 'normal'
    )
    dataset = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        split=split,
        depth_mode=depth_mode,
        sample_keys=sample_keys,
        semantic_n_classes=semantic_n_classes
    )

    assert dataset.depth_mode == depth_mode
    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[split]

    assert dataset.semantic_n_classes == semantic_n_classes + 1
    assert dataset.semantic_n_classes_without_void == semantic_n_classes
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_colors_without_void) == dataset.semantic_n_classes_without_void

    assert len(dataset.scene_class_names) == N_SCENE_CLASSES
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
        # instance
        assert sample['instance'].ndim == 2
        # normal
        normal = sample['normal']
        assert normal.ndim == 3
        assert normal.dtype == 'float32'
        norms = np.linalg.norm(normal, ord=2, axis=-1)
        mask = norms > 1e-7    # filter invalid pixels
        assert_almost_equal(norms[mask], 1, decimal=4)
        # scene
        assert isinstance(sample['scene'], int)
        # orientation
        assert isinstance(sample['orientations'], OrientationDict)
        for key, value in sample['orientations'].items():
            # check if orientation with key exists in instance
            assert (sample['instance'] == key).sum() > 0

            assert isinstance(key, int)
            assert isinstance(value, float)
            # assert that the encoding is in radians
            assert 0 <= value <= 2*np.pi

        if i >= 9:
            break


def test_dataset_computing():
    # as NYUv2 is quite small, we additionally test some functions
    dataset = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        sample_keys=('rgb', 'depth', 'semantic')
    )
    weights_1 = dataset.semantic_compute_class_weights(
        'median-frequency', n_threads=1
    )
    weights_10 = dataset.semantic_compute_class_weights(
        'median-frequency',
        n_threads=10
    )
    assert np.array_equal(weights_1, weights_10)
    assert np.array_equal(dataset.semantic_compute_class_weights(debug=True),
                          np.ones(dataset.semantic_n_classes_without_void))

    assert dataset.depth_compute_stats(n_threads=10)


@pytest.mark.parametrize('split', ('train', 'test'))
def test_scene_class_mapping(split):
    sample_keys = ('scene',)

    # create datasets
    dataset_original = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        split=split,
        sample_keys=sample_keys,
        scene_use_indoor_domestic_labels=False
    )

    dataset_remapped = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        split=split,
        sample_keys=sample_keys,
        scene_use_indoor_domestic_labels=True
    )

    # count samples
    def count(dataset):
        class_names = dataset.config.scene_label_list.class_names
        counts = {n: 0 for n in class_names}
        for sample in dataset:
            counts[class_names[sample['scene']]] += 1

        return counts

    counts_original = count(dataset_original)
    counts_remapped = count(dataset_remapped)

    # perform simple some checks
    assert sum(counts_remapped.values()) == N_SAMPLES[split]
    assert sum(counts_remapped.values()) == sum(counts_original.values())
    assert len(counts_remapped) == dataset_remapped.scene_n_classes

    assert dataset_original.scene_n_classes == dataset_original.scene_n_classes_without_void
    assert dataset_remapped.scene_n_classes == dataset_remapped.scene_n_classes_without_void + 1
