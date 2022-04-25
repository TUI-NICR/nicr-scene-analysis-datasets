# -*- coding: utf-8 -*-
"""
Simple (interface) tests for Hypersim dataset

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from nicr_scene_analysis_datasets import Hypersim
from nicr_scene_analysis_datasets.dataset_base import ExtrinsicCameraParametersNormalized
from nicr_scene_analysis_datasets.dataset_base import IntrinsicCameraParametersNormalized
from nicr_scene_analysis_datasets.dataset_base import OrientationDict
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


N_CLASSES_WITH_VOID = 40 + 1
N_SAMPLES = {
    None: {'train': 57443, 'valid': 7286, 'test': 7690},
    1: {'train': 57443, 'valid': 7286, 'test': 7690},
    2: {'train': 28722, 'valid': 3643, 'test': 3845},
    5: {'train': 11489, 'valid': 1458, 'test': 1538},
    10: {'train': 5745, 'valid': 729, 'test': 769},
    20: {'train': 2873, 'valid': 365, 'test': 385}
}
N_SCENE_CLASSES = 22


@pytest.mark.parametrize('split', ('train', 'valid', 'test'))
@pytest.mark.parametrize('depth_mode', ('raw', ))
@pytest.mark.parametrize('subsample', (None, 1, 2, 5, 10, 20))
def test_dataset(split, depth_mode, subsample):
    sample_keys = (
        'identifier',
        'rgb', 'rgb_intrinsics', 'depth', 'depth_intrinsics',
        'extrinsics',
        'semantic', 'instance', 'orientations', '3d_boxes', 'scene', 'normal'
    )
    dataset = Hypersim(
        dataset_path=DATASET_PATH_DICT['hypersim'],
        split=split,
        subsample=subsample,
        sample_keys=sample_keys,
        depth_mode=depth_mode
    )

    assert dataset.depth_mode == depth_mode
    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[subsample][split]

    assert dataset.semantic_n_classes == N_CLASSES_WITH_VOID
    assert dataset.semantic_n_classes_without_void == N_CLASSES_WITH_VOID - 1
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
        assert isinstance(sample['extrinsics'],
                          ExtrinsicCameraParametersNormalized)
        assert (3+4) == len(sample['extrinsics'])
        # inputs: rgb and depth
        assert sample['rgb'].ndim == 3
        assert isinstance(sample['rgb_intrinsics'],
                          IntrinsicCameraParametersNormalized)
        assert (2+2+6+2) == len(sample['rgb_intrinsics'])
        assert sample['depth'].ndim == 2
        assert isinstance(sample['depth_intrinsics'],
                          IntrinsicCameraParametersNormalized)
        assert (2+2+6+2+2) == len(sample['depth_intrinsics'])
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
            # Check if orientation with key exists in instance
            assert (sample['instance'] == key).sum() > 0
            assert isinstance(key, int)
            assert isinstance(value, float)
            # assert that the encoding is in radians
            assert 0 <= value <= 2*np.pi
        # 3d boxes
        assert isinstance(sample['3d_boxes'], dict)

        # verify that every instance has an orientation
        for instance_id in np.unique(sample['instance']):
            # void
            if instance_id == 0:
                continue
            assert(instance_id in sample['orientations'])

        if i >= 9:
            break


@pytest.mark.parametrize('split', ('train', 'valid', 'test'))
def test_scene_class_mapping(split):
    sample_keys = ('scene',)

    # create datasets
    dataset_original = Hypersim(
        dataset_path=DATASET_PATH_DICT['hypersim'],
        split=split,
        sample_keys=sample_keys,
        scene_use_indoor_domestic_labels=False
    )

    dataset_remapped = Hypersim(
        dataset_path=DATASET_PATH_DICT['hypersim'],
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
    assert sum(counts_remapped.values()) == N_SAMPLES[None][split]
    assert sum(counts_remapped.values()) == sum(counts_original.values())
    assert len(counts_remapped) == dataset_remapped.scene_n_classes

    assert dataset_original.scene_n_classes == dataset_original.scene_n_classes_without_void
    assert dataset_remapped.scene_n_classes == dataset_remapped.scene_n_classes_without_void + 1
