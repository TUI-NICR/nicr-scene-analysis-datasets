# -*- coding: utf-8 -*-
"""
Simple (interface) tests for ScanNet dataset

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import pytest

from nicr_scene_analysis_datasets import ScanNet
from nicr_scene_analysis_datasets.dataset_base import ExtrinsicCameraParametersNormalized
from nicr_scene_analysis_datasets.dataset_base import IntrinsicCameraParametersNormalized
from nicr_scene_analysis_datasets.dataset_base import SampleIdentifier
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


N_SAMPLES = {    # subsample is applied to each trajectory in a scene (folder)
    None: {'train': 1893422, 'valid': 530449, 'test': 208862},  # not used so far
    1: {'train': 1893422, 'valid': 530449, 'test': 208862},  # not used so far
    5: {'train': 379221, 'valid': 106217, 'test': 41827},    # used for mapping
    10: {'train': 189916, 'valid': 53193, 'test': 20942},    # used for mapping
    50: {'train': 38474, 'valid': 10767, 'test': 4223},  # default subsample !
    100: {'train': 19559, 'valid': 5465, 'test': 2135},
    200: {'train': 10098, 'valid': 2814, 'test': 1089},
    500: {'train': 4403, 'valid': 1222, 'test': 468}
}

N_SCENE_CLASSES = 21


@pytest.mark.parametrize('split', ('train', 'valid', 'test'))
@pytest.mark.parametrize('subsample', (50, 100, 200, 500))
@pytest.mark.parametrize('semantic_n_classes', (20, 40, 200, 549))
@pytest.mark.parametrize('instance_semantic_mode', ('raw', 'refined'))
def test_dataset(split, subsample, semantic_n_classes, instance_semantic_mode):
    dataset = ScanNet(
        dataset_path=DATASET_PATH_DICT['scannet'],
        split=split,
        subsample=subsample,
        depth_mode='raw',
        sample_keys=ScanNet.get_available_sample_keys(split),
        semantic_n_classes=semantic_n_classes,
        instance_semantic_mode=instance_semantic_mode
    )

    assert dataset.split == split

    assert len(dataset) == N_SAMPLES[subsample][split]

    assert dataset.semantic_n_classes == semantic_n_classes + 1
    assert dataset.semantic_n_classes_without_void == semantic_n_classes
    assert len(dataset.semantic_class_names) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_names_without_void) == dataset.semantic_n_classes_without_void
    assert len(dataset.semantic_class_colors) == dataset.semantic_n_classes
    assert len(dataset.semantic_class_colors_without_void) == dataset.semantic_n_classes_without_void

    assert len(dataset.scene_class_names) == N_SCENE_CLASSES
    assert len(dataset.cameras) == 2

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

        if 'test' != split:
            # semantic
            assert sample['semantic'].ndim == 2
            # instance
            assert sample['instance'].ndim == 2
            # scene
            assert isinstance(sample['scene'], int)

        if i >= 9:
            break


@pytest.mark.parametrize('split', ('train', 'valid'))
def test_scene_class_mapping(split):
    sample_keys = ('scene',)

    # create datasets (with default subsample!)
    dataset_original = ScanNet(
        dataset_path=DATASET_PATH_DICT['scannet'],
        split=split,
        sample_keys=sample_keys,
        scene_use_indoor_domestic_labels=False
    )

    dataset_remapped = ScanNet(
        dataset_path=DATASET_PATH_DICT['scannet'],
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
    assert sum(counts_remapped.values()) == N_SAMPLES[50][split]
    assert sum(counts_remapped.values()) == sum(counts_original.values())
    assert len(counts_remapped) == dataset_remapped.scene_n_classes

    assert dataset_original.scene_n_classes == dataset_original.scene_n_classes_without_void
    assert dataset_remapped.scene_n_classes == dataset_remapped.scene_n_classes_without_void + 1


@pytest.mark.parametrize('split', ('train', 'valid', 'test'))
def test_filter_camera(split):
    # just some random cameras and counts that we know
    sample_cameras = {   # for default subsample of 50
        'train': {
            'structureio_480x640': 688
        },
        'valid': {
            'structureio_968x1296': 10492,
        },
        'test': {'structureio_968x1296': 4223},
    }

    cameras = tuple(sample_cameras[split].keys())
    n_samples = tuple(sample_cameras[split].values())

    # create dataset with specified cameras
    dataset = ScanNet(
        dataset_path=DATASET_PATH_DICT['scannet'],
        split=split,
        sample_keys=ScanNet.get_available_sample_keys(split),
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
