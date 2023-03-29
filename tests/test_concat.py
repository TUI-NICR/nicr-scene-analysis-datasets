# -*- coding: utf-8 -*-
"""
Some common dataset tests

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from copy import deepcopy

import cv2

import pytest

from nicr_scene_analysis_datasets import ConcatDataset
from nicr_scene_analysis_datasets import get_dataset_class
from nicr_scene_analysis_datasets.pytorch import ConcatDataset as ConcatDatasetPyTorch
from nicr_scene_analysis_datasets.pytorch import get_dataset_class as get_dataset_class_pytorch
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


class SimpleDepthPreprocessor:
    def __call__(self, sample):
        sample['depth'] = cv2.resize(sample['depth'], (10, 10))
        return sample


@pytest.mark.parametrize('dataset_factory_and_class',
                         ((get_dataset_class, ConcatDataset),
                          (get_dataset_class_pytorch, ConcatDatasetPyTorch)))
def test_concatenated_dataset(dataset_factory_and_class):
    """Dataset concatenation"""
    dataset_factory, ConcatDatasetClass = dataset_factory_and_class

    sample_keys = ('identifier', 'depth')

    main_dataset = dataset_factory('sunrgbd')(
        dataset_path=DATASET_PATH_DICT['sunrgbd'],
        sample_keys=sample_keys,
        depth_force_mm=True,
        split='train',
        cameras=('kv1', 'kv2', 'xtion')   # remove realsense samples
    )

    dataset2 = dataset_factory('scannet')(
        dataset_path=DATASET_PATH_DICT['scannet'],
        sample_keys=sample_keys,
        split='train'
    )

    dataset3 = dataset_factory('nyuv2')(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        sample_keys=sample_keys,
        split='train'
    )

    dataset = ConcatDatasetClass(main_dataset, dataset2, dataset3)

    if issubclass(ConcatDatasetClass, ConcatDatasetPyTorch):
        # it is a pytorch dataset class, set a simple preprocessor
        preprocessor = SimpleDepthPreprocessor()
        dataset.preprocessor = preprocessor

        assert dataset.preprocessor == preprocessor
        assert main_dataset.preprocessor == preprocessor
        assert dataset2.preprocessor == preprocessor
        assert dataset3.preprocessor == preprocessor

    # simple tests
    n_samples_total = len(main_dataset) + len(dataset2) + len(dataset3)
    assert n_samples_total == len(main_dataset) + len(dataset2) + len(dataset3)

    # check that main_dataset is present
    offset = 0
    assert dataset[offset]['identifier'] == main_dataset[0]['identifier']
    assert (dataset[offset]['depth'] == main_dataset[0]['depth']).all()
    if issubclass(ConcatDatasetClass, ConcatDatasetPyTorch):
        # it is a pytorch dataset class, check that preprocessor was applied
        assert dataset[offset]['depth'].shape == (10, 10)

    # check that dataset2 is present
    offset += len(main_dataset)
    assert dataset[offset]['identifier'] == dataset2[0]['identifier']
    assert (dataset[offset]['depth'] == dataset2[0]['depth']).all()
    if issubclass(ConcatDatasetClass, ConcatDatasetPyTorch):
        # it is a pytorch dataset class, check that preprocessor was applied
        assert dataset[offset]['depth'].shape == (10, 10)

    # check that dataset3 is present
    offset += len(dataset2)
    assert dataset[offset]['identifier'] == dataset3[0]['identifier']
    assert (dataset[offset]['depth'] == dataset3[0]['depth']).all()
    if issubclass(ConcatDatasetClass, ConcatDatasetPyTorch):
        # it is a pytorch dataset class, check that preprocessor was applied
        assert dataset[offset]['depth'].shape == (10, 10)

    # check that negative indices work
    assert dataset[-n_samples_total]['identifier'] == dataset[0]['identifier']
    assert dataset[-1]['identifier'] == dataset3[-1]['identifier']

    # test with camera filter from outside
    n_samples_sunrgbd = len(main_dataset)
    with main_dataset.filter_camera('kv1') as ds:
        n_samples_sunrgbd_kv1 = len(ds)
        # concatenated dataset should change as well
        assert len(dataset) == n_samples_total - n_samples_sunrgbd + n_samples_sunrgbd_kv1
    # everything should be back to normal
    assert len(dataset) == n_samples_total
    assert len(main_dataset) == n_samples_sunrgbd

    # test with camera filter
    with dataset.filter_camera('kv1') as ds:
        assert len(ds) == n_samples_sunrgbd_kv1 + len(dataset3)
        assert ds.camera == 'kv1'
    # everything should be back to normal
    assert len(dataset) == n_samples_total
    assert dataset.camera is None

    # test with camera filter
    with dataset.filter_camera('structureio_480x640') as ds:
        assert len(ds) == len(dataset2)
        assert ds.camera == 'structureio_480x640'
    # everything should be back to normal
    assert len(dataset) == n_samples_total
    assert dataset.camera is None

    # test with camera filter without context
    dataset.filter_camera('kv1')
    assert len(ds) == n_samples_sunrgbd_kv1 + len(dataset3)
    dataset.filter_camera(None)
    # everything should be back to normal
    assert len(dataset) == n_samples_total

    # test copying
    dataset_copy = deepcopy(dataset)
    assert id(dataset_copy._main_dataset) != id(dataset._main_dataset)
    assert id(dataset_copy._additional_datasets[0]) != id(dataset._additional_datasets[0])
    dataset.filter_camera('kv1')
    assert len(dataset_copy) == n_samples_total
    assert dataset_copy.camera is None
