# -*- coding: utf-8 -*-
"""
Some common dataset tests for the d2 interface

.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import pytest

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

# The import registers the datasets to d2
from nicr_scene_analysis_datasets import d2 as nicr_d2
from nicr_scene_analysis_datasets import KNOWN_DATASETS
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
@pytest.mark.parametrize('dataset_split', ('test', 'valid', 'train'))
def test_d2_dataset(dataset_name, dataset_split):
    invalid_names = set({
        'coco_test',
        'nyuv2_valid',
        'scenenetrgbd_test',
        'sunrgbd_valid'
    })
    # Get the path of the dataset
    dataset_path = DATASET_PATH_DICT[dataset_name]
    # Set the path for the dataset, so that d2 can use it
    nicr_d2.set_dataset_path(dataset_path)
    # Get the correct name for using the dataset from the DatasetCatalog
    dataset_name_d2 = f'{dataset_name}_{dataset_split}'
    if dataset_name_d2 in invalid_names:
        return
    dataset = DatasetCatalog.get(dataset_name_d2)
    dataset_config = MetadataCatalog.get(dataset_name_d2).dataset_config

    for i, sample in enumerate(dataset):
        assert isinstance(sample, dict)
        assert 'identifier' in sample
        assert 'rgb' in sample
        assert 'semantic' in sample or 'semantic' not in dataset.get_available_sample_keys(dataset_split)
        assert 'instance' in sample or 'instance' not in dataset.get_available_sample_keys(dataset_split)

        if i >= 9:
            break


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
def test_d2_helper_functions(dataset_name):

    class DummyMapper:
        def __call__(self, data):
            data['test'] = True
            return data

    valid_datasets_for_test = set({
        'nyuv2',
        'hypersim',
        'sunrgbd'
    })
    if dataset_name not in valid_datasets_for_test:
        return

    # Get the path of the dataset
    dataset_path = DATASET_PATH_DICT[dataset_name]
    # Set the path for the dataset, so that d2 can use it
    nicr_d2.set_dataset_path(dataset_path)
    # Get the correct name for using the dataset from the DatasetCatalog
    dataset_name_d2 = f'{dataset_name}_test'
    dataset = DatasetCatalog.get(dataset_name_d2)
    dataset_config = MetadataCatalog.get(dataset_name_d2).dataset_config

    data_mapper = nicr_d2.NICRSceneAnalysisDatasetMapper(dataset_config)
    dummy_mapper = DummyMapper()
    chained_mapper = nicr_d2.NICRChainedDatasetMapper(
        [data_mapper, dummy_mapper]
    )

    for i, data in enumerate(dataset):
        mapped_data = chained_mapper(data)
        assert 'test' in mapped_data
        assert mapped_data['test']
        if i >= 9:
            break
