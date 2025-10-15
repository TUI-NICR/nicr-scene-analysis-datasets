# -*- coding: utf-8 -*-
"""
Some common dataset tests

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import time
import sys

import pytest

import numpy as np
from torch.utils.data import DataLoader

from nicr_scene_analysis_datasets import KNOWN_DATASETS
from nicr_scene_analysis_datasets.auxiliary_data import _AuxiliaryDataset
from nicr_scene_analysis_datasets.pytorch import get_dataset_class
from nicr_scene_analysis_datasets.pytorch import NYUv2
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
@pytest.mark.parametrize('with_auxiliary_data', [True, False])
# Skip if Python 3.8 or lower is used in combination with with_auxiliary_data.
# TODO: This is currently done, because the text embedding dicts were pickled
# with a version which seems to be incompatible with Python 3.8.
@pytest.mark.parametrize(
    'with_all_available_sample_keys',
    [True, False] if sys.version_info >= (3, 9) else [False]
)
def test_pytorch_dataset(dataset_name, with_auxiliary_data,
                         with_all_available_sample_keys):
    """Test that there is PyTorch dataset for all known datasets"""
    Dataset = get_dataset_class(
        dataset_name, with_auxiliary_data=with_auxiliary_data
    )
    kwargs = {}
    if with_all_available_sample_keys:
        # Dataset path is also passed to the function to get the sample keys
        # for axuiliary data the dataset was actually created with
        split = Dataset.SPLITS[0]
        kwargs['split'] = split
        additional_kwargs = {}

        if with_auxiliary_data:
            # Only the auxilary data wrapper can take this kwarg
            kwargs['dataset_path'] = DATASET_PATH_DICT[dataset_name]
            sample_keys = Dataset.get_available_sample_keys(**kwargs)

            # Set args so that auxiliary data is actually used
            if 'depth' in sample_keys:
                depth_estimator = \
                    Dataset.get_available_depth_estimators(**kwargs)
                if len(depth_estimator) > 0:
                    depth_estimator = depth_estimator[0]
                    additional_kwargs['depth_estimator'] = depth_estimator
            if 'image_embedding' in sample_keys:
                embedding_estimator = \
                    Dataset.get_available_image_embedding_estimators(**kwargs)
                if len(embedding_estimator) > 0:
                    embedding_estimator = embedding_estimator[0]
                    additional_kwargs['image_embedding_estimator'] = \
                        embedding_estimator
            if 'panoptic_embedding' in sample_keys:
                embedding_estimator = \
                    Dataset.get_available_panoptic_embedding_estimators(**kwargs)
                if len(embedding_estimator) > 0:
                    embedding_estimator = embedding_estimator[0]
                    additional_kwargs['panoptic_embedding_estimator'] = \
                        embedding_estimator

            semantic_classes = Dataset.SEMANTIC_N_CLASSES
            # Required because of behavior in generate_auxiliary_data script
            if isinstance(semantic_classes, list):
                additional_kwargs['semantic_n_classes'] = semantic_classes[0]
            elif isinstance(semantic_classes, tuple):
                if len(semantic_classes) > 1:
                    additional_kwargs['semantic_n_classes'] = \
                        semantic_classes[0]
        else:
            sample_keys = Dataset.get_available_sample_keys(**kwargs)

        kwargs.update(additional_kwargs)

        kwargs['sample_keys'] = sample_keys

    kwargs['dataset_path'] = DATASET_PATH_DICT[dataset_name]
    dataset = Dataset(**kwargs)
    if with_auxiliary_data:
        # Assert that the dataset was actually wrapped
        assert isinstance(dataset, _AuxiliaryDataset)
    for sample in dataset:
        if with_all_available_sample_keys and with_auxiliary_data:
            # Some general checks
            if 'depth' in kwargs['sample_keys']:
                assert 'depth' in sample
                assert isinstance(sample['depth'], np.ndarray)
                depth_img = sample['depth']
                assert depth_img.ndim == 2
            if 'image_embedding' in kwargs['sample_keys']:
                assert 'image_embedding' in sample
                image_embedding = sample['image_embedding']
                assert isinstance(image_embedding, np.ndarray)
                assert image_embedding.ndim == 1
            if 'panoptic_embedding' in kwargs['sample_keys']:
                assert 'panoptic_embedding' in sample
                panoptic_embedding = sample['panoptic_embedding']
                assert isinstance(panoptic_embedding, dict)
                for key, value in panoptic_embedding.items():
                    assert isinstance(key, int)
                    assert isinstance(value, np.ndarray)
                    assert value.ndim == 1
        break

    # ensure that preprocessor is working
    # stupid preprocessor that erases the entire sample
    dataset.preprocessor = lambda sample: {'dummy_key': 0}

    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=1,
                            persistent_workers=True)
    for i, sample in enumerate(dataloader):
        assert isinstance(sample, dict)
        assert 1 == len(sample)
        assert 'dummy_key' in sample

        if i >= 9:
            break


def test_caching_dataloader():
    # as NYUv2 is quite small, we additionally test some functions
    dataset = NYUv2(
        dataset_path=DATASET_PATH_DICT['nyuv2'],
        sample_keys=('rgb', 'semantic'),
        use_cache=True
    )

    n_worksers = 4

    dataloader = DataLoader(dataset,
                            batch_size=4,
                            shuffle=True,
                            num_workers=n_worksers,
                            persistent_workers=True)

    simple_sums = []
    durations = []
    for _ in range(4*n_worksers):
        start = time.time()
        sum_ = 0
        for sample in dataloader:
            sum_ += sample['rgb'].numpy().sum()
        end = time.time()

        simple_sums.append(sum_)
        durations.append(end-start)

        # print(simple_sums[-1], durations[-1])

    # note that all workers have to cache the dataset
    assert all(d < durations[0] for d in durations[-2*n_worksers:])
    assert all(s == simple_sums[0] for s in simple_sums[-2*n_worksers:])
