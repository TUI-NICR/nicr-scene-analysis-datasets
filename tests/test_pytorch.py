# -*- coding: utf-8 -*-
"""
Some common dataset tests

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import time

import pytest

from torch.utils.data import DataLoader

from nicr_scene_analysis_datasets import KNOWN_DATASETS
from nicr_scene_analysis_datasets.pytorch import get_dataset_class
from nicr_scene_analysis_datasets.pytorch import NYUv2
from nicr_scene_analysis_datasets.utils.testing import DATASET_PATH_DICT


@pytest.mark.parametrize('dataset_name', KNOWN_DATASETS)
def test_pytorch_dataset(dataset_name):
    """Test that there is PyTorch dataset for all known datasets"""
    Dataset = get_dataset_class(dataset_name)

    dataset = Dataset(dataset_path=DATASET_PATH_DICT[dataset_name])
    for sample in dataset:
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
