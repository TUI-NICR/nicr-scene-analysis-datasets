# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Tuple, Union

from collections import OrderedDict
import warnings

from ._base_dataset import DatasetBase


class ConcatDataset:
    def __init__(
        self,
        main_dataset: DatasetBase,
        *additional_datasets: DatasetBase
    ) -> None:
        self._main_dataset = main_dataset
        self._additional_datasets = additional_datasets
        self._datasets = (main_dataset,) + additional_datasets
        self._active_datasets = (main_dataset,) + additional_datasets

        # catch common misconfiguration
        for ds in self._datasets:
            if hasattr(ds, 'depth_force_mm') and not ds.depth_force_mm:
                # actually SUNRGB-D
                warnings.warn(
                    f"Detected '{ds.__class__.__name__}' dataset with "
                    "deviating depth scale, consider setting "
                    "`depth_force_mm` to 'True'."
                )

        # extract information from main dataset
        self._sample_keys = main_dataset.sample_keys
        # ensure that all additional datasets provide the sample keys of the
        # main dataset
        for ds in self._additional_datasets:
            assert all(sk in ds.sample_keys for sk in self._sample_keys)

        # handle cameras (create ordered union of all cameras)
        # note, we use dicts instead of sets to preserve the order (sets use a
        # random seed while hashing and, thus, do not guarantee insertion order)
        assert all(ds.camera is None for ds in self._datasets)
        cameras = []
        for ds in self._datasets:
            cameras.extend(ds.cameras)
        # note as of python 3.7, dicts guarantee insertion order, so we might
        # use dict in future
        self._cameras = tuple(OrderedDict.fromkeys(cameras).keys())

        self._camera = None

    def filter_camera(self, camera: Union[None, str]):
        assert camera is None or camera in self.cameras

        # apply filter to all datasets
        # note, not all datasets may support given camera, filter them using
        # active_datasets
        active_datasets = []
        for ds in self._datasets:
            if camera is None or camera in ds.cameras:
                ds.filter_camera(camera)
                active_datasets.append(ds)

        self._active_datasets = tuple(active_datasets)
        self._camera = camera

        return self

    def __enter__(self):
        # handles context stuff, e.g., with dataset.filter_camera('xy') as ds
        return self

    def __exit__(self, *exc: Any):
        # handles context stuff, e.g., with dataset.filter_camera('xy') as ds
        # reset camera filter
        self.filter_camera(None)

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._active_datasets)

    @property
    def datasets(self) -> Tuple[DatasetBase]:
        return self._active_datasets

    def _determine_dataset_and_idx(self, idx: int) -> Tuple[DatasetBase, int]:
        length = len(self)

        # ensure that idx is in valid range
        if not (-length <= idx < length):
            raise IndexError(f"Index {idx} out of range (length: {length}).")

        # handle negative indices
        if idx < 0:
            idx += length

        # note that the lengths may change if filter_dataset is called outside
        for ds in self._active_datasets:
            if idx < len(ds):
                return ds, idx
            idx -= len(ds)

    def load(self, sample_key: str, idx: int) -> Any:
        ds, ds_idx = self._determine_dataset_and_idx(idx)
        return ds._sample_key_loaders.get(sample_key.lower())(ds_idx)

    def __getitem__(self, idx: int):
        # note, we also reimplement __getitem__ to do index mapping stuff only
        # once per sample
        ds, ds_idx = self._determine_dataset_and_idx(idx)
        return ds[ds_idx]

    @property
    def cameras(self) -> Tuple[str]:
        return self._cameras

    @property
    def camera(self) -> Union[None, str]:
        return self._camera

    def __getstate__(self):
        # important for copying
        return self.__dict__

    def __setstate__(self, state):
        # important for copying
        self.__dict__ = state

    def __getattr__(self, name):
        if name not in self.__dict__ and '_main_dataset' in self.__dict__:
            # redirect all other attributes/calls to main dataset
            return getattr(self._main_dataset, name)

        return super().__getattr__(name)
