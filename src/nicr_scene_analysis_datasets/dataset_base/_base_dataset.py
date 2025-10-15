# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import abc
import dataclasses
import os
import warnings
from copy import deepcopy
from functools import lru_cache

import numpy as np

from ..utils.io import load_creation_metafile
from ._annotation import ExtrinsicCameraParametersNormalized
from ._annotation import MetaDict
from ._annotation import OrientationDict
from ._annotation import SampleIdentifier
from ._class_weighting import compute_class_weights
from ._config import DatasetConfig


class DatasetBase(abc.ABC):
    def __init__(
        self,
        *,
        dataset_path: Optional[str] = None,
        sample_keys: Tuple[str] = ('semantic',),
        use_cache: bool = False,
        cache_disable_deepcopy: bool = False,  # kwargs only in derived classes
        disable_prints: bool = False  # kwargs only in derived classes
    ) -> None:
        super().__init__()
        assert isinstance(sample_keys, (tuple, list)), f"Got: '{sample_keys}'"

        # force lowercase for sample keys
        sample_keys = tuple(sk.lower() for sk in sample_keys)

        self._dataset_path = dataset_path
        if self._dataset_path is not None:
            self._dataset_path = os.path.expanduser(self._dataset_path)

            # check dataset path - catch a common error
            assert os.path.exists(self._dataset_path), (
                f"Dataset path does not exist: '{self._dataset_path}'"
            )
        self._camera = None
        self._sample_keys = sample_keys
        self._sample_key_loaders = {}
        # will be determined in the first call to load_meta()
        self._sample_meta = None
        self._use_cache = use_cache
        self._cache_disable_deepcopy = cache_disable_deepcopy
        self._disable_prints = disable_prints

        self._instance_max_instances_per_image = 1 << 16

        if self._cache_disable_deepcopy:
            warnings.warn("Copying cache entries before returning is disabled."
                          "Be aware of this when modifying the returned "
                          "sample dicts.")

        # load creation meta
        if self._dataset_path is not None:
            self._creation_meta = load_creation_metafile(self._dataset_path)
            if self._creation_meta is None:
                warnings.warn(f"No creation meta file found at: "
                              f"'{dataset_path}'.")
        else:
            self._creation_meta = None

        # Note:
        # 'auto_register_sample_key_loaders' should NOT be called here as it
        # heavily slows down caching, instead, call it in the derived class at
        # the end of the constructor
        # self.auto_register_sample_key_loaders()

    def register_sample_key_loader(
        self,
        sample_key: str,
        func: Callable
    ) -> None:
        if self._use_cache:

            @lru_cache(None)
            def func_(*args, **kwargs):
                return func(*args, **kwargs)

        else:
            func_ = func

        self._sample_key_loaders[sample_key] = func_

    def auto_register_sample_key_loaders(self) -> None:
        for sample_key in self._sample_keys:
            func_name = f'_load_{sample_key}'
            if not hasattr(self, func_name):
                raise AttributeError(
                    f"Cannot auto register sample key loader for key: "
                    f"'{sample_key}' as {self.__class__.__name__} has no "
                    f"function called {func_name}."
                )
            self.register_sample_key_loader(
                sample_key,
                getattr(self, func_name)
            )

    def debug_print(self, *args, **kwargs):
        if not self._disable_prints:
            print(*args, **kwargs)

    @property
    def dataset_path(self) -> str:
        return self._dataset_path

    @property
    def creation_meta(self) -> Union[None, Dict]:
        # we may have multiple entries in the meta file, so we take the last
        # one, see create_or_update_creation_metafile()
        if self._creation_meta is not None:
            return self._creation_meta[-1]

        return None

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def sample_keys(self) -> Tuple[str]:
        return self._sample_keys

    def filter_camera(self, camera: Union[None, str]):
        assert camera is None or camera in self.cameras

        if self._use_cache and len(self.cameras) > 1:
            # clear cache for all loaders as index mapping changes
            for lru_wrapper in self._sample_key_loaders.values():
                lru_wrapper.cache_clear()

        self._camera = camera
        return self

    def __enter__(self):
        # handles context stuff, e.g., with dataset.filter_camera('xy') as ds
        return self

    def __exit__(self, *exc: Any):
        # handles context stuff, e.g., with dataset.filter_camera('xy') as ds
        # reset camera filter
        self.filter_camera(None)

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def _get_filename(self, idx: int) -> str:
        pass

    def __getitem__(self, idx: int):
        # load data for every key in _sample_keys (see init)
        sample = {key: self.load(key, idx) for key in self._sample_keys}

        if self._use_cache and not self._cache_disable_deepcopy:
            # copy sample dict be avoid side effects when changing entries
            # during preprocessing as this might modify the cache itself
            sample = deepcopy(sample)

        return sample

    def load(self, sample_key: str, idx: int) -> Any:
        return self._sample_key_loaders.get(sample_key.lower())(idx)

    @property
    @abc.abstractmethod
    def split(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def cameras(self) -> Tuple[str]:
        pass

    @property
    def camera(self) -> Union[None, str]:
        return self._camera

    @property
    @abc.abstractmethod
    def config(self) -> DatasetConfig:
        pass

    @staticmethod
    @abc.abstractmethod
    def get_available_sample_keys(split: str) -> Tuple[str]:
        pass

    def identifier2idx(
        self,
        identifier: SampleIdentifier,
        ensure_single_match: bool = True
    ) -> int:
        # this function might be useful for some use cases, however, as it
        # always iterates all identifiers it is not very efficient, do not
        # call it to much
        idx = -1
        for i in range(len(self)):
            if self._load_identifier(i) == tuple(identifier):
                if not ensure_single_match:
                    # first match is enough
                    return i
                else:
                    if -1 == idx:
                        # first match
                        idx = i
                    else:
                        # another match, raise error
                        raise ValueError(
                            f"Found at least two matches for identifier: "
                            f"'{identifier}, indices: {idx} and {i}'."
                        )

        if -1 != idx:
            # we got a single match, return it
            return idx

        raise ValueError(f"Could not find identifier: '{identifier}'")

    @abc.abstractmethod
    def _load_identifier(self, idx: int) -> SampleIdentifier:
        pass

    def _load_meta(self, idx: int) -> MetaDict:
        if self._sample_meta is None:
            # load meta data from config
            self._sample_meta = MetaDict(dataclasses.asdict(self.config))
        return self._sample_meta

    def _load_extrinsics(self, idx: int) -> ExtrinsicCameraParametersNormalized:
        # so far single extrinsic parameters for the entire sample
        raise NotImplementedError()

    def _load_semantic(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def _load_instance(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    def _load_orientations(self, idx: int) -> OrientationDict:
        raise NotImplementedError()

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def _load_scene(self, idx: int) -> int:
        raise NotImplementedError()

    def _load_normal(self, idx: int) -> np.ndarray:
        raise NotImplementedError()

    @property
    def scene_class_names(self) -> Tuple[str]:
        return self.config.scene_label_list.class_names

    @property
    def scene_n_classes(self) -> int:
        return len(self.config.scene_label_list)

    @property
    def scene_class_names_without_void(self) -> Tuple[str]:
        return self.config.scene_label_list_without_void.class_names

    @property
    def scene_n_classes_without_void(self) -> int:
        return len(self.config.scene_label_list_without_void)

    @property
    def instance_max_instances_per_image(self) -> int:
        return self._instance_max_instances_per_image

    def scene_compute_class_weights(
        self,
        weight_mode: str = 'linear',
        ignore_first_class=True,
        c: float = 1.02,
        n_threads: int = 1,
        debug: bool = False
    ) -> np.ndarray:
        return compute_class_weights(
            dataset=self,
            sample_key='scene',
            n_classes=self.scene_n_classes,
            ignore_first_class=ignore_first_class,
            weight_mode=weight_mode,
            c=c,
            n_threads=n_threads,
            debug=debug,
            verbose=not self._disable_prints
        )

    @property
    def semantic_class_names(self) -> Tuple[str]:
        return self.config.semantic_label_list.class_names

    @property
    def semantic_class_names_without_void(self) -> Tuple[str]:
        return self.config.semantic_label_list_without_void.class_names

    @property
    def semantic_class_colors(self) -> np.ndarray:
        return self.config.semantic_label_list.colors_array

    @property
    def semantic_class_colors_without_void(self) -> np.ndarray:
        return self.config.semantic_label_list_without_void.colors_array

    @property
    def semantic_n_classes(self) -> int:
        return len(self.config.semantic_label_list)

    @property
    def semantic_n_classes_without_void(self) -> int:
        return len(self.config.semantic_label_list_without_void)

    def semantic_get_colored(
        self,
        value: np.ndarray,
        with_void: bool = True
    ) -> np.ndarray:
        if with_void:
            colors = self.semantic_class_colors
        else:
            colors = self.semantic_class_colors_without_void
        cmap = np.asarray(colors, dtype='uint8')

        return cmap[value]

    @staticmethod
    def static_semantic_get_colored(
        value: np.ndarray,
        colors: Union[Tuple, List, np.ndarray]
    ) -> np.ndarray:
        cmap = np.asarray(colors, dtype='uint8')
        return cmap[value]

    def semantic_compute_class_weights(
        self,
        weight_mode: str = 'median-frequency',
        c: float = 1.02,
        n_threads: int = 1,
        debug: bool = False
    ) -> np.ndarray:
        return compute_class_weights(
            dataset=self,
            sample_key='semantic',
            n_classes=self.semantic_n_classes,
            ignore_first_class=True,
            weight_mode=weight_mode,
            c=c,
            n_threads=n_threads,
            debug=debug,
            verbose=not self._disable_prints
        )
