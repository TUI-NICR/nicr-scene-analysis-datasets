# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Any, Callable, Dict, List, Tuple, Union

import abc
from copy import deepcopy
from functools import lru_cache
import warnings

import numpy as np

from ._annotation import ExtrinsicCameraParametersNormalized
from ._annotation import OrientationDict
from ._annotation import SampleIdentifier
from ._config import DatasetConfig
from ._class_weighting import compute_class_weights


class DatasetBase(abc.ABC):
    def __init__(
        self,
        sample_keys: Tuple[str] = ('semantic',),
        use_cache: bool = False,
        cache_disable_deepcopy: bool = False,  # kwargs only in derived classes
        disable_prints: bool = False  # kwargs only in derived classes
    ) -> None:
        super().__init__()
        assert isinstance(sample_keys, (tuple, list)), f"Got: '{sample_keys}'"

        # force lowercase for sample keys
        sample_keys = tuple(sk.lower() for sk in sample_keys)

        self._camera = None
        self._sample_keys = sample_keys
        self._sample_key_loaders = {}
        self._use_cache = use_cache
        self._cache_disable_deepcopy = cache_disable_deepcopy
        self._disable_prints = disable_prints

        self._instance_max_instances_per_image = 1 << 16

        if self._cache_disable_deepcopy:
            warnings.warn("Copying cache entries before returning is disabled."
                          "Be aware of this when modifying the returned "
                          "sample dicts.")

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
                    f"function called {func_name}.")
            self.register_sample_key_loader(
                sample_key,
                getattr(self, func_name)
            )

    @property
    def use_cache(self) -> bool:
        return self._use_cache

    @property
    def sample_keys(self) -> Tuple[str]:
        return self._sample_keys

    def filter_camera(self, camera):
        assert camera in self.cameras

        if self._use_cache and len(self.cameras) > 1:
            # clear cache for all loaders as index mapping changes
            for lru_wrapper in self._sample_key_loaders.values():
                lru_wrapper.cache_clear()

        self._camera = camera
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc: Any):
        self._camera = None

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    def __getitem__(self, idx):
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
    def camera(self) -> str:
        return self._camera

    @property
    @abc.abstractmethod
    def config(self) -> DatasetConfig:
        pass

    @abc.abstractmethod
    def _load_identifier(self, idx: int) -> SampleIdentifier:
        pass

    def _load_extrinsics(self, idx: int) -> ExtrinsicCameraParametersNormalized:
        # so far single extrinsic parameters for the entire sample
        raise NotImplementedError()

    def _load_semantic(self, idx: int) -> np.array:
        raise NotImplementedError()

    def _load_instance(self, idx: int) -> np.array:
        raise NotImplementedError()

    def _load_orientations(self, idx: int) -> OrientationDict:
        raise NotImplementedError()

    def _load_3d_boxes(self, idx: int) -> Dict[str, Any]:
        raise NotImplementedError()

    def _load_scene(self, idx: int) -> int:
        raise NotImplementedError()

    def _load_normal(self, idx: int) -> np.array:
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
    ) -> np.array:
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
    def semantic_class_colors(self) -> np.array:
        return self.config.semantic_label_list.colors_array

    @property
    def semantic_class_colors_without_void(self) -> np.array:
        return self.config.semantic_label_list_without_void.colors_array

    @property
    def semantic_n_classes(self) -> int:
        return len(self.config.semantic_label_list)

    @property
    def semantic_n_classes_without_void(self) -> int:
        return len(self.config.semantic_label_list_without_void)

    def semantic_get_colored(
        self,
        value: np.array,
        with_void: bool = True
    ) -> np.array:
        if with_void:
            colors = self.semantic_class_colors
        else:
            colors = self.semantic_class_colors_without_void
        cmap = np.asarray(colors, dtype='uint8')

        return cmap[value]

    @staticmethod
    def static_semantic_get_colored(
        value: np.array,
        colors: Union[Tuple, List, np.array]
    ) -> np.array:
        cmap = np.asarray(colors, dtype='uint8')
        return cmap[value]

    def semantic_compute_class_weights(
        self,
        weight_mode: str = 'median-frequency',
        c: float = 1.02,
        n_threads: int = 1,
        debug: bool = False
    ) -> np.array:
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
