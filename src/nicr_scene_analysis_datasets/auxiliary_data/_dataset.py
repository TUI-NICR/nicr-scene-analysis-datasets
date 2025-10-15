# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple, Optional

import dataclasses
import os

import cv2
import numpy as np
from tqdm import tqdm

from ._config import build_dataset_config_with_auxiliary
from ..dataset_base import DatasetBase
from ..dataset_base import DepthDataset
from ..dataset_base import DepthStats
from ..dataset_base._annotation import PanopticEmbeddingDict
from ..utils.io import load_creation_metafile


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


class _AuxiliaryDataset(DatasetBase):
    def __init__(
        self,
        dataset_path: Optional[str] = None,
        depth_estimator: Optional[str] = None,
        image_embedding_estimator: Optional[str] = None,
        panoptic_embedding_estimator: Optional[str] = None,
        semantic_n_classes: Optional[int] = None,
        compute_mean_visual_embeddings: bool = False,
        normalize_embeddings_on_load: bool = True,
        **kwargs
    ) -> None:
        # Some datasets (e.g. coco or hypersim) only have one
        # class spectrum and doesn't take the kwarg.
        # As the wrapper dosn't know it beforehand, we need to check
        # if the kwarg is set.
        if not isinstance(self.SEMANTIC_N_CLASSES, int):
            if semantic_n_classes is not None:
                kwargs['semantic_n_classes'] = semantic_n_classes
        super().__init__(
            dataset_path=dataset_path,
            **kwargs
        )
        self._depth_estimator = depth_estimator
        self.AUXILIARY_DEPTH_DIR_FMT = 'depth_{:s}'
        self._available_depth_estimators = tuple()

        self._image_embedding_estimator = image_embedding_estimator
        self._available_image_embedding_estimators = tuple()

        self._panoptic_embedding_estimator = panoptic_embedding_estimator
        self._available_panoptic_embedding_estimators = tuple()
        self._semantic_n_classes_str = str(semantic_n_classes)

        # Already initialize it, so that we can use it in the config method.
        self.semantic_text_embedding_list = []
        self.scene_text_embedding_list = []
        self._mean_embedding_per_semantic_class = {}
        self._mean_image_embedding_per_semantic_class = {}
        self._normalize_embeddings_on_load = normalize_embeddings_on_load

        self.AUXILIARY_IMAGE_EMBEDDING_DIR_FMT = 'image_embedding_{:s}'
        self._available_image_embedding_estimators = tuple()

        self.AUXILIARY_PANOPTIC_EMBEDDING_DIR_FMT = 'panoptic_{:s}_embedding_{:s}'
        self._available_panoptic_embedding_estimators = tuple()

        self.AUXILIARY_SEMANTIC_CLASS_NAMES_FMT = 'semantic_{:s}_embedding_{:s}'

        self.TEXT_EMBEDDING_PROMPT = 'a detailed photo of a {:s}'

        self.AUXILIARY_SEGMENTATION_CLASS_NAMES_EMITTER_FMT = 'semantic_{:s}_embedding_{:s}'
        self._available_depth_estimators = \
            _AuxiliaryDataset.get_available_depth_estimators(
                split=self.split, dataset_path=dataset_path
            )
        self._available_image_embedding_estimators = \
            _AuxiliaryDataset.get_available_image_embedding_estimators(
                split=self.split, dataset_path=dataset_path
            )
        self._available_panoptic_embedding_estimators = \
            _AuxiliaryDataset.get_available_panoptic_embedding_estimators(
                split=self.split, dataset_path=dataset_path
            )

        # extend available depth estimators and stats from auxiliary meta
        auxiliary_depth_stats = _AuxiliaryDataset.get_key_from_creation_metafile(
            'auxiliary_depth_stats', dataset_path
        )
        available_depth_stats = {}
        if auxiliary_depth_stats is not None:
            for est in auxiliary_depth_stats:
                for est_split in auxiliary_depth_stats[est]:
                    if est not in available_depth_stats:
                        available_depth_stats[est] = {}
                    available_depth_stats[est][est_split] = DepthStats(
                        **auxiliary_depth_stats[est][est_split]
                    )

        self._auxiliary_depth_stats = available_depth_stats

        # The depth_mode is required at different places, and is member
        # of the DepthDataset class (getter only) and all inherited classes.
        # However if our underlying dataset is only a RGBDataset, we still
        # need to set the depth_mode to 'raw' as this is the default
        # value for the depth mode in the DepthDataset class.
        # As we can't set/overwrite the depth_mode in the parent class,
        # We only do if we are not a DepthDataset instance and
        # the depth estimation is used.
        if (
            not isinstance(self, DepthDataset) and
            self.use_depth_estimator()
        ):
            self.depth_mode = 'raw'

        # Try to load the reference text embeddings for scene classes
        self.scene_text_embedding_list = []
        self._auxiliary_scene_class_name_embeddings_path = \
            _AuxiliaryDataset.get_key_from_creation_metafile(
                'auxiliary_scene_class_name_embeddings', dataset_path
            )
        if self._auxiliary_scene_class_name_embeddings_path is not None and\
           self._image_embedding_estimator is not None:

            self._auxiliary_scene_class_name_embeddings_path = os.path.join(
                dataset_path, self._auxiliary_scene_class_name_embeddings_path
            )
            self._auxiliary_scene_class_name_embeddings = np.load(
                self._auxiliary_scene_class_name_embeddings_path,
                allow_pickle=True
            ).item()

            # The embeddings are specific to the number of scene classes
            auxiliary_scene_class_names_loaded = False
            if 'scene_use_indoor_domestic_labels' in kwargs:
                if kwargs['scene_use_indoor_domestic_labels']:
                    self._auxiliary_scene_class_name_embeddings = \
                        self._auxiliary_scene_class_name_embeddings[
                            f'{self._image_embedding_estimator}_indoor_domestic'
                        ]
                    auxiliary_scene_class_names_loaded = True

            if not auxiliary_scene_class_names_loaded:
                self._auxiliary_scene_class_name_embeddings = \
                    self._auxiliary_scene_class_name_embeddings[
                        self._image_embedding_estimator
                    ]

            # The embeddings are stored with the used prompt when they
            # were generated. We need to extract the embeddings for the
            # scene classes used in the dataset.
            for class_name in self.config.scene_label_list.class_names:
                class_prompt = self.TEXT_EMBEDDING_PROMPT.format(class_name)
                if class_prompt not in self._auxiliary_scene_class_name_embeddings:
                    raise ValueError(
                        f"Prompt '{class_prompt}' not available in scene "
                        "class name embeddings."
                    )
                scene_class_embedding = \
                    self._auxiliary_scene_class_name_embeddings[class_prompt]
                if self._normalize_embeddings_on_load:
                    scene_class_embedding = \
                        normalize_embedding(scene_class_embedding)
                self.scene_text_embedding_list.append(scene_class_embedding)

        # Try to load the reference text embeddings for semantic classes
        self.semantic_text_embedding_list = []
        self._auxiliary_semantic_class_name_embeddings_path = \
            _AuxiliaryDataset.get_key_from_creation_metafile(
                'auxiliary_semantic_class_name_embeddings', dataset_path
            )
        if self._auxiliary_semantic_class_name_embeddings_path is not None and \
           self._panoptic_embedding_estimator is not None:
            self._auxiliary_semantic_class_name_embeddings_path = os.path.join(
                dataset_path, self._auxiliary_semantic_class_name_embeddings_path
            )
            self._auxiliary_semantic_class_name_embeddings = np.load(
                self._auxiliary_semantic_class_name_embeddings_path,
                allow_pickle=True
            ).item()
            # The embeddings are specific to the number of semantic classes
            # and the panoptic embedding estimator
            self._auxiliary_semantic_class_name_embeddings = \
                self._auxiliary_semantic_class_name_embeddings[
                    self.AUXILIARY_SEMANTIC_CLASS_NAMES_FMT.format(
                        self._semantic_n_classes_str,
                        self._panoptic_embedding_estimator
                    )
                ]

            # The embeddings are stored with the used prompt when they
            # were generated. We need to extract the embeddings for the
            # semantic classes used in the dataset.
            for class_name in self.config.semantic_label_list.class_names:
                class_prompt = self.TEXT_EMBEDDING_PROMPT.format(class_name)
                if class_prompt not in self._auxiliary_semantic_class_name_embeddings:
                    raise ValueError(
                        f"Prompt '{class_prompt}' not available in semantic "
                        "class name embeddings."
                    )
                semantic_class_embedding = \
                    self._auxiliary_semantic_class_name_embeddings[class_prompt]
                if self._normalize_embeddings_on_load:
                    semantic_class_embedding = \
                        normalize_embedding(semantic_class_embedding)
                self.semantic_text_embedding_list.append(
                    semantic_class_embedding
                )

        self.mean_embedding_per_panoptic_class = None
        self.mean_image_embedding_per_panoptic_class = None

        if dataset_path is not None and compute_mean_visual_embeddings:
            self.compute_mean_embeddings()

    def compute_mean_embeddings(self):
        print("Computing mean embeddings per semantic class and image embedding class...")
        self._mean_embedding_per_semantic_class = {}
        self._mean_image_embedding_per_semantic_class = {}

        for idx in tqdm(range(len(self))):
            panoptic_embedding = self.load('panoptic_embedding', idx)
            image_embedding = self.load('image_embedding', idx)

            for panoptic_id, embedding_vec in panoptic_embedding.items():
                # TODO: This should not be hardcoded.
                # Extract semantic class ID
                semantic_id = int(panoptic_id) // (1 << 16)
                if semantic_id not in self._mean_embedding_per_semantic_class:
                    self._mean_embedding_per_semantic_class[semantic_id] = \
                        normalize_embedding(embedding_vec)
                    self._mean_image_embedding_per_semantic_class[semantic_id] = \
                        normalize_embedding(image_embedding)
                else:
                    self._mean_embedding_per_semantic_class[semantic_id] += \
                        normalize_embedding(embedding_vec)
                    self._mean_image_embedding_per_semantic_class[semantic_id] += \
                        normalize_embedding(image_embedding)

        # Handle the case when there are semantic classes, which are not present
        # in the training set (e.g. SUN RGB-D idx 20).
        for semantic_id in range(1, self.semantic_n_classes):
            if semantic_id not in self._mean_embedding_per_semantic_class:
                # Use text embedding as a fallback
                text_embedding = \
                    self.semantic_text_embedding_list[semantic_id]
                self._mean_embedding_per_semantic_class[semantic_id] = \
                    normalize_embedding(text_embedding)
                # the text embedding is already instance focused and never
                # needs to be adjusted according the formula, shown in the
                # paper:
                # mean_embedding_per_semantic_class - alpha * mean_image_embedding
                #
                # however, as we don't indicate, that a class is missing,
                # it's not obvious that the formula should not be applied
                # for some classes (i.e. the one without training examples).
                # to avoid confusion, we set the mean image embedding to
                # a small value, so that the formula can be applied without
                # any issues.
                self._mean_image_embedding_per_semantic_class[semantic_id] \
                    = np.zeros_like(text_embedding) + 1e-9

    @property
    def config(self):
        original_config = super().config
        if self.use_depth_estimator():
            # update depth stats
            depth_stats = self._auxiliary_depth_stats[self._depth_estimator]
            # depth stats should always be from the training split
            train_split = \
                self.split.replace('valid', 'train').replace('test', 'train')
            current_depth_stats = depth_stats.get(train_split, None)
            assert current_depth_stats is not None, \
                f"Depth stats for '{self._depth_estimator}' not available."
            original_config = dataclasses.replace(
                original_config, depth_stats=current_depth_stats
            )

        # Copy the original config and add auxiliary fields. Will
        # return a new DatasetConfigWithAuxiliary instance as
        # DatasetConfig is frozen and cannot be modified.
        return build_dataset_config_with_auxiliary(
            original_config,
            semantic_text_embeddings=self.semantic_text_embedding_list,
            scene_text_embeddings=self.scene_text_embedding_list,
            mean_embedding_per_semantic_class=self._mean_embedding_per_semantic_class,
            mean_image_embedding_per_semantic_class=self._mean_image_embedding_per_semantic_class
        )

    @staticmethod
    def get_potential_additional_sample_keys() -> Tuple[str]:
        return ('depth', 'image_embedding', 'panoptic_embedding')

    @staticmethod
    def get_key_from_creation_metafile(key, dataset_path):
        if dataset_path is None:
            return None
        meta_list = load_creation_metafile(dataset_path)
        assert isinstance(meta_list, list)

        # Collect all values for the requested key from additional_meta entries
        key_values = []
        for meta in meta_list:
            add_meta = meta.get('additional_meta')
            if add_meta is not None and key in add_meta:
                key_values.append(add_meta[key])

        if not key_values:
            return None

        # If only one value or all values are non-dict, return the last one
        # as this is the most recent value.
        if (
            len(key_values) == 1 or
            not any(isinstance(v, dict) for v in key_values)
        ):
            return key_values[-1]

        # Merge dictionary values recursively, non-dict values are overridden
        def merge_dicts(d1, d2):
            result = d1.copy()
            for k, v in d2.items():
                if (
                    k in result and isinstance(result[k], dict) and
                    isinstance(v, dict)
                ):
                    result[k] = merge_dicts(result[k], v)
                else:
                    result[k] = v
            return result

        # Handle dictionaries
        merged_result = {}
        for value in key_values:
            merged_result = merge_dicts(merged_result, value)

        return merged_result

    @staticmethod
    def get_available_sample_keys(split, dataset_path=None) -> Tuple[str]:
        if dataset_path is not None:
            sample_keys = []
            if _AuxiliaryDataset.get_key_from_creation_metafile(
                'auxiliary_depth_estimators', dataset_path
            ) is not None:
                sample_keys.append('depth')

            if _AuxiliaryDataset.get_key_from_creation_metafile(
                'auxiliary_image_embedding_estimators', dataset_path
            ) is not None:
                sample_keys.append('image_embedding')

            if _AuxiliaryDataset.get_key_from_creation_metafile(
                'auxiliary_panoptic_embedding_estimators', dataset_path
            ) is not None:
                sample_keys.append('panoptic_embedding')
            return tuple(sample_keys)

        return _AuxiliaryDataset.get_potential_additional_sample_keys()

    @staticmethod
    def get_available_depth_estimators(split, dataset_path) -> Tuple[str]:
        estimators = _AuxiliaryDataset.get_key_from_creation_metafile(
            'auxiliary_depth_estimators', dataset_path
        )
        if estimators is None:
            return tuple()
        return tuple(estimators)

    @staticmethod
    def get_available_image_embedding_estimators(split, dataset_path) -> Tuple[str]:
        estimators = _AuxiliaryDataset.get_key_from_creation_metafile(
            'auxiliary_image_embedding_estimators', dataset_path
        )
        if estimators is None:
            return tuple()
        return tuple(estimators)

    @staticmethod
    def get_available_panoptic_embedding_estimators(split, dataset_path) -> Tuple[str]:
        estimators = _AuxiliaryDataset.get_key_from_creation_metafile(
            'auxiliary_panoptic_embedding_estimators', dataset_path
        )
        if estimators is None:
            return tuple()
        return tuple(estimators)

    def add_new_depth_estimator(self, name: str) -> None:
        self._available_depth_estimators += (name,)

    def use_depth_estimator(self) -> bool:
        # Check if the depth estimator is part of the auxiliary data
        # creation meta.
        return self._depth_estimator in self._available_depth_estimators

    def use_image_embedding_estimator(self) -> bool:
        if self._image_embedding_estimator is None:
            return False
        return self._image_embedding_estimator in self._available_image_embedding_estimators

    def use_panoptic_embedding_estimator(self) -> bool:
        if self._panoptic_embedding_estimator is None:
            return False
        return self._panoptic_embedding_estimator in self._available_panoptic_embedding_estimators

    def get_filepath(self, path, filename):
        return os.path.join(
            os.path.abspath(self.dataset_path),
            self.split,
            path,
            filename
        )

    def _load_auxiliary_image(self, path, filename) -> np.ndarray:
        fp = self.get_filepath(path, filename)
        # Load the image
        img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(f"Unable to load image: '{fp}'")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_depth(self, idx) -> np.ndarray:
        # use "real" depth image (generated by some depth estimator)).
        # this would be the case for datasets which already have depth
        # images, e.g. NYUv2, ScanNet, etc. and don't rely on estimation
        # of depth images.
        if not self.use_depth_estimator():
            # Fallback to the parent class of the underlying dataset.
            if hasattr(super(), '_load_depth'):
                return super()._load_depth(idx)
            else:
                # Error, if depth should be loaded, depth estimation
                # should not be used, but the parent class does not
                # implement the _load_depth method.
                raise ValueError(
                    f"Depth estimator '{self._depth_estimator}' or depth "
                    "not available."
                )

        filename = self._get_filename(idx)
        depth_img = self._load_auxiliary_image(
            self.AUXILIARY_DEPTH_DIR_FMT.format(self._depth_estimator),
            filename + '.png'
        )
        return depth_img

    def _load_image_embedding(self, idx) -> np.ndarray:
        if not self.use_image_embedding_estimator():
            raise ValueError(
                f"Image embedding estimator '{self._image_embedding_estimator}' not available."
            )
        filename = self._get_filename(idx)
        image_embedding_path = \
            self.AUXILIARY_IMAGE_EMBEDDING_DIR_FMT.format(self._image_embedding_estimator)

        fp = self.get_filepath(image_embedding_path, f'{filename}.npz')

        image_embedding = np.load(fp)['arr_0']
        # Drop batch axis if present
        if image_embedding.ndim == 2:
            image_embedding = image_embedding[0]

        if self._normalize_embeddings_on_load:
            image_embedding = normalize_embedding(image_embedding)

        return image_embedding

    def _load_panoptic_embedding(self, idx) -> PanopticEmbeddingDict:
        if not self.use_panoptic_embedding_estimator():
            raise ValueError(
                f"Panoptic embedding estimator '{self._panoptic_embedding_estimator}' not available."
            )
        filename = self._get_filename(idx)
        panoptic_embedding_path = \
            self.AUXILIARY_PANOPTIC_EMBEDDING_DIR_FMT.format(
                self._semantic_n_classes_str,
                self._panoptic_embedding_estimator
            )
        fp = self.get_filepath(
            panoptic_embedding_path,
            f'{filename}.npz'
        )

        panoptic_embeddings = dict(np.load(fp))
        # Drop batch axis if present
        output = {}
        for key in panoptic_embeddings:
            if panoptic_embeddings[key].ndim == 2:
                output[key] = panoptic_embeddings[key][0]
            else:
                output[key] = panoptic_embeddings[key]

        # Normalize the embeddings
        if self._normalize_embeddings_on_load:
            output = {
                key: normalize_embedding(value) for key, value in output.items()
            }

        # embeddings where stored using savez_compressed which don't support
        # integer keys. We cast the key to integer, as the panoptic id will
        # always be an integer.
        output = {
            int(key): value for key, value in output.items()
        }

        # Convert to PanopticEmbeddingDict
        output = PanopticEmbeddingDict(output)
        return output


def wrap_dataset_with_auxiliary_data(original_dataset):
    class DatasetWithAuxiliaryData(_AuxiliaryDataset, original_dataset):
        # This is a workaround as sample keys are static methods
        # part of the individual datasets, which we need to extend.
        # Doing this without the workaround is tricky as we cannot
        # override static methods in a way that they can access the
        # on execution.
        # Another solution would be to lossen the static method
        # restriction and make them class methods, or include the sample keys
        # in the creation meta file.
        @staticmethod
        def get_available_sample_keys(
            split: str, dataset_path: Optional[str] = None
        ) -> Tuple[str]:
            current_dataset_keys = \
                original_dataset.get_available_sample_keys(split)
            auxiliary_data_keys = \
                _AuxiliaryDataset.get_available_sample_keys(split, dataset_path)
            return current_dataset_keys + auxiliary_data_keys

    return DatasetWithAuxiliaryData
