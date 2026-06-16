# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, List

import dataclasses

from ..dataset_base import DatasetConfig


@dataclasses.dataclass(frozen=True)
class DatasetConfigWithAuxiliary(DatasetConfig):
    semantic_text_embeddings: List = None
    scene_text_embeddings: List = None
    mean_embedding_per_semantic_class: Dict = None
    mean_image_embedding_per_semantic_class: Dict = None
    mean_image_embedding_per_scene_class: Dict = None


def build_dataset_config_with_auxiliary(
    original_config: DatasetConfig,
    semantic_text_embeddings: List,
    scene_text_embeddings: List,
    mean_embedding_per_semantic_class: Dict,
    mean_image_embedding_per_semantic_class: Dict,
    mean_image_embedding_per_scene_class: Dict,
) -> DatasetConfigWithAuxiliary:
    """
    Creates a new DatasetConfigWithAuxiliary instance by copying attributes
    from the original config and adding auxiliary fields.
    """
    # Create a new instance of DatasetConfigWithAuxiliary
    # Note: We didn't just use dataclasses.asdict(original_config) as it would
    # also convert its members to a dict, which is not what we want.
    new_config = DatasetConfigWithAuxiliary(
        semantic_label_list=original_config.semantic_label_list,
        semantic_label_list_without_void=original_config.semantic_label_list_without_void,
        scene_label_list=original_config.scene_label_list,
        scene_label_list_without_void=original_config.scene_label_list_without_void,
        depth_stats=original_config.depth_stats,
        semantic_text_embeddings=semantic_text_embeddings,
        scene_text_embeddings=scene_text_embeddings,
        mean_embedding_per_semantic_class=mean_embedding_per_semantic_class,
        mean_image_embedding_per_semantic_class=mean_image_embedding_per_semantic_class,
        mean_image_embedding_per_scene_class=mean_image_embedding_per_scene_class,
    )
    return new_config
