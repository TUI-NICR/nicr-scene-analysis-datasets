# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import dataclasses
from typing import Union

from ._annotation import SemanticLabelList
from ._annotation import SceneLabelList
from ._meta import DepthStats


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    semantic_label_list: SemanticLabelList
    semantic_label_list_without_void: SemanticLabelList
    scene_label_list: SceneLabelList
    scene_label_list_without_void: SceneLabelList
    depth_stats: Union[DepthStats, None]


def build_dataset_config(
    semantic_label_list: SemanticLabelList,
    scene_label_list: Union[SceneLabelList, None] = None,
    depth_stats: Union[DepthStats, None] = None
) -> DatasetConfig:
    """
    Builds a dataset config from a semantic and scene label list and known
    depth stats.

    Notes
    -----
    The function assumes that the first element in the semantic label list has
    the void label.
    """
    scene_label_list = scene_label_list or SceneLabelList(())

    # build semantic label list without void
    semantic_label_list_without_void = SemanticLabelList(())
    for idx, label in enumerate(semantic_label_list):
        # skip void
        if idx == 0:
            # we always have 0 as void
            continue
        semantic_label_list_without_void.add_label(label)

    # build scene label list without void
    scene_label_list_without_void = SceneLabelList(())
    for label in scene_label_list:
        # skip void
        if 'void' == label.class_name.lower():
            # indoor domestic class labels contain a void class
            continue
        scene_label_list_without_void.add_label(label)

    # create dataset config
    config = DatasetConfig(
        semantic_label_list,
        semantic_label_list_without_void,
        scene_label_list,
        scene_label_list_without_void,
        depth_stats
    )

    return config
