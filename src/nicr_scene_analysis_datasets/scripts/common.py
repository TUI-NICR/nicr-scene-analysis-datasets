# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import List, Optional, Tuple, Union

import numpy as np

from .. import get_dataset_class
from ..utils import img as img_utils


DATASET_COLORMAPS = {
    'auto_n': {},
    'cityscapes_19': {'semantic_n_classes': 19},
    'cityscapes_33': {'semantic_n_classes': 33},
    'coco': {},
    'hypersim': {},
    'nyuv2_13': {'semantic_n_classes': 13},
    'nyuv2_40': {'semantic_n_classes': 40},
    'nyuv2_894': {'semantic_n_classes': 894},
    'scannet_20': {'semantic_n_classes': 20},
    'scannet_40': {'semantic_n_classes': 40},
    'scannet_200': {'semantic_n_classes': 200},
    'scannet_549': {'semantic_n_classes': 549},
    'scenenetrgbd': {},
    'sunrgbd': {},
    'visual_distinct': {}
}

AVAILABLE_COLORMAPS = tuple(DATASET_COLORMAPS.keys())


def get_colormap(
    name: str,
    n: Optional[int] = 256,
    return_names: bool = False
) -> Union[np.ndarray, Tuple[List[str], np.ndarray]]:
    if 'auto_n' == name:
        # generate color map with n colors
        colors = np.array(
            img_utils.get_colormap(n)
        )
        names = [f'{i}' for i in range(n)]
    elif 'visual_distinct' == name:
        # use visually distinct colors (useful for visualizing instances)
        colors = np.array(
            img_utils.get_visual_distinct_colormap(with_void=True)
        )
        names = [f'{i}' for i in range(colors.shape[0])]
    else:
        # use colors from dataset
        dataset_name = name.split('_')[0]
        dataset = get_dataset_class(dataset_name)(
            disable_prints=True,
            **DATASET_COLORMAPS[name]
        )
        # with void class
        colors = dataset.config.semantic_label_list.colors_array
        names = dataset.config.semantic_label_list.class_names

    if not return_names:
        return colors

    return colors, names


def print_section(section_name: str, section_content: str = ''):
    print(f"===== {section_name.upper()} =====")
    if section_content:
        print(section_content+"\n")
