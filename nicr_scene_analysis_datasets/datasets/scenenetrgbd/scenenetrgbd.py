# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from ...dataset_base import DepthStats
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList


class SceneNetRGBDMeta:
    SPLITS = ('train', 'valid')

    # calculated using a subsampled dataset (see prepare_dataset.py):
    # --n_random_views_to_include_train 3
    # --n_random_views_to_include_valid 6
    # --force_at_least_n_classes_in_view 4
    # see: my_dataset.depth_compute_stats() for calculation
    TRAIN_SPLIT_DEPTH_STATS = DepthStats(
        min=0.0,
        max=20076.0,
        mean=4006.9281155769777,
        std=2459.7763971709933,
    )

    DEPTH_MODES = ('refined',)

    CAMERAS = ('camera1',)     # just a dummy camera name

    DEPTH_DIR = 'depth'
    RGB_DIR = 'rgb'
    SEMANTIC_13_DIR = 'semantic_13'
    SEMANTIC_13_COLORED_DIR = 'semantic_13_colored'

    # number of classes without void (NYUv2 classes)
    SEMANTIC_N_CLASSES = 13
    # there are no orientations, thus, it is set to None
    SEMANTIC_LABEL_LIST = SemanticLabelList((
        # class_name, is_thing, use orientations, color
        SemanticLabel('void',      False, None, (0,   0,   0)),
        SemanticLabel('bed',       True,  None, (0,   0,   255)),
        SemanticLabel('books',     True,  None, (232, 88,  47)),
        SemanticLabel('ceiling',   False, None, (0,   217, 0)),
        SemanticLabel('chair',     True,  None, (148, 0,   240)),
        SemanticLabel('floor',     False, None, (222, 241, 23)),
        SemanticLabel('furniture', True,  None, (255, 205, 205)),
        SemanticLabel('objects',   True,  None, (0,   223, 228)),
        SemanticLabel('picture',   True,  None, (106, 135, 204)),
        SemanticLabel('sofa',      True,  None, (116, 28,  41)),
        SemanticLabel('table',     True,  None, (240, 35,  235)),
        SemanticLabel('tv',        True,  None, (0,   166, 156)),
        SemanticLabel('wall',      False, None, (249, 139, 0)),
        SemanticLabel('window',    True,  None, (225, 228, 194)),
    ))
