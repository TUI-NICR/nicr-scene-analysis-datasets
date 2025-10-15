# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Leonard Rabes <leonard.rabes@tu-ilmenau.de>
"""
from cityscapesscripts.helpers.labels import labels

from ...dataset_base import DepthStats
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList


class CityscapesMeta:
    SPLITS = ('train', 'valid', 'test')

    _DATA_SAMPLE_KEYS = ('identifier', 'meta', 'rgb', 'depth')
    _ANNOTATION_SAMPLE_KEYS = ('semantic', 'instance')
    SPLIT_SAMPLE_KEYS = {
        SPLITS[0]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
        SPLITS[1]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
        SPLITS[2]: _DATA_SAMPLE_KEYS,
    }

    # calculated over the whole train split
    # see: my_dataset.depth_compute_stats() for calculation
    TRAIN_SPLIT_DEPTH_STATS = DepthStats(
        min=3.7578125,
        max=300.0,   # see _load_depth() in dataset.py
        mean=31.715617493177906,
        std=38.70280704877372,
    )
    TRAIN_SPLIT_DEPTH_STATS_DISPARITY = DepthStats(
        min=1.0,
        max=32257.0,
        mean=9069.706336834102,
        std=7178.335960071306
    )

    DEPTH_MODES = ('raw',)

    CAMERAS = ('camera1',)    # just a dummy camera name

    # number of semantic classes without void/unlabeled and
    # license plate (class 34)
    SEMANTIC_N_CLASSES = (19, 33)

    SEMANTIC_LABEL_LIST_REDUCED = SemanticLabelList((
        SemanticLabel('void', False, False, (0, 0, 0)),
    ))
    SEMANTIC_LABEL_LIST_FULL = SemanticLabelList((
        SemanticLabel('void', False, False, (0, 0, 0)),
    ))

    SEMANTIC_CLASS_MAPPING_REDUCED = {
        c: labels[c].trainId+1 if not labels[c].ignoreInEval else 0
        for c in range(1+33)
    }

    for idx, label in enumerate(labels):
        semantic_label = SemanticLabel(
            class_name=label.name,
            is_thing=label.hasInstances,
            use_orientations=False,
            color=label.color
        )

        if not label.ignoreInEval:
            SEMANTIC_LABEL_LIST_REDUCED.add_label(semantic_label)
        # 1+33 classes (0: unlabeled), ignore license plate
        if idx < 33:
            SEMANTIC_LABEL_LIST_FULL.add_label(semantic_label)

    # DEPTH_DIR = 'depth'    # refined depth does not exist
    DEPTH_RAW_DIR = 'depth_raw'
    DISPARITY_RAW_DIR = 'disparity_raw'
    RGB_DIR = 'rgb'
    SEMANTIC_FULL_DIR = 'semantic_33'
    SEMANTIC_REDUCED_DIR = 'semantic_19'
    INSTANCE_DIR = 'instance'
