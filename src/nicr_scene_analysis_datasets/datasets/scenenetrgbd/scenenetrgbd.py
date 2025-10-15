# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from ...dataset_base import DepthStats
from ...dataset_base import SceneLabel
from ...dataset_base import SceneLabelList
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList


class SceneNetRGBDMeta:
    SPLITS = ('train', 'valid')
    SPLIT_FILELIST_FILENAMES = {SPLITS[0]: 'train.txt', SPLITS[1]: 'valid.txt'}

    _DATA_SAMPLE_KEYS = ('identifier', 'rgb', 'depth')
    _ANNOTATION_SAMPLE_KEYS = ('semantic', 'instance', 'scene')
    SPLIT_SAMPLE_KEYS = {
        SPLITS[0]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
        SPLITS[1]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
    }

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
    INSTANCES_DIR = 'instance'
    SCENE_CLASS_DIR = 'scene'

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

    # original scene labels
    SCENE_LABEL_LIST = SceneLabelList((
        SceneLabel('bathroom'),
        SceneLabel('bedroom'),
        SceneLabel('kitchen'),
        SceneLabel('living_room'),
        SceneLabel('office')
    ))

    # scene labels for indoor domestic environments
    # mapping dict with new labels as keys and tuple of old labels as values
    SCENE_LABEL_MAPPING_INDOOR_DOMESTIC = {
        SceneLabel('void'): (
        ),
        SceneLabel('bathroom'): (
            SceneLabel('bathroom'),
        ),
        SceneLabel('bedroom'): (
            SceneLabel('bedroom'),
        ),
        SceneLabel('dining room'): (
        ),
        SceneLabel('discussion room'): (
        ),
        SceneLabel('hallway'): (
        ),
        SceneLabel('kitchen'): (
            SceneLabel('kitchen'),
        ),
        SceneLabel('living room'): (
            SceneLabel('living_room'),
        ),
        SceneLabel('office'): (
            SceneLabel('office'),
        ),
        SceneLabel('other indoor'): (
        ),
        SceneLabel('stairs'): (
        )
    }

    SCENE_LABEL_LIST_INDOOR_DOMESTIC = SceneLabelList(
        tuple(SCENE_LABEL_MAPPING_INDOOR_DOMESTIC.keys())
    )
    # create index mapping
    SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX = {}
    for new_label, old_labels in SCENE_LABEL_MAPPING_INDOOR_DOMESTIC.items():
        for old_label in old_labels:
            old_idx = SCENE_LABEL_LIST.index(old_label)
            new_idx = SCENE_LABEL_LIST_INDOOR_DOMESTIC.index(new_label)
            SCENE_LABEL_IDX_TO_SCENE_LABEL_INDOOR_DOMESTIC_IDX[old_idx] = new_idx
