# -*- coding: utf-8 -*-
"""
.. codeauthor:: Marius Engelhardt <marius.engelhardt@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import numpy as np

from ...dataset_base import DepthStats
from ...dataset_base import SceneLabel
from ...dataset_base import SceneLabelList
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList
from ...utils.img import get_colormap


def _get_camera_intrinsic_normalized():
    width_pixels = 1024
    height_pixels = 768
    fov_x = np.pi/3.0
    fov_y = fov_x*(height_pixels/width_pixels)
    f_x = (width_pixels/np.tan(fov_x/2))/2
    f_y = (height_pixels/np.tan(fov_y/2))/2

    return {
        'fx': f_x/width_pixels, 'fy': f_y/height_pixels,
        'cx': 0.5, 'cy': 0.5,
        'k1': 0, 'k2': 0, 'k3': 0, 'k4': 0, 'k5': 0, 'k6': 0,
        'p1': 0, 'p2': 0
    }


class HypersimMeta:
    SPLITS = ('train', 'valid', 'test')

    @classmethod
    def get_split_filelist_filenames(cls, subsample=None):
        if subsample is None or subsample == 1:
            return {cls.SPLITS[0]: 'train.txt',
                    cls.SPLITS[1]: 'valid.txt',
                    cls.SPLITS[2]: 'test.txt'}
        elif isinstance(subsample, int):
            return {cls.SPLITS[0]: f'train_every_{subsample}th.txt',
                    cls.SPLITS[1]: f'valid_every_{subsample}th.txt',
                    cls.SPLITS[2]: f'test_every_{subsample}th.txt'}
        else:
            raise ValueError(f"Unknown subsample: `{subsample}`")

    SPLIT_DIRS = {SPLITS[0]: 'train',
                  SPLITS[1]: 'valid',
                  SPLITS[2]: 'test'}

    # calculated over the whole train split from train.txt (no subsample)
    # see: my_dataset.depth_compute_stats() for calculation
    TRAIN_SPLIT_DEPTH_STATS = DepthStats(
        min=1.0,
        max=65535.0,
        mean=6245.599769632095,    # updated in v040
        std=7062.149390036199    # updated in v040
    )
    # TODO(v050): remove old depth stats
    _TRAIN_SPLIT_DEPTH_STATS_V030 = DepthStats(
        min=1.0,
        max=65535.0,
        mean=6249.621001070915,
        std=6249.621001070915
    )

    # This is equal to two bytes.
    # The real number is 4166, but two bytes is just nicer for memory layout.
    MAX_INSTANCES_PER_IMAGE = 1 << 16

    # note: depth mode is raw may since we have to exclude the zero depth
    # values that come from clipping the depth to uint16
    DEPTH_MODES = ('raw',)

    CAMERAS = ('virtual',)  # just a dummy camera name
    RGB_INTRINSICS_NORMALIZED = {
        'virtual': _get_camera_intrinsic_normalized()
    }
    DEPTH_INTRINSICS_NORMALIZED = {
        'virtual': {**_get_camera_intrinsic_normalized(), 'a': 0.001, 'b': -1},
    }

    DEPTH_DIR = 'depth'
    RGB_DIR = 'rgb'
    SEMANTIC_DIR = 'semantic_40'
    SEMANTIC_COLORED_DIR = 'semantic_40_colored'
    EXTRINSICS_DIR = 'extrinsics'
    INSTANCES_DIR = 'instance'
    BOXES_3D_DIR = 'boxes_3d'
    ORIENTATIONS_DIR = 'orientations'
    NORMAL_DIR = 'normal'
    SCENE_CLASS_DIR = 'scene_class'

    SEMANTIC_N_CLASSES = 40
    # original hypersim scene labels
    SCENE_LABEL_LIST = SceneLabelList((
        SceneLabel('art gallery'),
        SceneLabel('bathroom'),
        SceneLabel('bedroom'),
        SceneLabel('courtyard'),
        SceneLabel('dining room'),
        SceneLabel('hall'),
        SceneLabel('hallway'),
        SceneLabel('hotel lobby'),
        SceneLabel('kitchen'),
        SceneLabel('lecture theater'),
        SceneLabel('library'),
        SceneLabel('living room'),
        SceneLabel('office'),
        SceneLabel('office (building foyer)'),
        SceneLabel('office (conference room)'),
        SceneLabel('office (home)'),
        SceneLabel('office (waiting area)'),
        SceneLabel('other'),
        SceneLabel('restaurant'),
        SceneLabel('retail space'),
        SceneLabel('staircase'),
        SceneLabel('transit station')
    ))

    # scene labels for indoor domestic environments
    # mapping dict with new labels as keys and tuple of old labels as values
    SCENE_LABEL_MAPPING_INDOOR_DOMESTIC = {
        SceneLabel('void'): (
            SceneLabel('courtyard'),    # more outdoor than indoor
            SceneLabel('transit station'),     # not really domestic
        ),
        SceneLabel('bathroom'): (
            SceneLabel('bathroom'),
        ),
        SceneLabel('bedroom'): (
            SceneLabel('bedroom'),
        ),
        SceneLabel('dining room'): (
            SceneLabel('dining room'),
        ),
        SceneLabel('discussion room'): (
            SceneLabel('office (conference room)'),
            SceneLabel('office (waiting area)'),
        ),
        SceneLabel('hallway'): (
            SceneLabel('hallway'),
        ),
        SceneLabel('kitchen'): (
            SceneLabel('kitchen'),
        ),
        SceneLabel('living room'): (
            SceneLabel('living room'),
        ),
        SceneLabel('office'): (
            SceneLabel('office'),
            SceneLabel('office (home)'),
        ),
        SceneLabel('other indoor'): (
            SceneLabel('art gallery'),
            SceneLabel('hall'),
            SceneLabel('hotel lobby'),
            SceneLabel('lecture theater'),
            SceneLabel('library'),
            SceneLabel('office (building foyer)'),
            SceneLabel('other'),
            SceneLabel('restaurant'),
            SceneLabel('retail space'),
        ),
        SceneLabel('stairs'): (
            SceneLabel('staircase'),
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

    # semantic labels
    CLASS_COLORS = tuple(tuple(c) for c in get_colormap(1 + SEMANTIC_N_CLASSES))
    SEMANTIC_LABEL_LIST = SemanticLabelList([
        # class_name, is_thing, use orientations, color
        SemanticLabel('void',           False, False, CLASS_COLORS[0]),
        SemanticLabel('wall',           False, False, CLASS_COLORS[1]),
        SemanticLabel('floor',          False, False, CLASS_COLORS[2]),
        SemanticLabel('cabinet',        True,  True,  CLASS_COLORS[3]),
        SemanticLabel('bed',            True,  True,  CLASS_COLORS[4]),
        SemanticLabel('chair',          True,  True,  CLASS_COLORS[5]),
        SemanticLabel('sofa',           True,  True,  CLASS_COLORS[6]),
        SemanticLabel('table',          True,  False, CLASS_COLORS[7]),
        SemanticLabel('door',           True,  False, CLASS_COLORS[8]),
        SemanticLabel('window',         True,  False, CLASS_COLORS[9]),
        SemanticLabel('bookshelf',      True,  True,  CLASS_COLORS[10]),
        SemanticLabel('picture',        True,  False, CLASS_COLORS[11]),
        SemanticLabel('counter',        True,  False, CLASS_COLORS[12]),
        SemanticLabel('blinds',         True,  False, CLASS_COLORS[13]),
        SemanticLabel('desk',           True,  False, CLASS_COLORS[14]),
        SemanticLabel('shelves',        True,  True,  CLASS_COLORS[15]),
        SemanticLabel('curtain',        True,  False, CLASS_COLORS[16]),
        SemanticLabel('dresser',        True,  True,  CLASS_COLORS[17]),
        SemanticLabel('pillow',         True,  False, CLASS_COLORS[18]),
        SemanticLabel('mirror',         True,  False, CLASS_COLORS[19]),
        SemanticLabel('floor mat',      True,  False, CLASS_COLORS[20]),
        SemanticLabel('clothes',        True,  False, CLASS_COLORS[21]),
        SemanticLabel('ceiling',        False, False, CLASS_COLORS[22]),
        SemanticLabel('books',          True,  False, CLASS_COLORS[23]),
        SemanticLabel('refrigerator',   True,  True,  CLASS_COLORS[24]),
        SemanticLabel('television',     True,  True,  CLASS_COLORS[25]),
        SemanticLabel('paper',          True,  False, CLASS_COLORS[26]),
        SemanticLabel('towel',          True,  False, CLASS_COLORS[27]),
        SemanticLabel('shower curtain', True,  False, CLASS_COLORS[28]),
        SemanticLabel('box',            True,  False, CLASS_COLORS[29]),
        SemanticLabel('whiteboard',     True,  False, CLASS_COLORS[30]),
        SemanticLabel('person',         True,  True,  CLASS_COLORS[31]),
        SemanticLabel('night stand',    True,  True,  CLASS_COLORS[32]),
        SemanticLabel('toilet',         True,  True,  CLASS_COLORS[33]),
        SemanticLabel('sink',           True,  False, CLASS_COLORS[34]),
        SemanticLabel('lamp',           True,  False, CLASS_COLORS[35]),
        SemanticLabel('bathtub',        True,  False, CLASS_COLORS[36]),
        SemanticLabel('bag',            True,  False, CLASS_COLORS[37]),
        SemanticLabel('otherstructure', True,  False, CLASS_COLORS[38]),
        SemanticLabel('otherfurniture', True,  False, CLASS_COLORS[39]),
        SemanticLabel('otherprop',      True,  False, CLASS_COLORS[40])
    ])

    # exclude scenes/camera trajectories that are probably detrimental to
    # training used in prepare_dataset
    BLACKLIST = {
        'ai_003_001': 'cam_00',  # train
        'ai_004_009': 'cam_01',  # train
        'ai_013_001': '*',  # train (no depth)
        'ai_015_006': '*',  # train
        'ai_023_008': 'cam_01',  # train (no labels)
        'ai_026_020': '*',  # train (no labels)
        'ai_023_009': '*',  # train (no labels)
        'ai_038_007': '*',  # train
        'ai_023_004': 'cam_01',  # train (no labels)
        'ai_023_006': '*',  # train (no labels)
        'ai_026_013': '*',  # train (no labels)
        'ai_026_018': '*',  # train (no labels)
        # 'ai_044_004': 'cam_01',  # test (no labels)
        'ai_046_001': '*',  # train
        # 'ai_046_009': '*',         # already excluded in csv
        'ai_048_004': '*',  # train
        'ai_053_005': '*'  # val
    }
