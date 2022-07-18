# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""

from ...dataset_base import DepthStats
from ...dataset_base import SceneLabel
from ...dataset_base import SceneLabelList
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList
from ..nyuv2.nyuv2 import NYUv2Meta


class SUNRGBDMeta:
    SPLITS = ('train', 'test')
    SPLIT_FILELIST_FILENAMES = {SPLITS[0]: 'train.txt', SPLITS[1]: 'test.txt'}

    # note that mean and std differ depending on the selected depth_mode
    # however, the impact is marginal, therefore, we decided to use the
    # stats for refined depth for both cases
    # stats for raw:
    # - mean: 18320.348967710495
    # - std: 8898.658819551309
    # - min: 161.0
    # - max: 65400.0
    # see: my_dataset.depth_compute_stats() for calculation
    TRAIN_SPLIT_DEPTH_STATS = DepthStats(
        min=1.0,
        max=65528.0,
        mean=19025.14930492213,
        std=9880.916071806689
    )

    DEPTH_MODES = ('refined', 'raw')

    CAMERAS = ('realsense', 'kv2', 'kv1', 'xtion')

    IMAGE_DIR = 'rgb'
    SEMANTIC_DIR = 'semantic'
    SEMANTIC_COLORED_DIR = 'semantic_colored'
    SEMANTIC_COLORED_DIR_SUN = 'semantic_colored_sunrgbd'
    SEMANTIC_COLORED_DIR_NYUV2 = 'semantic_colored_nyuv2'
    DEPTH_DIR = 'depth'
    DEPTH_DIR_RAW = 'depth_raw'
    EXTRINSICS_DIR = 'extrinsics'
    INTRINSICS_DIR = 'intrinsics'

    INSTANCES_DIR = 'instance'
    ORIENTATIONS_DIR = 'orientations'
    BOX_DIR = 'boxes'
    SCENE_CLASS_DIR = 'scene_class'

    # original SUNRGBD scene labels
    SCENE_LABEL_LIST = SceneLabelList((
        SceneLabel('basement'),
        SceneLabel('bathroom'),
        SceneLabel('bedroom'),
        SceneLabel('bookstore'),
        SceneLabel('cafeteria'),
        SceneLabel('classroom'),
        SceneLabel('coffee_room'),
        SceneLabel('computer_room'),
        SceneLabel('conference_room'),
        SceneLabel('corridor'),
        SceneLabel('dancing_room'),
        SceneLabel('dinette'),
        SceneLabel('dining_area'),
        SceneLabel('dining_room'),
        SceneLabel('discussion_area'),
        SceneLabel('exhibition'),
        SceneLabel('furniture_store'),
        SceneLabel('gym'),
        SceneLabel('home'),
        SceneLabel('home_office'),
        SceneLabel('hotel_room'),
        SceneLabel('idk'),
        SceneLabel('indoor_balcony'),
        SceneLabel('kitchen'),
        SceneLabel('lab'),
        SceneLabel('laundromat'),
        SceneLabel('lecture_theatre'),
        SceneLabel('library'),
        SceneLabel('living_room'),
        SceneLabel('lobby'),
        SceneLabel('mail_room'),
        SceneLabel('music_room'),
        SceneLabel('office'),
        SceneLabel('office_dining'),
        SceneLabel('office_kitchen'),
        SceneLabel('playroom'),
        SceneLabel('printer_room'),
        SceneLabel('reception'),
        SceneLabel('reception_room'),
        SceneLabel('recreation_room'),
        SceneLabel('rest_space'),
        SceneLabel('stairs'),
        SceneLabel('storage_room'),
        SceneLabel('study'),
        SceneLabel('study_space'),
    ))

    # scene labels for indoor domestic environments
    # mapping dict with new labels as keys and tuple of old labels as values
    SCENE_LABEL_MAPPING_INDOOR_DOMESTIC = {
        SceneLabel('void'): (
            SceneLabel('furniture_store'),    # contains various scenes -> void
            SceneLabel('hotel_room'),    # contains various scenes -> void
            SceneLabel('idk'),    # we also not, contains various scenes -> void
            SceneLabel('indoor_balcony'),    # only few samples
            SceneLabel('music_room'),    # only few samples
        ),
        SceneLabel('bathroom'): (
            SceneLabel('bathroom'),
        ),
        SceneLabel('bedroom'): (
            SceneLabel('bedroom'),
        ),
        SceneLabel('dining room'): (
            SceneLabel('dinette'),
            SceneLabel('dining_area'),
            SceneLabel('dining_room'),
        ),
        SceneLabel('discussion room'): (
            SceneLabel('conference_room'),
            SceneLabel('discussion_area'),
            SceneLabel('office_dining'),
            SceneLabel('rest_space'),
        ),
        SceneLabel('hallway'): (
            SceneLabel('corridor'),
        ),
        SceneLabel('kitchen'): (
            SceneLabel('coffee_room'),
            SceneLabel('kitchen'),
            SceneLabel('office_kitchen'),
        ),
        SceneLabel('living room'): (
            SceneLabel('living_room'),
        ),
        SceneLabel('office'): (
            SceneLabel('computer_room'),
            SceneLabel('home_office'),
            SceneLabel('office'),
            SceneLabel('study'),
            SceneLabel('study_space')
        ),
        SceneLabel('other indoor'): (
            SceneLabel('basement'),
            SceneLabel('bookstore'),
            SceneLabel('cafeteria'),
            SceneLabel('classroom'),
            SceneLabel('dancing_room'),
            SceneLabel('exhibition'),
            SceneLabel('gym'),
            SceneLabel('home'),
            SceneLabel('lab'),
            SceneLabel('laundromat'),
            SceneLabel('lecture_theatre'),
            SceneLabel('library'),
            SceneLabel('lobby'),
            SceneLabel('mail_room'),
            SceneLabel('playroom'),
            SceneLabel('printer_room'),
            SceneLabel('reception'),
            SceneLabel('reception_room'),
            SceneLabel('recreation_room'),
            SceneLabel('storage_room'),
        ),
        SceneLabel('stairs'): (
            SceneLabel('stairs'),
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

    # number of semantic classes without void
    SEMANTIC_N_CLASSES = 37

    SEMANTIC_CLASS_COLORS = (
        (0,   0,   0),
        (119, 119, 119),
        (244, 243, 131),
        (137, 28,  157),
        (150, 255, 255),
        (54,  114, 113),
        (0,   0,   176),
        (255, 69,  0),
        (87,  112, 255),
        (0,   163, 33),
        (255, 150, 255),
        (255, 180, 10),
        (101, 70,  86),
        (38,  230, 0),
        (255, 120, 70),
        (117, 41,  121),
        (150, 255, 0),
        (132, 0,   255),
        (24,  209, 255),
        (191, 130, 35),
        (219, 200, 109),
        (154, 62,  86),
        (255, 190, 190),
        (255, 0,   255),
        (152, 163, 55),
        (192, 79,  212),
        (230, 230, 230),
        (53,  130, 64),
        (155, 249, 152),
        (87,  64,  34),
        (214, 209, 175),
        (170, 0,   59),
        (255, 0,   0),
        (193, 195, 234),
        (70,  72,  115),
        (255, 255, 0),
        (52,  57,  131),
        (12, 83, 45)
    )

    SEMANTIC_CLASS_COLORS_NYUV2 = NYUv2Meta.SEMANTIC_CLASS_COLORS_40[:38]

    SEMANTIC_LABEL_LIST = SemanticLabelList((
        # class_name, is_thing, use orientations, color
        SemanticLabel('void',           False, False, SEMANTIC_CLASS_COLORS[0]),
        SemanticLabel('wall',           False, False, SEMANTIC_CLASS_COLORS[1]),
        SemanticLabel('floor',          False, False, SEMANTIC_CLASS_COLORS[2]),
        SemanticLabel('cabinet',        True,  True,  SEMANTIC_CLASS_COLORS[3]),
        SemanticLabel('bed',            True,  True,  SEMANTIC_CLASS_COLORS[4]),
        SemanticLabel('chair',          True,  True,  SEMANTIC_CLASS_COLORS[5]),
        SemanticLabel('sofa',           True,  True,  SEMANTIC_CLASS_COLORS[6]),
        SemanticLabel('table',          True,  False, SEMANTIC_CLASS_COLORS[7]),
        SemanticLabel('door',           True,  False, SEMANTIC_CLASS_COLORS[8]),
        SemanticLabel('window',         True,  False, SEMANTIC_CLASS_COLORS[9]),
        SemanticLabel('bookshelf',      True,  True,  SEMANTIC_CLASS_COLORS[10]),
        SemanticLabel('picture',        True,  False, SEMANTIC_CLASS_COLORS[11]),
        SemanticLabel('counter',        True,  False, SEMANTIC_CLASS_COLORS[12]),
        SemanticLabel('blinds',         True,  False, SEMANTIC_CLASS_COLORS[13]),
        SemanticLabel('desk',           True,  False, SEMANTIC_CLASS_COLORS[14]),
        SemanticLabel('shelves',        True,  True,  SEMANTIC_CLASS_COLORS[15]),
        SemanticLabel('curtain',        True,  False, SEMANTIC_CLASS_COLORS[16]),
        SemanticLabel('dresser',        True,  True,  SEMANTIC_CLASS_COLORS[17]),
        SemanticLabel('pillow',         True,  False, SEMANTIC_CLASS_COLORS[18]),
        SemanticLabel('mirror',         True,  False, SEMANTIC_CLASS_COLORS[19]),
        SemanticLabel('floor mat',      True,  False, SEMANTIC_CLASS_COLORS[20]),
        SemanticLabel('clothes',        True,  False, SEMANTIC_CLASS_COLORS[21]),
        SemanticLabel('ceiling',        False, False, SEMANTIC_CLASS_COLORS[22]),
        SemanticLabel('books',          True,  False, SEMANTIC_CLASS_COLORS[23]),
        SemanticLabel('refrigerator',   True,  True,  SEMANTIC_CLASS_COLORS[24]),
        SemanticLabel('television',     True,  True,  SEMANTIC_CLASS_COLORS[25]),
        SemanticLabel('paper',          True,  False, SEMANTIC_CLASS_COLORS[26]),
        SemanticLabel('towel',          True,  False, SEMANTIC_CLASS_COLORS[27]),
        SemanticLabel('shower curtain', True,  False, SEMANTIC_CLASS_COLORS[28]),
        SemanticLabel('box',            True,  False, SEMANTIC_CLASS_COLORS[29]),
        SemanticLabel('whiteboard',     True,  False, SEMANTIC_CLASS_COLORS[30]),
        SemanticLabel('person',         True,  True,  SEMANTIC_CLASS_COLORS[31]),
        SemanticLabel('night stand',    True,  True,  SEMANTIC_CLASS_COLORS[32]),
        SemanticLabel('toilet',         True,  True,  SEMANTIC_CLASS_COLORS[33]),
        SemanticLabel('sink',           True,  False, SEMANTIC_CLASS_COLORS[34]),
        SemanticLabel('lamp',           True,  False, SEMANTIC_CLASS_COLORS[35]),
        SemanticLabel('bathtub',        True,  False, SEMANTIC_CLASS_COLORS[36]),
        SemanticLabel('bag',            True,  False, SEMANTIC_CLASS_COLORS[37]),
    ))

    # create SEMANTIC_LABEL_LIST_NYUV2_COLORS as copy from SEMANTIC_LABEL_LIST
    # and only change color
    SEMANTIC_LABEL_LIST_NYUV2_COLORS = SemanticLabelList((
        SemanticLabel(sem_label.class_name,
                      sem_label.is_thing,
                      sem_label.use_orientations,
                      color)
        for sem_label, color in zip(SEMANTIC_LABEL_LIST, SEMANTIC_CLASS_COLORS_NYUV2)
    ))

    SEMANTIC_CLASS_NAMES_GERMAN = (
        'Void',
        'Wand',
        'Boden',
        'Schrank',
        'Bett',
        'Stuhl',
        'Sofa',
        'Tisch',
        'Tür',
        'Fenster',
        'Bücherregal',
        'Bild',
        'Tresen',
        'Jalousien',
        'Schreibtisch',
        'Regal',
        'Vorhang',
        'Kommode',
        'Kissen',
        'Spiegel',
        'Bodenmatte',
        'Kleidung',
        'Zimmerdecke',
        'Bücher',
        'Kühlschrank',
        'Fernseher',
        'Papier',
        'Handtuch',
        'Duschvorhang',
        'Kiste',
        'Whiteboard',
        'Person',
        'Nachttisch',
        'Toilette',
        'Waschbecken',
        'Lampe',
        'Badewanne',
        'Tasche'
        )

    # create SEMANTIC_LABEL_LIST_GERMAN as copy from SEMANTIC_LABEL_LIST
    # but with translated label
    SEMANTIC_LABEL_LIST_GERMAN = SemanticLabelList((
        SemanticLabel(class_name_german,
                      sem_label.is_thing,
                      sem_label.use_orientations,
                      sem_label.color)
        for sem_label, class_name_german in zip(SEMANTIC_LABEL_LIST, SEMANTIC_CLASS_NAMES_GERMAN)
    ))
