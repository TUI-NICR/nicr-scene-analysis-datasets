# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList


class COCOMeta:
    SPLITS = ('train', 'valid')
    SPLIT_FILELIST_FILENAMES = {SPLITS[0]: 'train.txt', SPLITS[1]: 'valid.txt'}

    _DATA_SAMPLE_KEYS = ('identifier', 'rgb')
    _ANNOTATION_SAMPLE_KEYS = ('semantic', 'instance')
    SPLIT_SAMPLE_KEYS = {
        SPLITS[0]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
        SPLITS[1]: _DATA_SAMPLE_KEYS+_ANNOTATION_SAMPLE_KEYS,
    }

    CAMERAS = ('cameras1',)    # just a dummy camera name

    # it is intended that this folder is named "image" and not "rgb" as COCO
    # has some gray scale images
    IMAGE_DIR = 'image'
    SEMANTIC_DIR = 'semantic'
    SEMANTIC_COLORED_DIR = 'semantic_colored'
    INSTANCES_DIR = 'instance'

    # number of semantic classes without void
    SEMANTIC_N_CLASSES = 133

    # there are no orientations, thus, it is set to None
    SEMANTIC_LABEL_LIST = SemanticLabelList((
        # class_name, is_thing, use orientations, color
        SemanticLabel('void',                  False, None, (0,   0,    0)),
        SemanticLabel('person',                True,  None, (220, 20,   60)),
        SemanticLabel('bicycle',               True,  None, (119, 11,   32)),
        SemanticLabel('car',                   True,  None, (0,   0,    142)),
        SemanticLabel('motorcycle',            True,  None, (0,   0,    230)),
        SemanticLabel('airplane',              True,  None, (106, 0,    228)),
        SemanticLabel('bus',                   True,  None, (0,   60,   100)),
        SemanticLabel('train',                 True,  None, (0,   80,   100)),
        SemanticLabel('truck',                 True,  None, (0,   0,    70)),
        SemanticLabel('boat',                  True,  None, (0,   0,    192)),
        SemanticLabel('traffic light',         True,  None, (250, 170, 30)),
        SemanticLabel('fire hydrant',          True,  None, (100, 170, 30)),
        SemanticLabel('stop sign',             True,  None, (220, 220, 0)),
        SemanticLabel('parking meter',         True,  None, (175, 116, 175)),
        SemanticLabel('bench',                 True,  None, (250, 0,    30)),
        SemanticLabel('bird',                  True,  None, (165, 42,   42)),
        SemanticLabel('cat',                   True,  None, (255, 77,   255)),
        SemanticLabel('dog',                   True,  None, (0,   226,  252)),
        SemanticLabel('horse',                 True,  None, (182, 182,  255)),
        SemanticLabel('sheep',                 True,  None, (0,   82,   0)),
        SemanticLabel('cow',                   True,  None, (120, 166,  157)),
        SemanticLabel('elephant',              True,  None, (110, 76,   0)),
        SemanticLabel('bear',                  True,  None, (174, 57,   255)),
        SemanticLabel('zebra',                 True,  None, (199, 100,  0)),
        SemanticLabel('giraffe',               True,  None, (72,  0,    118)),
        SemanticLabel('backpack',              True,  None, (255, 179,  240)),
        SemanticLabel('umbrella',              True,  None, (0,   125,  92)),
        SemanticLabel('handbag',               True,  None, (209, 0,    151)),
        SemanticLabel('tie',                   True,  None, (188, 208,  182)),
        SemanticLabel('suitcase',              True,  None, (0,   220,  176)),
        SemanticLabel('frisbee',               True,  None, (255, 99,   164)),
        SemanticLabel('skis',                  True,  None, (92,  0,    73)),
        SemanticLabel('snowboard',             True,  None, (133, 129,  255)),
        SemanticLabel('sports ball',           True,  None, (78,  180, 255)),
        SemanticLabel('kite',                  True,  None, (0,   228,  0)),
        SemanticLabel('baseball bat',          True,  None, (174, 255, 243)),
        SemanticLabel('baseball glove',        True,  None, (45,  89,  255)),
        SemanticLabel('skateboard',            True,  None, (134, 134,  103)),
        SemanticLabel('surfboard',             True,  None, (145, 148,  174)),
        SemanticLabel('tennis racket',         True,  None, (255, 208, 186)),
        SemanticLabel('bottle',                True,  None, (197, 226,  255)),
        SemanticLabel('wine glass',            True,  None, (171, 134, 1)),
        SemanticLabel('cup',                   True,  None, (109, 63,   54)),
        SemanticLabel('fork',                  True,  None, (207, 138,  255)),
        SemanticLabel('knife',                 True,  None, (151, 0,    95)),
        SemanticLabel('spoon',                 True,  None, (9,   80,   61)),
        SemanticLabel('bowl',                  True,  None, (84,  105,  51)),
        SemanticLabel('banana',                True,  None, (74,  65,   105)),
        SemanticLabel('apple',                 True,  None, (166, 196,  102)),
        SemanticLabel('sandwich',              True,  None, (208, 195,  210)),
        SemanticLabel('orange',                True,  None, (255, 109,  65)),
        SemanticLabel('broccoli',              True,  None, (0,   143,  149)),
        SemanticLabel('carrot',                True,  None, (179, 0,    194)),
        SemanticLabel('hot dog',               True,  None, (209, 99,  106)),
        SemanticLabel('pizza',                 True,  None, (5,   121,  0)),
        SemanticLabel('donut',                 True,  None, (227, 255,  205)),
        SemanticLabel('cake',                  True,  None, (147, 186,  208)),
        SemanticLabel('chair',                 True,  None, (153, 69,   1)),
        SemanticLabel('couch',                 True,  None, (3,   95,   161)),
        SemanticLabel('potted plant',          True,  None, (163, 255, 0)),
        SemanticLabel('bed',                   True,  None, (119, 0,    170)),
        SemanticLabel('dining table',          True,  None, (0,   182, 199)),
        SemanticLabel('toilet',                True,  None, (0,   165,  120)),
        SemanticLabel('tv',                    True,  None, (183, 130,  88)),
        SemanticLabel('laptop',                True,  None, (95,  32,   0)),
        SemanticLabel('mouse',                 True,  None, (130, 114,  135)),
        SemanticLabel('remote',                True,  None, (110, 129,  133)),
        SemanticLabel('keyboard',              True,  None, (166, 74,   118)),
        SemanticLabel('cell phone',            True,  None, (219, 142, 185)),
        SemanticLabel('microwave',             True,  None, (79,  210,  114)),
        SemanticLabel('oven',                  True,  None, (178, 90,   62)),
        SemanticLabel('toaster',               True,  None, (65,  70,   15)),
        SemanticLabel('sink',                  True,  None, (127, 167,  115)),
        SemanticLabel('refrigerator',          True,  None, (59,  105,  106)),
        SemanticLabel('book',                  True,  None, (142, 108,  45)),
        SemanticLabel('clock',                 True,  None, (196, 172,  0)),
        SemanticLabel('vase',                  True,  None, (95,  54,   80)),
        SemanticLabel('scissors',              True,  None, (128, 76,   255)),
        SemanticLabel('teddy bear',            True,  None, (201, 57,  1)),
        SemanticLabel('hair drier',            True,  None, (246, 0,   122)),
        SemanticLabel('toothbrush',            True,  None, (191, 162,  208)),
        SemanticLabel('banner',                False, None, (255, 255,  128)),
        SemanticLabel('blanket',               False, None, (147, 211,  203)),
        SemanticLabel('bridge',                False, None, (150, 100,  100)),
        SemanticLabel('cardboard',             False, None, (168, 171,  172)),
        SemanticLabel('counter',               False, None, (146, 112,  198)),
        SemanticLabel('curtain',               False, None, (210, 170,  100)),
        SemanticLabel('door-stuff',            False, None, (92,  136,  89)),
        SemanticLabel('floor-wood',            False, None, (218, 88,   184)),
        SemanticLabel('flower',                False, None, (241, 129,  0)),
        SemanticLabel('fruit',                 False, None, (217, 17,   255)),
        SemanticLabel('gravel',                False, None, (124, 74,   181)),
        SemanticLabel('house',                 False, None, (70,  70,   70)),
        SemanticLabel('light',                 False, None, (255, 228,  255)),
        SemanticLabel('mirror-stuff',          False, None, (154, 208,  0)),
        SemanticLabel('net',                   False, None, (193, 0,    92)),
        SemanticLabel('pillow',                False, None, (76,  91,   113)),
        SemanticLabel('platform',              False, None, (255, 180,  195)),
        SemanticLabel('playingfield',          False, None, (106, 154,  176)),
        SemanticLabel('railroad',              False, None, (230, 150,  140)),
        SemanticLabel('river',                 False, None, (60,  143,  255)),
        SemanticLabel('road',                  False, None, (128, 64,   128)),
        SemanticLabel('roof',                  False, None, (92,  82,   55)),
        SemanticLabel('sand',                  False, None, (254, 212,  124)),
        SemanticLabel('sea',                   False, None, (73,  77,   174)),
        SemanticLabel('shelf',                 False, None, (255, 160,  98)),
        SemanticLabel('snow',                  False, None, (255, 255,  255)),
        SemanticLabel('stairs',                False, None, (104, 84,   109)),
        SemanticLabel('tent',                  False, None, (169, 164,  131)),
        SemanticLabel('towel',                 False, None, (225, 199,  255)),
        SemanticLabel('wall-brick',            False, None, (137, 54,   74)),
        SemanticLabel('wall-stone',            False, None, (135, 158,  223)),
        SemanticLabel('wall-tile',             False, None, (7,   246,  231)),
        SemanticLabel('wall-wood',             False, None, (107, 255,  200)),
        SemanticLabel('water-other',           False, None, (58,  41,   149)),
        SemanticLabel('window-blind',          False, None, (183, 121,  142)),
        SemanticLabel('window-other',          False, None, (255, 73,   97)),
        SemanticLabel('tree-merged',           False, None, (107, 142,  35)),
        SemanticLabel('fence-merged',          False, None, (190, 153,  153)),
        SemanticLabel('ceiling-merged',        False, None, (146, 139,  141)),
        SemanticLabel('sky-other-merged',      False, None, (70,  130,  180)),
        SemanticLabel('cabinet-merged',        False, None, (134, 199,  156)),
        SemanticLabel('table-merged',          False, None, (209, 226,  140)),
        SemanticLabel('floor-other-merged',    False, None, (96,  36,   108)),
        SemanticLabel('pavement-merged',       False, None, (96,  96,   96)),
        SemanticLabel('mountain-merged',       False, None, (64,  170,  64)),
        SemanticLabel('grass-merged',          False, None, (152, 251,  152)),
        SemanticLabel('dirt-merged',           False, None, (208, 229,  228)),
        SemanticLabel('paper-merged',          False, None, (206, 186,  171)),
        SemanticLabel('food-other-merged',     False, None, (152, 161,  64)),
        SemanticLabel('building-other-merged', False, None, (116, 112,  0)),
        SemanticLabel('rock-merged',           False, None, (0,   114,  143)),
        SemanticLabel('wall-other-merged',     False, None, (102, 102,  156)),
        SemanticLabel('rug-merged',            False, None, (250, 141, 255))
    ))

    # COCO IDs are not contiguous
    COCO_ID = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19,
               20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38,
               39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55,
               56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75,
               76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93,
               95, 100, 107, 109, 112, 118, 119, 122, 125, 128, 130, 133, 138,
               141, 144, 145, 147, 148, 149, 151, 154, 155, 156, 159, 161, 166,
               168, 171, 175, 176, 177, 178, 180, 181, 184, 185, 186, 187, 188,
               189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200)
