# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from ...dataset_base import SceneLabel
from ...dataset_base import SceneLabelList
from ...dataset_base import SemanticLabel
from ...dataset_base import SemanticLabelList


class ADE20KMeta:
    @classmethod
    def get_split_filelist_filename(cls, split):
        return f'{split}.txt'

    SPLIT_TRAIN_CHALLENGE_2016 = 'train_challenge_2016'
    SPLIT_VALID_CHALLENGE_2016 = 'valid_challenge_2016'
    SPLIT_TRAIN_PANOPTIC_2017 = 'train_panoptic_2017'
    SPLIT_VALID_PANOPTIC_2017 = 'valid_panoptic_2017'

    SPLITS = (
        SPLIT_TRAIN_PANOPTIC_2017, SPLIT_VALID_PANOPTIC_2017,
        SPLIT_TRAIN_CHALLENGE_2016, SPLIT_VALID_CHALLENGE_2016,
    )

    _DATA_SAMPLE_KEYS = ('identifier', 'meta', 'rgb')
    # TODO: scene was commented, as the scene labels loaded from the
    # text file often do not match the scene label of the scene label list.
    # Often it seems like the .txt uses _ instead of / but it wasn't
    # consistent. We should check this and fix it.
    SPLIT_SAMPLE_KEYS = {
        SPLIT_TRAIN_CHALLENGE_2016: _DATA_SAMPLE_KEYS + ('semantic', 'scene'),
        SPLIT_VALID_CHALLENGE_2016: _DATA_SAMPLE_KEYS + ('semantic', 'scene'),
        SPLIT_TRAIN_PANOPTIC_2017: _DATA_SAMPLE_KEYS + ('semantic', 'instance', 'scene'),
        SPLIT_VALID_PANOPTIC_2017: _DATA_SAMPLE_KEYS + ('semantic', 'instance', 'scene'),
    }

    CAMERAS = ('cameras1',)    # just a dummy camera name
    IMAGE_DIR = 'rgb'
    SEMANTIC_DIR = 'semantic_150'
    INSTANCES_DIR = 'instance'
    PARTS_DIR = 'parts'
    SCENE_CLASS_DIR = 'scene_class'

    SEMANTIC_N_CLASSES = 150

    # based on Panoptic SegFormer: https://github.com/zhiqi-li/Panoptic-SegFormer/blob/master/converter/panoptic_ade20k_categories.json
    # same thing/stuff classes as used as well in MaskFormer, Mask2Former, and OneFormer
    # note, we additionally inserted a void class at the beginning
    SEMANTIC_LABEL_LIST_CHALLENGE_150 = SemanticLabelList((
            # class_name, is_thing, use orientations, color
            SemanticLabel('void',                                                          False, None, (0,   0,   0)),    # 0
            SemanticLabel('wall',                                                          False, None, (120, 120, 120)),  # 1
            SemanticLabel('building',                                                      False, None, (180, 120, 120)),  # 2
            SemanticLabel('sky',                                                           False, None, (6,   230, 230)),  # 3
            SemanticLabel('floor',                                                         False, None, (80,  50,  50)),   # 4
            SemanticLabel('tree',                                                          False, None, (4,   200, 3)),    # 5
            SemanticLabel('ceiling',                                                       False, None, (120, 120, 80)),   # 6
            SemanticLabel('road, route',                                                   False, None, (140, 140, 140)),  # 7
            SemanticLabel('bed',                                                           True,  None, (204, 5,   255)),  # 8
            SemanticLabel('window ',                                                       True,  None, (230, 230, 230)),  # 9
            SemanticLabel('grass',                                                         False, None, (4,   250, 7)),    # 10
            SemanticLabel('cabinet',                                                       True,  None, (224, 5,   255)),  # 11
            SemanticLabel('sidewalk, pavement',                                            False, None, (235, 255, 7)),    # 12
            SemanticLabel('person',                                                        True,  None, (150, 5,   61)),   # 13
            SemanticLabel('earth, ground',                                                 False, None, (120, 120, 70)),   # 14
            SemanticLabel('door',                                                          True,  None, (8,   255, 51)),   # 15
            SemanticLabel('table',                                                         True,  None, (255, 6,   82)),   # 16
            SemanticLabel('mountain, mount',                                               False, None, (143, 255, 140)),  # 17
            SemanticLabel('plant',                                                         False, None, (204, 255, 4)),    # 18
            SemanticLabel('curtain',                                                       True,  None, (255, 51,  7)),    # 19
            SemanticLabel('chair',                                                         True,  None, (204, 70,  3)),    # 20
            SemanticLabel('car',                                                           True,  None, (0,   102, 200)),  # 21
            SemanticLabel('water',                                                         False, None, (61,  230, 250)),  # 22
            SemanticLabel('painting, picture',                                             True,  None, (255, 6,   51)),   # 23
            SemanticLabel('sofa',                                                          True,  None, (11,  102, 255)),  # 24
            SemanticLabel('shelf',                                                         True,  None, (255, 7,   71)),   # 25
            SemanticLabel('house',                                                         False, None, (255, 9,   224)),  # 26
            SemanticLabel('sea',                                                           False, None, (9,   7,   230)),  # 27
            SemanticLabel('mirror',                                                        True,  None, (220, 220, 220)),  # 28
            SemanticLabel('rug',                                                           False, None, (255, 9,   92)),   # 29
            SemanticLabel('field',                                                         False, None, (112, 9,   255)),  # 30
            SemanticLabel('armchair',                                                      True,  None, (8,   255, 214)),  # 31
            SemanticLabel('seat',                                                          True,  None, (7,   255, 224)),  # 32
            SemanticLabel('fence',                                                         True,  None, (255, 184, 6)),    # 33
            SemanticLabel('desk',                                                          True,  None, (10,  255, 71)),   # 34
            SemanticLabel('rock, stone',                                                   False, None, (255, 41,  10)),   # 35
            SemanticLabel('wardrobe, closet, press',                                       True,  None, (7,   255, 255)),  # 36
            SemanticLabel('lamp',                                                          True,  None, (224, 255, 8)),    # 37
            SemanticLabel('tub',                                                           True,  None, (102, 8,   255)),  # 38
            SemanticLabel('rail',                                                          True,  None, (255, 61,  6)),    # 39
            SemanticLabel('cushion',                                                       True,  None, (255, 194, 7)),    # 40
            SemanticLabel('base, pedestal, stand',                                         False, None, (255, 122, 8)),    # 41
            SemanticLabel('box',                                                           True,  None, (0,   255, 20)),   # 42
            SemanticLabel('column, pillar',                                                True,  None, (255, 8,   41)),   # 43
            SemanticLabel('signboard, sign',                                               True,  None, (255, 5,   153)),  # 44
            SemanticLabel('chest of drawers, chest, bureau, dresser',                      True,  None, (6,   51,  255)),  # 45
            SemanticLabel('counter',                                                       True,  None, (235, 12,  255)),  # 46
            SemanticLabel('sand',                                                          False, None, (160, 150, 20)),   # 47
            SemanticLabel('sink',                                                          True,  None, (0,   163, 255)),  # 48
            SemanticLabel('skyscraper',                                                    False, None, (140, 140, 200)),  # 49
            SemanticLabel('fireplace',                                                     True,  None, (250, 10,  15)),   # 50
            SemanticLabel('refrigerator, icebox',                                          True,  None, (20,  255, 0)),    # 51
            SemanticLabel('grandstand, covered stand',                                     False, None, (31,  255, 0)),    # 52
            SemanticLabel('path',                                                          False, None, (255, 31,  0)),    # 53
            SemanticLabel('stairs',                                                        True,  None, (255, 224, 0)),    # 54
            SemanticLabel('runway',                                                        False, None, (153, 255, 0)),    # 55
            SemanticLabel('case, display case, showcase, vitrine',                         True,  None, (0,   0,   255)),  # 56
            SemanticLabel('pool table, billiard table, snooker table',                     True,  None, (255, 71,  0)),    # 57
            SemanticLabel('pillow',                                                        True,  None, (0,   235, 255)),  # 58
            SemanticLabel('screen door, screen',                                           True,  None, (0,   173, 255)),  # 59
            SemanticLabel('stairway, staircase',                                           False, None, (31,  0,   255)),  # 60
            SemanticLabel('river',                                                         False, None, (11,  200, 200)),  # 61
            SemanticLabel('bridge, span',                                                  False, None, (255, 82,  0)),    # 62
            SemanticLabel('bookcase',                                                      True,  None, (0,   255, 245)),  # 63
            SemanticLabel('blind, screen',                                                 False, None, (0,   61,  255)),  # 64
            SemanticLabel('coffee table',                                                  True,  None, (0,   255, 112)),  # 65
            SemanticLabel('toilet, can, commode, crapper, pot, potty, stool, throne',      True,  None, (0,   255, 133)),  # 66
            SemanticLabel('flower',                                                        True,  None, (255, 0,   0)),    # 67
            SemanticLabel('book',                                                          True,  None, (255, 163, 0)),    # 68
            SemanticLabel('hill',                                                          False, None, (255, 102, 0)),    # 69
            SemanticLabel('bench',                                                         True,  None, (194, 255, 0)),    # 70
            SemanticLabel('countertop',                                                    True,  None, (0,   143, 255)),  # 71
            SemanticLabel('stove',                                                         True,  None, (51,  255, 0)),    # 72
            SemanticLabel('palm, palm tree',                                               True,  None, (0,   82,  255)),  # 73
            SemanticLabel('kitchen island',                                                True,  None, (0,   255, 41)),   # 74
            SemanticLabel('computer',                                                      True,  None, (0,   255, 173)),  # 75
            SemanticLabel('swivel chair',                                                  True,  None, (10,  0,   255)),  # 76
            SemanticLabel('boat',                                                          True,  None, (173, 255, 0)),    # 77
            SemanticLabel('bar',                                                           False, None, (0,   255, 153)),  # 78
            SemanticLabel('arcade machine',                                                True,  None, (255, 92,  0)),    # 79
            SemanticLabel('hovel, hut, hutch, shack, shanty',                              False, None, (255, 0,   255)),  # 80
            SemanticLabel('bus',                                                           True,  None, (255, 0,   245)),  # 81
            SemanticLabel('towel',                                                         True,  None, (255, 0,   102)),  # 82
            SemanticLabel('light',                                                         True,  None, (255, 173, 0)),    # 83
            SemanticLabel('truck',                                                         True,  None, (255, 0,   20)),   # 84
            SemanticLabel('tower',                                                         False, None, (255, 184, 184)),  # 85
            SemanticLabel('chandelier',                                                    True,  None, (0,   31,  255)),  # 86
            SemanticLabel('awning, sunshade, sunblind',                                    True,  None, (0,   255, 61)),   # 87
            SemanticLabel('street lamp',                                                   True,  None, (0,   71,  255)),  # 88
            SemanticLabel('booth',                                                         True,  None, (255, 0,   204)),  # 89
            SemanticLabel('tv',                                                            True,  None, (0,   255, 194)),  # 90
            SemanticLabel('plane',                                                         True,  None, (0,   255, 82)),   # 91
            SemanticLabel('dirt track',                                                    False, None, (0,   10,  255)),  # 92
            SemanticLabel('clothes',                                                       True,  None, (0,   112, 255)),  # 93
            SemanticLabel('pole',                                                          True,  None, (51,  0,   255)),  # 94
            SemanticLabel('land, ground, soil',                                            False, None, (0,   194, 255)),  # 95
            SemanticLabel('bannister, banister, balustrade, balusters, handrail',          True,  None, (0,   122, 255)),  # 96
            SemanticLabel('escalator, moving staircase, moving stairway',                  False, None, (0,   255, 163)),  # 97
            SemanticLabel('ottoman, pouf, pouffe, puff, hassock',                          True,  None, (255, 153, 0)),    # 98
            SemanticLabel('bottle',                                                        True,  None, (0,   255, 10)),   # 99
            SemanticLabel('buffet, counter, sideboard',                                    False, None, (255, 112, 0)),    # 100
            SemanticLabel('poster, posting, placard, notice, bill, card',                  False, None, (143, 255, 0)),    # 101
            SemanticLabel('stage',                                                         False, None, (82,  0,   255)),  # 102
            SemanticLabel('van',                                                           True,  None, (163, 255, 0)),    # 103
            SemanticLabel('ship',                                                          True,  None, (255, 235, 0)),    # 104
            SemanticLabel('fountain',                                                      True,  None, (8,   184, 170)),  # 105
            SemanticLabel('conveyer belt, conveyor belt, conveyer, conveyor, transporter', False, None, (133, 0,   255)),  # 106
            SemanticLabel('canopy',                                                        False, None, (0,   255, 92)),   # 107
            SemanticLabel('washer, automatic washer, washing machine',                     True,  None, (184, 0,   255)),  # 108
            SemanticLabel('plaything, toy',                                                True,  None, (255, 0,   31)),   # 109
            SemanticLabel('pool',                                                          False, None, (0,   184, 255)),  # 110
            SemanticLabel('stool',                                                         True,  None, (0,   214, 255)),  # 111
            SemanticLabel('barrel, cask',                                                  True,  None, (255, 0,   112)),  # 112
            SemanticLabel('basket, handbasket',                                            True,  None, (92,  255, 0)),    # 113
            SemanticLabel('falls',                                                         False, None, (0,   224, 255)),  # 114
            SemanticLabel('tent',                                                          False, None, (112, 224, 255)),  # 115
            SemanticLabel('bag',                                                           True,  None, (70,  184, 160)),  # 116
            SemanticLabel('minibike, motorbike',                                           True,  None, (163, 0,   255)),  # 117
            SemanticLabel('cradle',                                                        False, None, (153, 0,   255)),  # 118
            SemanticLabel('oven',                                                          True,  None, (71,  255, 0)),    # 119
            SemanticLabel('ball',                                                          True,  None, (255, 0,   163)),  # 120
            SemanticLabel('food, solid food',                                              True,  None, (255, 204, 0)),    # 121
            SemanticLabel('step, stair',                                                   True,  None, (255, 0,   143)),  # 122
            SemanticLabel('tank, storage tank',                                            False, None, (0,   255, 235)),  # 123
            SemanticLabel('trade name',                                                    True,  None, (133, 255, 0)),    # 124
            SemanticLabel('microwave',                                                     True,  None, (255, 0,   235)),  # 125
            SemanticLabel('pot',                                                           True,  None, (245, 0,   255)),  # 126
            SemanticLabel('animal',                                                        True,  None, (255, 0,   122)),  # 127
            SemanticLabel('bicycle',                                                       True,  None, (255, 245, 0)),    # 128
            SemanticLabel('lake',                                                          False, None, (10,  190, 212)),  # 129
            SemanticLabel('dishwasher',                                                    True,  None, (214, 255, 0)),    # 130
            SemanticLabel('screen',                                                        True,  None, (0,   204, 255)),  # 131
            SemanticLabel('blanket, cover',                                                False, None, (20,  0,   255)),  # 132
            SemanticLabel('sculpture',                                                     True,  None, (255, 255, 0)),    # 133
            SemanticLabel('hood, exhaust hood',                                            True,  None, (0,   153, 255)),  # 134
            SemanticLabel('sconce',                                                        True,  None, (0,   41,  255)),  # 135
            SemanticLabel('vase',                                                          True,  None, (0,   255, 204)),  # 136
            SemanticLabel('traffic light',                                                 True,  None, (41,  0,   255)),  # 137
            SemanticLabel('tray',                                                          True,  None, (41,  255, 0)),    # 138
            SemanticLabel('trash can',                                                     True,  None, (173, 0,   255)),  # 139
            SemanticLabel('fan',                                                           True,  None, (0,   245, 255)),  # 140
            SemanticLabel('pier',                                                          False, None, (71,  0,   255)),  # 141
            SemanticLabel('crt screen',                                                    False, None, (122, 0,   255)),  # 142
            SemanticLabel('plate',                                                         True,  None, (0,   255, 184)),  # 143
            SemanticLabel('monitor',                                                       True,  None, (0,   92,  255)),  # 144
            SemanticLabel('bulletin board',                                                True,  None, (184, 255, 0)),    # 145
            SemanticLabel('shower',                                                        False, None, (0,   133, 255)),  # 146
            SemanticLabel('radiator',                                                      True,  None, (255, 214, 0)),    # 147
            SemanticLabel('glass, drinking glass',                                         True,  None, (25,  194, 194)),  # 148
            SemanticLabel('clock',                                                         True,  None, (102, 255, 0)),    # 149
            SemanticLabel('flag',                                                          True,  None, (92,  0,   255)),  # 150
    ))
    SEMANTIC_CLASS_COLORS_CHALLENGE_150 = SEMANTIC_LABEL_LIST_CHALLENGE_150.colors

    # list of scene labels used
    # based on sceneCategories.txt file provided in ADEChallengeData2016
    # notes:
    # - there is a "misc" class which can be interpreted as void class
    # - we ordered the classes by name ascending
    SCENE_LABEL_LIST_CHALLENGE_1055 = SceneLabelList((
        SceneLabel(class_name='abbey'),    # 0
        SceneLabel(class_name='access_road'),    # 1
        SceneLabel(class_name='acropolis'),    # 2
        SceneLabel(class_name='air_base'),    # 3
        SceneLabel(class_name='aircraft_carrier_object'),    # 4
        SceneLabel(class_name='airfield'),    # 5
        SceneLabel(class_name='airlock'),    # 6
        SceneLabel(class_name='airplane'),    # 7
        SceneLabel(class_name='airplane_cabin'),    # 8
        SceneLabel(class_name='airport'),    # 9
        SceneLabel(class_name='airport_terminal'),    # 10
        SceneLabel(class_name='airport_ticket_counter'),    # 11
        SceneLabel(class_name='alcove'),    # 12
        SceneLabel(class_name='alley'),    # 13
        SceneLabel(class_name='amphitheater'),    # 14
        SceneLabel(class_name='amphitheater_indoor'),    # 15
        SceneLabel(class_name='amusement_arcade'),    # 16
        SceneLabel(class_name='amusement_park'),    # 17
        SceneLabel(class_name='anechoic_chamber'),    # 18
        SceneLabel(class_name='apartment_building_outdoor'),    # 19
        SceneLabel(class_name='apse_indoor'),    # 20
        SceneLabel(class_name='apse_outdoor'),    # 21
        SceneLabel(class_name='aquarium'),    # 22
        SceneLabel(class_name='aquatic_theater'),    # 23
        SceneLabel(class_name='aqueduct'),    # 24
        SceneLabel(class_name='arbor'),    # 25
        SceneLabel(class_name='arcade'),    # 26
        SceneLabel(class_name='arch'),    # 27
        SceneLabel(class_name='archaelogical_excavation'),    # 28
        SceneLabel(class_name='archipelago'),    # 29
        SceneLabel(class_name='archive'),    # 30
        SceneLabel(class_name='armory'),    # 31
        SceneLabel(class_name='army_base'),    # 32
        SceneLabel(class_name='arrival_gate_indoor'),    # 33
        SceneLabel(class_name='arrival_gate_outdoor'),    # 34
        SceneLabel(class_name='art_gallery'),    # 35
        SceneLabel(class_name='art_school'),    # 36
        SceneLabel(class_name='art_studio'),    # 37
        SceneLabel(class_name='artificial'),    # 38
        SceneLabel(class_name='artists_loft'),    # 39
        SceneLabel(class_name='assembly_hall'),    # 40
        SceneLabel(class_name='assembly_line'),    # 41
        SceneLabel(class_name='assembly_plant'),    # 42
        SceneLabel(class_name='athletic_field_indoor'),    # 43
        SceneLabel(class_name='athletic_field_outdoor'),    # 44
        SceneLabel(class_name='atrium_home'),    # 45
        SceneLabel(class_name='atrium_public'),    # 46
        SceneLabel(class_name='attic'),    # 47
        SceneLabel(class_name='auditorium'),    # 48
        SceneLabel(class_name='auto_factory'),    # 49
        SceneLabel(class_name='auto_mechanics_indoor'),    # 50
        SceneLabel(class_name='auto_mechanics_outdoor'),    # 51
        SceneLabel(class_name='auto_racing_paddock'),    # 52
        SceneLabel(class_name='auto_showroom'),    # 53
        SceneLabel(class_name='awning_deck'),    # 54
        SceneLabel(class_name='back_porch'),    # 55
        SceneLabel(class_name='backdrop'),    # 56
        SceneLabel(class_name='backroom'),    # 57
        SceneLabel(class_name='backseat'),    # 58
        SceneLabel(class_name='backstage'),    # 59
        SceneLabel(class_name='backstage_outdoor'),    # 60
        SceneLabel(class_name='backstairs'),    # 61
        SceneLabel(class_name='backstairs_indoor'),    # 62
        SceneLabel(class_name='backwoods'),    # 63
        SceneLabel(class_name='badlands'),    # 64
        SceneLabel(class_name='badminton_court_indoor'),    # 65
        SceneLabel(class_name='badminton_court_outdoor'),    # 66
        SceneLabel(class_name='baggage_claim'),    # 67
        SceneLabel(class_name='balcony_interior'),    # 68
        SceneLabel(class_name='ball_pit'),    # 69
        SceneLabel(class_name='ballet'),    # 70
        SceneLabel(class_name='ballroom'),    # 71
        SceneLabel(class_name='balustrade'),    # 72
        SceneLabel(class_name='bamboo_forest'),    # 73
        SceneLabel(class_name='bank_indoor'),    # 74
        SceneLabel(class_name='bank_outdoor'),    # 75
        SceneLabel(class_name='bank_vault'),    # 76
        SceneLabel(class_name='banquet_hall'),    # 77
        SceneLabel(class_name='baptistry_indoor'),    # 78
        SceneLabel(class_name='baptistry_outdoor'),    # 79
        SceneLabel(class_name='bar'),    # 80
        SceneLabel(class_name='barbeque'),    # 81
        SceneLabel(class_name='barbershop'),    # 82
        SceneLabel(class_name='barn'),    # 83
        SceneLabel(class_name='barndoor'),    # 84
        SceneLabel(class_name='barnyard'),    # 85
        SceneLabel(class_name='barrack'),    # 86
        SceneLabel(class_name='barrel_storage'),    # 87
        SceneLabel(class_name='baseball'),    # 88
        SceneLabel(class_name='baseball_field'),    # 89
        SceneLabel(class_name='basement'),    # 90
        SceneLabel(class_name='basilica'),    # 91
        SceneLabel(class_name='basin_outdoor'),    # 92
        SceneLabel(class_name='basketball'),    # 93
        SceneLabel(class_name='basketball_court_indoor'),    # 94
        SceneLabel(class_name='basketball_court_outdoor'),    # 95
        SceneLabel(class_name='bath_indoor'),    # 96
        SceneLabel(class_name='bath_outdoor'),    # 97
        SceneLabel(class_name='bathhouse'),    # 98
        SceneLabel(class_name='bathhouse_outdoor'),    # 99
        SceneLabel(class_name='bathroom'),    # 100
        SceneLabel(class_name='batters_box'),    # 101
        SceneLabel(class_name='batting_cage_indoor'),    # 102
        SceneLabel(class_name='batting_cage_outdoor'),    # 103
        SceneLabel(class_name='battlefield'),    # 104
        SceneLabel(class_name='battlement'),    # 105
        SceneLabel(class_name='bay'),    # 106
        SceneLabel(class_name='bayou'),    # 107
        SceneLabel(class_name='bazaar_indoor'),    # 108
        SceneLabel(class_name='bazaar_outdoor'),    # 109
        SceneLabel(class_name='beach'),    # 110
        SceneLabel(class_name='beach_house'),    # 111
        SceneLabel(class_name='beauty_salon'),    # 112
        SceneLabel(class_name='bedchamber'),    # 113
        SceneLabel(class_name='bedroom'),    # 114
        SceneLabel(class_name='beer_garden'),    # 115
        SceneLabel(class_name='beer_hall'),    # 116
        SceneLabel(class_name='belfry'),    # 117
        SceneLabel(class_name='bell_foundry'),    # 118
        SceneLabel(class_name='berth'),    # 119
        SceneLabel(class_name='berth_deck'),    # 120
        SceneLabel(class_name='betting_shop'),    # 121
        SceneLabel(class_name='bicycle_racks'),    # 122
        SceneLabel(class_name='bindery'),    # 123
        SceneLabel(class_name='biology_laboratory'),    # 124
        SceneLabel(class_name='bistro_indoor'),    # 125
        SceneLabel(class_name='bistro_outdoor'),    # 126
        SceneLabel(class_name='bleachers_indoor'),    # 127
        SceneLabel(class_name='bleachers_outdoor'),    # 128
        SceneLabel(class_name='block'),    # 129
        SceneLabel(class_name='boardwalk'),    # 130
        SceneLabel(class_name='boat'),    # 131
        SceneLabel(class_name='boat_deck'),    # 132
        SceneLabel(class_name='boathouse'),    # 133
        SceneLabel(class_name='bog'),    # 134
        SceneLabel(class_name='bomb_shelter_indoor'),    # 135
        SceneLabel(class_name='bookbindery'),    # 136
        SceneLabel(class_name='bookshelf'),    # 137
        SceneLabel(class_name='bookstore'),    # 138
        SceneLabel(class_name='booth'),    # 139
        SceneLabel(class_name='booth_indoor'),    # 140
        SceneLabel(class_name='booth_outdoor'),    # 141
        SceneLabel(class_name='botanical_garden'),    # 142
        SceneLabel(class_name='bottle_storage'),    # 143
        SceneLabel(class_name='bottomland'),    # 144
        SceneLabel(class_name='bow_window_indoor'),    # 145
        SceneLabel(class_name='bow_window_outdoor'),    # 146
        SceneLabel(class_name='bowling_alley'),    # 147
        SceneLabel(class_name='box_seat'),    # 148
        SceneLabel(class_name='boxing_ring'),    # 149
        SceneLabel(class_name='breakfast_table'),    # 150
        SceneLabel(class_name='breakroom'),    # 151
        SceneLabel(class_name='brewery_indoor'),    # 152
        SceneLabel(class_name='brewery_outdoor'),    # 153
        SceneLabel(class_name='bric-a-brac'),    # 154
        SceneLabel(class_name='brickyard_indoor'),    # 155
        SceneLabel(class_name='brickyard_outdoor'),    # 156
        SceneLabel(class_name='bridge'),    # 157
        SceneLabel(class_name='bridle_path'),    # 158
        SceneLabel(class_name='broadleaf'),    # 159
        SceneLabel(class_name='brooklet'),    # 160
        SceneLabel(class_name='bubble_chamber'),    # 161
        SceneLabel(class_name='buffet'),    # 162
        SceneLabel(class_name='building_complex'),    # 163
        SceneLabel(class_name='building_facade'),    # 164
        SceneLabel(class_name='bulkhead'),    # 165
        SceneLabel(class_name='bullpen'),    # 166
        SceneLabel(class_name='bullring'),    # 167
        SceneLabel(class_name='bunk_bed'),    # 168
        SceneLabel(class_name='burial_chamber'),    # 169
        SceneLabel(class_name='bus_depot_indoor'),    # 170
        SceneLabel(class_name='bus_depot_outdoor'),    # 171
        SceneLabel(class_name='bus_interior'),    # 172
        SceneLabel(class_name='bus_shelter'),    # 173
        SceneLabel(class_name='bus_station_indoor'),    # 174
        SceneLabel(class_name='bus_station_outdoor'),    # 175
        SceneLabel(class_name='butchers_shop'),    # 176
        SceneLabel(class_name='butte'),    # 177
        SceneLabel(class_name='bypass'),    # 178
        SceneLabel(class_name='byroad'),    # 179
        SceneLabel(class_name='cabana'),    # 180
        SceneLabel(class_name='cabin_cruiser'),    # 181
        SceneLabel(class_name='cabin_indoor'),    # 182
        SceneLabel(class_name='cabin_outdoor'),    # 183
        SceneLabel(class_name='cafeteria'),    # 184
        SceneLabel(class_name='call_center'),    # 185
        SceneLabel(class_name='campsite'),    # 186
        SceneLabel(class_name='campus'),    # 187
        SceneLabel(class_name='candy_store'),    # 188
        SceneLabel(class_name='canteen'),    # 189
        SceneLabel(class_name='canyon'),    # 190
        SceneLabel(class_name='car_dealership'),    # 191
        SceneLabel(class_name='caravansary'),    # 192
        SceneLabel(class_name='cardroom'),    # 193
        SceneLabel(class_name='cargo_container_interior'),    # 194
        SceneLabel(class_name='cargo_deck'),    # 195
        SceneLabel(class_name='cargo_helicopter'),    # 196
        SceneLabel(class_name='carport_indoor'),    # 197
        SceneLabel(class_name='carport_outdoor'),    # 198
        SceneLabel(class_name='carrousel'),    # 199
        SceneLabel(class_name='cascade'),    # 200
        SceneLabel(class_name='casino_indoor'),    # 201
        SceneLabel(class_name='casino_outdoor'),    # 202
        SceneLabel(class_name='castle'),    # 203
        SceneLabel(class_name='catacomb'),    # 204
        SceneLabel(class_name='cataract'),    # 205
        SceneLabel(class_name='cathedral_indoor'),    # 206
        SceneLabel(class_name='cathedral_outdoor'),    # 207
        SceneLabel(class_name='catwalk'),    # 208
        SceneLabel(class_name='cavern_indoor'),    # 209
        SceneLabel(class_name='cavern_outdoor'),    # 210
        SceneLabel(class_name='cellar'),    # 211
        SceneLabel(class_name='cemetery'),    # 212
        SceneLabel(class_name='chair_lift'),    # 213
        SceneLabel(class_name='chalet'),    # 214
        SceneLabel(class_name='chaparral'),    # 215
        SceneLabel(class_name='chapel'),    # 216
        SceneLabel(class_name='checkout_counter'),    # 217
        SceneLabel(class_name='cheese_factory'),    # 218
        SceneLabel(class_name='chemical_plant'),    # 219
        SceneLabel(class_name='chemistry_lab'),    # 220
        SceneLabel(class_name='chicken_coop_indoor'),    # 221
        SceneLabel(class_name='chicken_coop_outdoor'),    # 222
        SceneLabel(class_name='chicken_farm_indoor'),    # 223
        SceneLabel(class_name='chicken_farm_outdoor'),    # 224
        SceneLabel(class_name='childs_room'),    # 225
        SceneLabel(class_name='choir_loft_interior'),    # 226
        SceneLabel(class_name='chuck_wagon'),    # 227
        SceneLabel(class_name='church_indoor'),    # 228
        SceneLabel(class_name='church_outdoor'),    # 229
        SceneLabel(class_name='circus_tent_indoor'),    # 230
        SceneLabel(class_name='circus_tent_outdoor'),    # 231
        SceneLabel(class_name='city'),    # 232
        SceneLabel(class_name='classroom'),    # 233
        SceneLabel(class_name='clean_room'),    # 234
        SceneLabel(class_name='cliff'),    # 235
        SceneLabel(class_name='clock_tower_indoor'),    # 236
        SceneLabel(class_name='cloister_indoor'),    # 237
        SceneLabel(class_name='cloister_outdoor'),    # 238
        SceneLabel(class_name='closet'),    # 239
        SceneLabel(class_name='clothing_store'),    # 240
        SceneLabel(class_name='coast'),    # 241
        SceneLabel(class_name='coast_road'),    # 242
        SceneLabel(class_name='cockpit'),    # 243
        SceneLabel(class_name='cocktail_lounge'),    # 244
        SceneLabel(class_name='coffee_shop'),    # 245
        SceneLabel(class_name='computer_room'),    # 246
        SceneLabel(class_name='conference_center'),    # 247
        SceneLabel(class_name='conference_hall'),    # 248
        SceneLabel(class_name='conference_room'),    # 249
        SceneLabel(class_name='confessional'),    # 250
        SceneLabel(class_name='construction_site'),    # 251
        SceneLabel(class_name='control_room'),    # 252
        SceneLabel(class_name='control_tower_indoor'),    # 253
        SceneLabel(class_name='control_tower_outdoor'),    # 254
        SceneLabel(class_name='convenience_store_indoor'),    # 255
        SceneLabel(class_name='convenience_store_outdoor'),    # 256
        SceneLabel(class_name='coral_reef'),    # 257
        SceneLabel(class_name='corn_field'),    # 258
        SceneLabel(class_name='corner'),    # 259
        SceneLabel(class_name='corral'),    # 260
        SceneLabel(class_name='corridor'),    # 261
        SceneLabel(class_name='cottage'),    # 262
        SceneLabel(class_name='cottage_garden'),    # 263
        SceneLabel(class_name='country_house'),    # 264
        SceneLabel(class_name='country_road'),    # 265
        SceneLabel(class_name='courthouse'),    # 266
        SceneLabel(class_name='courtroom'),    # 267
        SceneLabel(class_name='courtyard'),    # 268
        SceneLabel(class_name='covered_bridge_interior'),    # 269
        SceneLabel(class_name='crawl_space'),    # 270
        SceneLabel(class_name='creek'),    # 271
        SceneLabel(class_name='crevasse'),    # 272
        SceneLabel(class_name='crosswalk'),    # 273
        SceneLabel(class_name='cultivated'),    # 274
        SceneLabel(class_name='customhouse'),    # 275
        SceneLabel(class_name='cybercafe'),    # 276
        SceneLabel(class_name='dacha'),    # 277
        SceneLabel(class_name='dairy_indoor'),    # 278
        SceneLabel(class_name='dairy_outdoor'),    # 279
        SceneLabel(class_name='dam'),    # 280
        SceneLabel(class_name='dance_floor'),    # 281
        SceneLabel(class_name='dance_school'),    # 282
        SceneLabel(class_name='darkroom'),    # 283
        SceneLabel(class_name='day_care_center'),    # 284
        SceneLabel(class_name='deck-house_boat_deck_house'),    # 285
        SceneLabel(class_name='deck-house_deck_house'),    # 286
        SceneLabel(class_name='delicatessen'),    # 287
        SceneLabel(class_name='dentists_office'),    # 288
        SceneLabel(class_name='department_store'),    # 289
        SceneLabel(class_name='departure_lounge'),    # 290
        SceneLabel(class_name='desert_road'),    # 291
        SceneLabel(class_name='diner_indoor'),    # 292
        SceneLabel(class_name='diner_outdoor'),    # 293
        SceneLabel(class_name='dinette_home'),    # 294
        SceneLabel(class_name='dining_area'),    # 295
        SceneLabel(class_name='dining_car'),    # 296
        SceneLabel(class_name='dining_hall'),    # 297
        SceneLabel(class_name='dining_room'),    # 298
        SceneLabel(class_name='dirt_track'),    # 299
        SceneLabel(class_name='discotheque'),    # 300
        SceneLabel(class_name='distillery'),    # 301
        SceneLabel(class_name='ditch'),    # 302
        SceneLabel(class_name='diving_board'),    # 303
        SceneLabel(class_name='dock'),    # 304
        SceneLabel(class_name='dolmen'),    # 305
        SceneLabel(class_name='donjon'),    # 306
        SceneLabel(class_name='door'),    # 307
        SceneLabel(class_name='doorway_indoor'),    # 308
        SceneLabel(class_name='doorway_outdoor'),    # 309
        SceneLabel(class_name='dorm_room'),    # 310
        SceneLabel(class_name='downtown'),    # 311
        SceneLabel(class_name='drainage_ditch'),    # 312
        SceneLabel(class_name='dress_shop'),    # 313
        SceneLabel(class_name='dressing_room'),    # 314
        SceneLabel(class_name='drill_rig'),    # 315
        SceneLabel(class_name='driveway'),    # 316
        SceneLabel(class_name='driving_range_indoor'),    # 317
        SceneLabel(class_name='driving_range_outdoor'),    # 318
        SceneLabel(class_name='drugstore'),    # 319
        SceneLabel(class_name='dry'),    # 320
        SceneLabel(class_name='dry_dock'),    # 321
        SceneLabel(class_name='dugout'),    # 322
        SceneLabel(class_name='earth_fissure'),    # 323
        SceneLabel(class_name='east_asia'),    # 324
        SceneLabel(class_name='editing_room'),    # 325
        SceneLabel(class_name='electrical_substation'),    # 326
        SceneLabel(class_name='elevated_catwalk'),    # 327
        SceneLabel(class_name='elevator_interior'),    # 328
        SceneLabel(class_name='elevator_lobby'),    # 329
        SceneLabel(class_name='elevator_shaft'),    # 330
        SceneLabel(class_name='embankment'),    # 331
        SceneLabel(class_name='embassy'),    # 332
        SceneLabel(class_name='embrasure'),    # 333
        SceneLabel(class_name='engine_room'),    # 334
        SceneLabel(class_name='entrance'),    # 335
        SceneLabel(class_name='entrance_hall'),    # 336
        SceneLabel(class_name='entranceway_indoor'),    # 337
        SceneLabel(class_name='entranceway_outdoor'),    # 338
        SceneLabel(class_name='entryway_outdoor'),    # 339
        SceneLabel(class_name='escalator_indoor'),    # 340
        SceneLabel(class_name='escalator_outdoor'),    # 341
        SceneLabel(class_name='escarpment'),    # 342
        SceneLabel(class_name='establishment'),    # 343
        SceneLabel(class_name='estaminet'),    # 344
        SceneLabel(class_name='estuary'),    # 345
        SceneLabel(class_name='excavation'),    # 346
        SceneLabel(class_name='exhibition_hall'),    # 347
        SceneLabel(class_name='exterior'),    # 348
        SceneLabel(class_name='fabric_store'),    # 349
        SceneLabel(class_name='factory_indoor'),    # 350
        SceneLabel(class_name='factory_outdoor'),    # 351
        SceneLabel(class_name='fairway'),    # 352
        SceneLabel(class_name='fan'),    # 353
        SceneLabel(class_name='farm'),    # 354
        SceneLabel(class_name='farm_building'),    # 355
        SceneLabel(class_name='farmhouse'),    # 356
        SceneLabel(class_name='fastfood_restaurant'),    # 357
        SceneLabel(class_name='feed_bunk'),    # 358
        SceneLabel(class_name='fence'),    # 359
        SceneLabel(class_name='ferryboat_indoor'),    # 360
        SceneLabel(class_name='field_house'),    # 361
        SceneLabel(class_name='field_road'),    # 362
        SceneLabel(class_name='field_tent_indoor'),    # 363
        SceneLabel(class_name='field_tent_outdoor'),    # 364
        SceneLabel(class_name='fire_escape'),    # 365
        SceneLabel(class_name='fire_station'),    # 366
        SceneLabel(class_name='fire_trench'),    # 367
        SceneLabel(class_name='fireplace'),    # 368
        SceneLabel(class_name='firing_range_indoor'),    # 369
        SceneLabel(class_name='firing_range_outdoor'),    # 370
        SceneLabel(class_name='fish_farm'),    # 371
        SceneLabel(class_name='fishmarket'),    # 372
        SceneLabel(class_name='fishpond'),    # 373
        SceneLabel(class_name='fitting_room_interior'),    # 374
        SceneLabel(class_name='fjord'),    # 375
        SceneLabel(class_name='flashflood'),    # 376
        SceneLabel(class_name='flatlet'),    # 377
        SceneLabel(class_name='flea_market_indoor'),    # 378
        SceneLabel(class_name='flea_market_outdoor'),    # 379
        SceneLabel(class_name='floating_dock'),    # 380
        SceneLabel(class_name='floating_dry_dock'),    # 381
        SceneLabel(class_name='flood'),    # 382
        SceneLabel(class_name='flood_plain'),    # 383
        SceneLabel(class_name='florist_shop_indoor'),    # 384
        SceneLabel(class_name='florist_shop_outdoor'),    # 385
        SceneLabel(class_name='flowerbed'),    # 386
        SceneLabel(class_name='flume_indoor'),    # 387
        SceneLabel(class_name='fly_bridge'),    # 388
        SceneLabel(class_name='flying_buttress'),    # 389
        SceneLabel(class_name='food_court'),    # 390
        SceneLabel(class_name='football'),    # 391
        SceneLabel(class_name='football_field'),    # 392
        SceneLabel(class_name='foothill'),    # 393
        SceneLabel(class_name='forecourt'),    # 394
        SceneLabel(class_name='foreshore'),    # 395
        SceneLabel(class_name='forest_fire'),    # 396
        SceneLabel(class_name='forest_path'),    # 397
        SceneLabel(class_name='forest_road'),    # 398
        SceneLabel(class_name='forklift'),    # 399
        SceneLabel(class_name='formal_garden'),    # 400
        SceneLabel(class_name='fort'),    # 401
        SceneLabel(class_name='fortress'),    # 402
        SceneLabel(class_name='foundry_indoor'),    # 403
        SceneLabel(class_name='foundry_outdoor'),    # 404
        SceneLabel(class_name='fountain'),    # 405
        SceneLabel(class_name='freestanding'),    # 406
        SceneLabel(class_name='freeway'),    # 407
        SceneLabel(class_name='freight_elevator'),    # 408
        SceneLabel(class_name='front_porch'),    # 409
        SceneLabel(class_name='frontseat'),    # 410
        SceneLabel(class_name='funeral_chapel'),    # 411
        SceneLabel(class_name='funeral_home'),    # 412
        SceneLabel(class_name='furnace_room'),    # 413
        SceneLabel(class_name='galley'),    # 414
        SceneLabel(class_name='game_room'),    # 415
        SceneLabel(class_name='gangplank'),    # 416
        SceneLabel(class_name='garage_indoor'),    # 417
        SceneLabel(class_name='garage_outdoor'),    # 418
        SceneLabel(class_name='garbage_dump'),    # 419
        SceneLabel(class_name='garden'),    # 420
        SceneLabel(class_name='gas_station'),    # 421
        SceneLabel(class_name='gas_well'),    # 422
        SceneLabel(class_name='gasworks'),    # 423
        SceneLabel(class_name='gate'),    # 424
        SceneLabel(class_name='gatehouse'),    # 425
        SceneLabel(class_name='gazebo_interior'),    # 426
        SceneLabel(class_name='general_store_indoor'),    # 427
        SceneLabel(class_name='general_store_outdoor'),    # 428
        SceneLabel(class_name='geodesic_dome_indoor'),    # 429
        SceneLabel(class_name='geodesic_dome_outdoor'),    # 430
        SceneLabel(class_name='ghost_town'),    # 431
        SceneLabel(class_name='gift_shop'),    # 432
        SceneLabel(class_name='glacier'),    # 433
        SceneLabel(class_name='glade'),    # 434
        SceneLabel(class_name='glen'),    # 435
        SceneLabel(class_name='golf_course'),    # 436
        SceneLabel(class_name='gorge'),    # 437
        SceneLabel(class_name='granary'),    # 438
        SceneLabel(class_name='grape_arbor'),    # 439
        SceneLabel(class_name='great_hall'),    # 440
        SceneLabel(class_name='greengrocery'),    # 441
        SceneLabel(class_name='greenhouse_indoor'),    # 442
        SceneLabel(class_name='greenhouse_outdoor'),    # 443
        SceneLabel(class_name='grotto'),    # 444
        SceneLabel(class_name='grove'),    # 445
        SceneLabel(class_name='guardhouse'),    # 446
        SceneLabel(class_name='guardroom'),    # 447
        SceneLabel(class_name='guesthouse'),    # 448
        SceneLabel(class_name='gulch'),    # 449
        SceneLabel(class_name='gun_deck_indoor'),    # 450
        SceneLabel(class_name='gun_deck_outdoor'),    # 451
        SceneLabel(class_name='gun_store'),    # 452
        SceneLabel(class_name='gymnasium_indoor'),    # 453
        SceneLabel(class_name='gymnasium_outdoor'),    # 454
        SceneLabel(class_name='hacienda'),    # 455
        SceneLabel(class_name='hallway'),    # 456
        SceneLabel(class_name='handball_court'),    # 457
        SceneLabel(class_name='hangar_indoor'),    # 458
        SceneLabel(class_name='hangar_outdoor'),    # 459
        SceneLabel(class_name='harbor'),    # 460
        SceneLabel(class_name='hardware_store'),    # 461
        SceneLabel(class_name='hat_shop'),    # 462
        SceneLabel(class_name='hatchery'),    # 463
        SceneLabel(class_name='hayfield'),    # 464
        SceneLabel(class_name='hayloft'),    # 465
        SceneLabel(class_name='head_shop'),    # 466
        SceneLabel(class_name='hearth'),    # 467
        SceneLabel(class_name='heath'),    # 468
        SceneLabel(class_name='hedge_maze'),    # 469
        SceneLabel(class_name='hedgerow'),    # 470
        SceneLabel(class_name='heliport'),    # 471
        SceneLabel(class_name='hen_yard'),    # 472
        SceneLabel(class_name='herb_garden'),    # 473
        SceneLabel(class_name='highway'),    # 474
        SceneLabel(class_name='hill'),    # 475
        SceneLabel(class_name='hillock'),    # 476
        SceneLabel(class_name='hockey'),    # 477
        SceneLabel(class_name='hollow'),    # 478
        SceneLabel(class_name='home_office'),    # 479
        SceneLabel(class_name='home_theater'),    # 480
        SceneLabel(class_name='hoodoo'),    # 481
        SceneLabel(class_name='hospital'),    # 482
        SceneLabel(class_name='hospital_room'),    # 483
        SceneLabel(class_name='hot_spring'),    # 484
        SceneLabel(class_name='hot_tub_indoor'),    # 485
        SceneLabel(class_name='hot_tub_outdoor'),    # 486
        SceneLabel(class_name='hotel_breakfast_area'),    # 487
        SceneLabel(class_name='hotel_outdoor'),    # 488
        SceneLabel(class_name='hotel_room'),    # 489
        SceneLabel(class_name='house'),    # 490
        SceneLabel(class_name='housing_estate'),    # 491
        SceneLabel(class_name='housing_project'),    # 492
        SceneLabel(class_name='howdah'),    # 493
        SceneLabel(class_name='hunting_lodge_indoor'),    # 494
        SceneLabel(class_name='hunting_lodge_outdoor'),    # 495
        SceneLabel(class_name='hut'),    # 496
        SceneLabel(class_name='hutment'),    # 497
        SceneLabel(class_name='ice_cream_parlor'),    # 498
        SceneLabel(class_name='ice_floe'),    # 499
        SceneLabel(class_name='ice_shelf'),    # 500
        SceneLabel(class_name='ice_skating_rink_indoor'),    # 501
        SceneLabel(class_name='ice_skating_rink_outdoor'),    # 502
        SceneLabel(class_name='iceberg'),    # 503
        SceneLabel(class_name='igloo'),    # 504
        SceneLabel(class_name='imaret'),    # 505
        SceneLabel(class_name='incinerator_indoor'),    # 506
        SceneLabel(class_name='incinerator_outdoor'),    # 507
        SceneLabel(class_name='indoor_procenium'),    # 508
        SceneLabel(class_name='indoor_round'),    # 509
        SceneLabel(class_name='indoor_seats'),    # 510
        SceneLabel(class_name='industrial_area'),    # 511
        SceneLabel(class_name='industrial_park'),    # 512
        SceneLabel(class_name='inlet'),    # 513
        SceneLabel(class_name='inn_indoor'),    # 514
        SceneLabel(class_name='inn_outdoor'),    # 515
        SceneLabel(class_name='insane_asylum'),    # 516
        SceneLabel(class_name='irrigation_ditch'),    # 517
        SceneLabel(class_name='islet'),    # 518
        SceneLabel(class_name='jacuzzi_indoor'),    # 519
        SceneLabel(class_name='jacuzzi_outdoor'),    # 520
        SceneLabel(class_name='jail_cell'),    # 521
        SceneLabel(class_name='jail_indoor'),    # 522
        SceneLabel(class_name='jail_outdoor'),    # 523
        SceneLabel(class_name='japanese_garden'),    # 524
        SceneLabel(class_name='jetty'),    # 525
        SceneLabel(class_name='jewelry_shop'),    # 526
        SceneLabel(class_name='joss_house'),    # 527
        SceneLabel(class_name='juke_joint'),    # 528
        SceneLabel(class_name='jungle'),    # 529
        SceneLabel(class_name='junk_pile'),    # 530
        SceneLabel(class_name='junkyard'),    # 531
        SceneLabel(class_name='jury_box'),    # 532
        SceneLabel(class_name='kasbah'),    # 533
        SceneLabel(class_name='kennel_indoor'),    # 534
        SceneLabel(class_name='kennel_outdoor'),    # 535
        SceneLabel(class_name='kindergarden_classroom'),    # 536
        SceneLabel(class_name='kiosk_indoor'),    # 537
        SceneLabel(class_name='kiosk_outdoor'),    # 538
        SceneLabel(class_name='kitchen'),    # 539
        SceneLabel(class_name='kitchenette'),    # 540
        SceneLabel(class_name='kraal'),    # 541
        SceneLabel(class_name='lab_classroom'),    # 542
        SceneLabel(class_name='laboratorywet'),    # 543
        SceneLabel(class_name='labyrinth_indoor'),    # 544
        SceneLabel(class_name='labyrinth_outdoor'),    # 545
        SceneLabel(class_name='lagoon'),    # 546
        SceneLabel(class_name='landfill'),    # 547
        SceneLabel(class_name='landing'),    # 548
        SceneLabel(class_name='landing_deck'),    # 549
        SceneLabel(class_name='landing_strip'),    # 550
        SceneLabel(class_name='laundromat'),    # 551
        SceneLabel(class_name='lava_flow'),    # 552
        SceneLabel(class_name='lavatory'),    # 553
        SceneLabel(class_name='lawn'),    # 554
        SceneLabel(class_name='layby'),    # 555
        SceneLabel(class_name='lean-to'),    # 556
        SceneLabel(class_name='lean-to_tent'),    # 557
        SceneLabel(class_name='lecture_room'),    # 558
        SceneLabel(class_name='legislative_chamber'),    # 559
        SceneLabel(class_name='levee'),    # 560
        SceneLabel(class_name='library'),    # 561
        SceneLabel(class_name='library_indoor'),    # 562
        SceneLabel(class_name='library_outdoor'),    # 563
        SceneLabel(class_name='lido_deck_indoor'),    # 564
        SceneLabel(class_name='lido_deck_outdoor'),    # 565
        SceneLabel(class_name='lift_bridge'),    # 566
        SceneLabel(class_name='lighthouse'),    # 567
        SceneLabel(class_name='limousine_interior'),    # 568
        SceneLabel(class_name='liquor_store_indoor'),    # 569
        SceneLabel(class_name='liquor_store_outdoor'),    # 570
        SceneLabel(class_name='living_room'),    # 571
        SceneLabel(class_name='loading_dock'),    # 572
        SceneLabel(class_name='lobby'),    # 573
        SceneLabel(class_name='lock_chamber'),    # 574
        SceneLabel(class_name='locker_room'),    # 575
        SceneLabel(class_name='loft'),    # 576
        SceneLabel(class_name='loge'),    # 577
        SceneLabel(class_name='loggia_outdoor'),    # 578
        SceneLabel(class_name='lookout_station_indoor'),    # 579
        SceneLabel(class_name='lookout_station_outdoor'),    # 580
        SceneLabel(class_name='lower_deck'),    # 581
        SceneLabel(class_name='luggage_van'),    # 582
        SceneLabel(class_name='lumberyard_indoor'),    # 583
        SceneLabel(class_name='lumberyard_outdoor'),    # 584
        SceneLabel(class_name='lyceum'),    # 585
        SceneLabel(class_name='machine_shop'),    # 586
        SceneLabel(class_name='manhole'),    # 587
        SceneLabel(class_name='mansard'),    # 588
        SceneLabel(class_name='mansion'),    # 589
        SceneLabel(class_name='manufactured_home'),    # 590
        SceneLabel(class_name='market_indoor'),    # 591
        SceneLabel(class_name='market_outdoor'),    # 592
        SceneLabel(class_name='marsh'),    # 593
        SceneLabel(class_name='martial_arts_gym'),    # 594
        SceneLabel(class_name='massage_room'),    # 595
        SceneLabel(class_name='mastaba'),    # 596
        SceneLabel(class_name='maternity_ward'),    # 597
        SceneLabel(class_name='mausoleum'),    # 598
        SceneLabel(class_name='meadow'),    # 599
        SceneLabel(class_name='meat_house'),    # 600
        SceneLabel(class_name='medina'),    # 601
        SceneLabel(class_name='megalith'),    # 602
        SceneLabel(class_name='menhir'),    # 603
        SceneLabel(class_name='mens_store_outdoor'),    # 604
        SceneLabel(class_name='mental_institution_indoor'),    # 605
        SceneLabel(class_name='mental_institution_outdoor'),    # 606
        SceneLabel(class_name='mesa'),    # 607
        SceneLabel(class_name='mesoamerican'),    # 608
        SceneLabel(class_name='mess_hall'),    # 609
        SceneLabel(class_name='mews'),    # 610
        SceneLabel(class_name='mezzanine'),    # 611
        SceneLabel(class_name='military_headquarters'),    # 612
        SceneLabel(class_name='military_hospital'),    # 613
        SceneLabel(class_name='military_hut'),    # 614
        SceneLabel(class_name='military_tent'),    # 615
        SceneLabel(class_name='millpond'),    # 616
        SceneLabel(class_name='millrace'),    # 617
        SceneLabel(class_name='mine'),    # 618
        SceneLabel(class_name='mineral_bath'),    # 619
        SceneLabel(class_name='mineshaft'),    # 620
        SceneLabel(class_name='mini_golf_course_indoor'),    # 621
        SceneLabel(class_name='mini_golf_course_outdoor'),    # 622
        SceneLabel(class_name='misc'),    # 623
        SceneLabel(class_name='mission'),    # 624
        SceneLabel(class_name='mobile_home'),    # 625
        SceneLabel(class_name='monastery_indoor'),    # 626
        SceneLabel(class_name='monastery_outdoor'),    # 627
        SceneLabel(class_name='moon_bounce'),    # 628
        SceneLabel(class_name='moor'),    # 629
        SceneLabel(class_name='morgue'),    # 630
        SceneLabel(class_name='mosque_indoor'),    # 631
        SceneLabel(class_name='mosque_outdoor'),    # 632
        SceneLabel(class_name='motel'),    # 633
        SceneLabel(class_name='mountain'),    # 634
        SceneLabel(class_name='mountain_path'),    # 635
        SceneLabel(class_name='mountain_road'),    # 636
        SceneLabel(class_name='mountain_snowy'),    # 637
        SceneLabel(class_name='movie_theater_indoor'),    # 638
        SceneLabel(class_name='movie_theater_outdoor'),    # 639
        SceneLabel(class_name='mudflat'),    # 640
        SceneLabel(class_name='museum_indoor'),    # 641
        SceneLabel(class_name='museum_outdoor'),    # 642
        SceneLabel(class_name='music_store'),    # 643
        SceneLabel(class_name='music_studio'),    # 644
        SceneLabel(class_name='natural'),    # 645
        SceneLabel(class_name='natural_history_museum'),    # 646
        SceneLabel(class_name='natural_spring'),    # 647
        SceneLabel(class_name='naval_base'),    # 648
        SceneLabel(class_name='needleleaf'),    # 649
        SceneLabel(class_name='newsroom'),    # 650
        SceneLabel(class_name='newsstand_indoor'),    # 651
        SceneLabel(class_name='newsstand_outdoor'),    # 652
        SceneLabel(class_name='nightclub'),    # 653
        SceneLabel(class_name='nook'),    # 654
        SceneLabel(class_name='nuclear_power_plant_indoor'),    # 655
        SceneLabel(class_name='nuclear_power_plant_outdoor'),    # 656
        SceneLabel(class_name='nunnery'),    # 657
        SceneLabel(class_name='nursery'),    # 658
        SceneLabel(class_name='nursing_home'),    # 659
        SceneLabel(class_name='nursing_home_outdoor'),    # 660
        SceneLabel(class_name='oasis'),    # 661
        SceneLabel(class_name='oast_house'),    # 662
        SceneLabel(class_name='observation_station'),    # 663
        SceneLabel(class_name='observatory_indoor'),    # 664
        SceneLabel(class_name='observatory_outdoor'),    # 665
        SceneLabel(class_name='observatory_post'),    # 666
        SceneLabel(class_name='ocean'),    # 667
        SceneLabel(class_name='ocean_deep'),    # 668
        SceneLabel(class_name='ocean_shallow'),    # 669
        SceneLabel(class_name='office'),    # 670
        SceneLabel(class_name='office_building'),    # 671
        SceneLabel(class_name='office_cubicles'),    # 672
        SceneLabel(class_name='oil_refinery_indoor'),    # 673
        SceneLabel(class_name='oil_refinery_outdoor'),    # 674
        SceneLabel(class_name='oilrig'),    # 675
        SceneLabel(class_name='one-way_street'),    # 676
        SceneLabel(class_name='open-hearth_furnace'),    # 677
        SceneLabel(class_name='operating_room'),    # 678
        SceneLabel(class_name='operating_table'),    # 679
        SceneLabel(class_name='optician'),    # 680
        SceneLabel(class_name='orchard'),    # 681
        SceneLabel(class_name='orchestra_pit'),    # 682
        SceneLabel(class_name='organ_loft_interior'),    # 683
        SceneLabel(class_name='orlop_deck'),    # 684
        SceneLabel(class_name='ossuary'),    # 685
        SceneLabel(class_name='outbuilding'),    # 686
        SceneLabel(class_name='outcropping'),    # 687
        SceneLabel(class_name='outhouse_indoor'),    # 688
        SceneLabel(class_name='outhouse_outdoor'),    # 689
        SceneLabel(class_name='outside'),    # 690
        SceneLabel(class_name='overpass'),    # 691
        SceneLabel(class_name='oyster_bar'),    # 692
        SceneLabel(class_name='oyster_farm'),    # 693
        SceneLabel(class_name='packaging_plant'),    # 694
        SceneLabel(class_name='pagoda'),    # 695
        SceneLabel(class_name='palace'),    # 696
        SceneLabel(class_name='palace_hall'),    # 697
        SceneLabel(class_name='palestra'),    # 698
        SceneLabel(class_name='pantry'),    # 699
        SceneLabel(class_name='paper_mill'),    # 700
        SceneLabel(class_name='parade_ground'),    # 701
        SceneLabel(class_name='park'),    # 702
        SceneLabel(class_name='parking_garage_indoor'),    # 703
        SceneLabel(class_name='parking_garage_outdoor'),    # 704
        SceneLabel(class_name='parking_lot'),    # 705
        SceneLabel(class_name='parkway'),    # 706
        SceneLabel(class_name='parlor'),    # 707
        SceneLabel(class_name='particle_accelerator'),    # 708
        SceneLabel(class_name='party_tent_indoor'),    # 709
        SceneLabel(class_name='party_tent_outdoor'),    # 710
        SceneLabel(class_name='passenger_deck'),    # 711
        SceneLabel(class_name='pasture'),    # 712
        SceneLabel(class_name='patio'),    # 713
        SceneLabel(class_name='patio_indoor'),    # 714
        SceneLabel(class_name='pavement'),    # 715
        SceneLabel(class_name='pavilion'),    # 716
        SceneLabel(class_name='pawnshop'),    # 717
        SceneLabel(class_name='pawnshop_outdoor'),    # 718
        SceneLabel(class_name='pedestrian_overpass_indoor'),    # 719
        SceneLabel(class_name='penalty_box'),    # 720
        SceneLabel(class_name='performance'),    # 721
        SceneLabel(class_name='perfume_shop'),    # 722
        SceneLabel(class_name='pet_shop'),    # 723
        SceneLabel(class_name='pharmacy'),    # 724
        SceneLabel(class_name='phone_booth'),    # 725
        SceneLabel(class_name='physics_laboratory'),    # 726
        SceneLabel(class_name='piano_store'),    # 727
        SceneLabel(class_name='picnic_area'),    # 728
        SceneLabel(class_name='pier'),    # 729
        SceneLabel(class_name='pig_farm'),    # 730
        SceneLabel(class_name='pilothouse_indoor'),    # 731
        SceneLabel(class_name='pilothouse_outdoor'),    # 732
        SceneLabel(class_name='pinetum'),    # 733
        SceneLabel(class_name='piste_road'),    # 734
        SceneLabel(class_name='pitchers_mound'),    # 735
        SceneLabel(class_name='pizzeria'),    # 736
        SceneLabel(class_name='pizzeria_outdoor'),    # 737
        SceneLabel(class_name='planetarium_indoor'),    # 738
        SceneLabel(class_name='planetarium_outdoor'),    # 739
        SceneLabel(class_name='plantation_house'),    # 740
        SceneLabel(class_name='platform'),    # 741
        SceneLabel(class_name='playground'),    # 742
        SceneLabel(class_name='playroom'),    # 743
        SceneLabel(class_name='plaza'),    # 744
        SceneLabel(class_name='plunge'),    # 745
        SceneLabel(class_name='podium_indoor'),    # 746
        SceneLabel(class_name='podium_outdoor'),    # 747
        SceneLabel(class_name='police_station'),    # 748
        SceneLabel(class_name='pond'),    # 749
        SceneLabel(class_name='pontoon_bridge'),    # 750
        SceneLabel(class_name='poolroom_home'),    # 751
        SceneLabel(class_name='poop_deck'),    # 752
        SceneLabel(class_name='porch'),    # 753
        SceneLabel(class_name='portico'),    # 754
        SceneLabel(class_name='portrait_studio'),    # 755
        SceneLabel(class_name='postern'),    # 756
        SceneLabel(class_name='powder_room'),    # 757
        SceneLabel(class_name='power_plant_outdoor'),    # 758
        SceneLabel(class_name='preserve'),    # 759
        SceneLabel(class_name='print_shop'),    # 760
        SceneLabel(class_name='priory'),    # 761
        SceneLabel(class_name='promenade'),    # 762
        SceneLabel(class_name='promenade_deck'),    # 763
        SceneLabel(class_name='pub_indoor'),    # 764
        SceneLabel(class_name='pub_outdoor'),    # 765
        SceneLabel(class_name='pueblo'),    # 766
        SceneLabel(class_name='pulpit'),    # 767
        SceneLabel(class_name='pump_room'),    # 768
        SceneLabel(class_name='pumping_station'),    # 769
        SceneLabel(class_name='putting_green'),    # 770
        SceneLabel(class_name='quadrangle'),    # 771
        SceneLabel(class_name='questionable'),    # 772
        SceneLabel(class_name='quicksand'),    # 773
        SceneLabel(class_name='quonset_hut_indoor'),    # 774
        SceneLabel(class_name='quonset_hut_outdoor'),    # 775
        SceneLabel(class_name='racecourse'),    # 776
        SceneLabel(class_name='raceway'),    # 777
        SceneLabel(class_name='raft'),    # 778
        SceneLabel(class_name='rail_indoor'),    # 779
        SceneLabel(class_name='rail_outdoor'),    # 780
        SceneLabel(class_name='railroad_track'),    # 781
        SceneLabel(class_name='railway_yard'),    # 782
        SceneLabel(class_name='rainforest'),    # 783
        SceneLabel(class_name='ramp'),    # 784
        SceneLabel(class_name='ranch'),    # 785
        SceneLabel(class_name='ranch_house'),    # 786
        SceneLabel(class_name='reading_room'),    # 787
        SceneLabel(class_name='reception'),    # 788
        SceneLabel(class_name='reception_room'),    # 789
        SceneLabel(class_name='recreation_room'),    # 790
        SceneLabel(class_name='rectory'),    # 791
        SceneLabel(class_name='recycling_plant_indoor'),    # 792
        SceneLabel(class_name='recycling_plant_outdoor'),    # 793
        SceneLabel(class_name='refectory'),    # 794
        SceneLabel(class_name='repair_shop'),    # 795
        SceneLabel(class_name='residential_neighborhood'),    # 796
        SceneLabel(class_name='resort'),    # 797
        SceneLabel(class_name='rest_area'),    # 798
        SceneLabel(class_name='rest_stop'),    # 799
        SceneLabel(class_name='restaurant'),    # 800
        SceneLabel(class_name='restaurant_kitchen'),    # 801
        SceneLabel(class_name='restaurant_patio'),    # 802
        SceneLabel(class_name='restroom_indoor'),    # 803
        SceneLabel(class_name='restroom_outdoor'),    # 804
        SceneLabel(class_name='retaining_wall'),    # 805
        SceneLabel(class_name='revolving_door'),    # 806
        SceneLabel(class_name='rice_paddy'),    # 807
        SceneLabel(class_name='riding_arena'),    # 808
        SceneLabel(class_name='rift_valley'),    # 809
        SceneLabel(class_name='river'),    # 810
        SceneLabel(class_name='road'),    # 811
        SceneLabel(class_name='road_cut'),    # 812
        SceneLabel(class_name='road_indoor'),    # 813
        SceneLabel(class_name='road_outdoor'),    # 814
        SceneLabel(class_name='rock_arch'),    # 815
        SceneLabel(class_name='rock_garden'),    # 816
        SceneLabel(class_name='rodeo'),    # 817
        SceneLabel(class_name='roller_skating_rink_indoor'),    # 818
        SceneLabel(class_name='roller_skating_rink_outdoor'),    # 819
        SceneLabel(class_name='rolling_mill'),    # 820
        SceneLabel(class_name='roof'),    # 821
        SceneLabel(class_name='roof_garden'),    # 822
        SceneLabel(class_name='room'),    # 823
        SceneLabel(class_name='root_cellar'),    # 824
        SceneLabel(class_name='rope_bridge'),    # 825
        SceneLabel(class_name='rotisserie'),    # 826
        SceneLabel(class_name='roundabout'),    # 827
        SceneLabel(class_name='roundhouse'),    # 828
        SceneLabel(class_name='rubble'),    # 829
        SceneLabel(class_name='ruin'),    # 830
        SceneLabel(class_name='runway'),    # 831
        SceneLabel(class_name='sacristy'),    # 832
        SceneLabel(class_name='safari_park'),    # 833
        SceneLabel(class_name='salon'),    # 834
        SceneLabel(class_name='saloon'),    # 835
        SceneLabel(class_name='salt_plain'),    # 836
        SceneLabel(class_name='sanatorium'),    # 837
        SceneLabel(class_name='sand'),    # 838
        SceneLabel(class_name='sand_trap'),    # 839
        SceneLabel(class_name='sandbar'),    # 840
        SceneLabel(class_name='sandbox'),    # 841
        SceneLabel(class_name='sauna'),    # 842
        SceneLabel(class_name='savanna'),    # 843
        SceneLabel(class_name='sawmill'),    # 844
        SceneLabel(class_name='schoolhouse'),    # 845
        SceneLabel(class_name='schoolyard'),    # 846
        SceneLabel(class_name='science_laboratory'),    # 847
        SceneLabel(class_name='science_museum'),    # 848
        SceneLabel(class_name='scriptorium'),    # 849
        SceneLabel(class_name='scrubland'),    # 850
        SceneLabel(class_name='scullery'),    # 851
        SceneLabel(class_name='sea_cliff'),    # 852
        SceneLabel(class_name='seaside'),    # 853
        SceneLabel(class_name='seawall'),    # 854
        SceneLabel(class_name='security_check_point'),    # 855
        SceneLabel(class_name='semidesert'),    # 856
        SceneLabel(class_name='server_room'),    # 857
        SceneLabel(class_name='sewer'),    # 858
        SceneLabel(class_name='sewing_room'),    # 859
        SceneLabel(class_name='shed'),    # 860
        SceneLabel(class_name='shelter'),    # 861
        SceneLabel(class_name='shelter_deck'),    # 862
        SceneLabel(class_name='shelter_tent'),    # 863
        SceneLabel(class_name='shipping_room'),    # 864
        SceneLabel(class_name='shipyard_outdoor'),    # 865
        SceneLabel(class_name='shoe_shop'),    # 866
        SceneLabel(class_name='shop'),    # 867
        SceneLabel(class_name='shopfront'),    # 868
        SceneLabel(class_name='shopping_mall_indoor'),    # 869
        SceneLabel(class_name='shopping_mall_outdoor'),    # 870
        SceneLabel(class_name='shore'),    # 871
        SceneLabel(class_name='shower'),    # 872
        SceneLabel(class_name='shower_room'),    # 873
        SceneLabel(class_name='shrine'),    # 874
        SceneLabel(class_name='shrubbery'),    # 875
        SceneLabel(class_name='sidewalk'),    # 876
        SceneLabel(class_name='signal_box'),    # 877
        SceneLabel(class_name='sinkhole'),    # 878
        SceneLabel(class_name='ski_jump'),    # 879
        SceneLabel(class_name='ski_lodge'),    # 880
        SceneLabel(class_name='ski_resort'),    # 881
        SceneLabel(class_name='ski_slope'),    # 882
        SceneLabel(class_name='sky'),    # 883
        SceneLabel(class_name='skyscraper'),    # 884
        SceneLabel(class_name='skywalk_indoor'),    # 885
        SceneLabel(class_name='skywalk_outdoor'),    # 886
        SceneLabel(class_name='slum'),    # 887
        SceneLabel(class_name='snack_bar'),    # 888
        SceneLabel(class_name='snowbank'),    # 889
        SceneLabel(class_name='snowfield'),    # 890
        SceneLabel(class_name='soccer'),    # 891
        SceneLabel(class_name='south_asia'),    # 892
        SceneLabel(class_name='spillway'),    # 893
        SceneLabel(class_name='sporting_goods_store'),    # 894
        SceneLabel(class_name='squash_court'),    # 895
        SceneLabel(class_name='stable'),    # 896
        SceneLabel(class_name='stadium_outdoor'),    # 897
        SceneLabel(class_name='stage_indoor'),    # 898
        SceneLabel(class_name='stage_outdoor'),    # 899
        SceneLabel(class_name='stage_set'),    # 900
        SceneLabel(class_name='staircase'),    # 901
        SceneLabel(class_name='stall'),    # 902
        SceneLabel(class_name='starting_gate'),    # 903
        SceneLabel(class_name='stateroom'),    # 904
        SceneLabel(class_name='station'),    # 905
        SceneLabel(class_name='steam_plant_outdoor'),    # 906
        SceneLabel(class_name='steel_mill_indoor'),    # 907
        SceneLabel(class_name='steel_mill_outdoor'),    # 908
        SceneLabel(class_name='stone_circle'),    # 909
        SceneLabel(class_name='storage_room'),    # 910
        SceneLabel(class_name='store'),    # 911
        SceneLabel(class_name='storm_cellar'),    # 912
        SceneLabel(class_name='street'),    # 913
        SceneLabel(class_name='streetcar_track'),    # 914
        SceneLabel(class_name='strip_mall'),    # 915
        SceneLabel(class_name='strip_mine'),    # 916
        SceneLabel(class_name='student_center'),    # 917
        SceneLabel(class_name='student_residence'),    # 918
        SceneLabel(class_name='study_hall'),    # 919
        SceneLabel(class_name='submarine_interior'),    # 920
        SceneLabel(class_name='subway_interior'),    # 921
        SceneLabel(class_name='sugar_refinery'),    # 922
        SceneLabel(class_name='sun_deck'),    # 923
        SceneLabel(class_name='sunroom'),    # 924
        SceneLabel(class_name='supermarket'),    # 925
        SceneLabel(class_name='supply_chamber'),    # 926
        SceneLabel(class_name='sushi_bar'),    # 927
        SceneLabel(class_name='swamp'),    # 928
        SceneLabel(class_name='swimming_hole'),    # 929
        SceneLabel(class_name='swimming_pool_indoor'),    # 930
        SceneLabel(class_name='swimming_pool_outdoor'),    # 931
        SceneLabel(class_name='synagogue_indoor'),    # 932
        SceneLabel(class_name='synagogue_outdoor'),    # 933
        SceneLabel(class_name='t-bar_lift'),    # 934
        SceneLabel(class_name='tannery'),    # 935
        SceneLabel(class_name='taxistand'),    # 936
        SceneLabel(class_name='taxiway'),    # 937
        SceneLabel(class_name='tea_garden'),    # 938
        SceneLabel(class_name='teahouse'),    # 939
        SceneLabel(class_name='tearoom'),    # 940
        SceneLabel(class_name='teashop'),    # 941
        SceneLabel(class_name='television_room'),    # 942
        SceneLabel(class_name='television_studio'),    # 943
        SceneLabel(class_name='tennis_court_indoor'),    # 944
        SceneLabel(class_name='tennis_court_outdoor'),    # 945
        SceneLabel(class_name='tent_outdoor'),    # 946
        SceneLabel(class_name='terrace_farm'),    # 947
        SceneLabel(class_name='theater_outdoor'),    # 948
        SceneLabel(class_name='threshing_floor'),    # 949
        SceneLabel(class_name='thriftshop'),    # 950
        SceneLabel(class_name='throne_room'),    # 951
        SceneLabel(class_name='ticket_booth'),    # 952
        SceneLabel(class_name='ticket_window_indoor'),    # 953
        SceneLabel(class_name='tidal_basin'),    # 954
        SceneLabel(class_name='tidal_river'),    # 955
        SceneLabel(class_name='tiltyard'),    # 956
        SceneLabel(class_name='tobacco_shop_indoor'),    # 957
        SceneLabel(class_name='toll_plaza'),    # 958
        SceneLabel(class_name='tollbooth'),    # 959
        SceneLabel(class_name='tollgate'),    # 960
        SceneLabel(class_name='tomb'),    # 961
        SceneLabel(class_name='topiary_garden'),    # 962
        SceneLabel(class_name='tower'),    # 963
        SceneLabel(class_name='town_house'),    # 964
        SceneLabel(class_name='toyshop'),    # 965
        SceneLabel(class_name='track_outdoor'),    # 966
        SceneLabel(class_name='tract_housing'),    # 967
        SceneLabel(class_name='trading_floor'),    # 968
        SceneLabel(class_name='traffic_island'),    # 969
        SceneLabel(class_name='trailer_park'),    # 970
        SceneLabel(class_name='train_interior'),    # 971
        SceneLabel(class_name='train_railway'),    # 972
        SceneLabel(class_name='train_station_outdoor'),    # 973
        SceneLabel(class_name='tree_farm'),    # 974
        SceneLabel(class_name='tree_house'),    # 975
        SceneLabel(class_name='trellis'),    # 976
        SceneLabel(class_name='trench'),    # 977
        SceneLabel(class_name='trestle_bridge'),    # 978
        SceneLabel(class_name='truck_stop'),    # 979
        SceneLabel(class_name='tundra'),    # 980
        SceneLabel(class_name='turkish_bath'),    # 981
        SceneLabel(class_name='upper_balcony'),    # 982
        SceneLabel(class_name='urban'),    # 983
        SceneLabel(class_name='utility_room'),    # 984
        SceneLabel(class_name='valley'),    # 985
        SceneLabel(class_name='van_interior'),    # 986
        SceneLabel(class_name='vat'),    # 987
        SceneLabel(class_name='vegetable_garden'),    # 988
        SceneLabel(class_name='vegetation'),    # 989
        SceneLabel(class_name='vehicle'),    # 990
        SceneLabel(class_name='velodrome_indoor'),    # 991
        SceneLabel(class_name='velodrome_outdoor'),    # 992
        SceneLabel(class_name='ventilation_shaft'),    # 993
        SceneLabel(class_name='veranda'),    # 994
        SceneLabel(class_name='vestibule'),    # 995
        SceneLabel(class_name='vestry'),    # 996
        SceneLabel(class_name='veterinarians_office'),    # 997
        SceneLabel(class_name='viaduct'),    # 998
        SceneLabel(class_name='videostore'),    # 999
        SceneLabel(class_name='village'),    # 1000
        SceneLabel(class_name='vinery'),    # 1001
        SceneLabel(class_name='vineyard'),    # 1002
        SceneLabel(class_name='volcano'),    # 1003
        SceneLabel(class_name='volleyball_court_indoor'),    # 1004
        SceneLabel(class_name='volleyball_court_outdoor'),    # 1005
        SceneLabel(class_name='voting_booth'),    # 1006
        SceneLabel(class_name='waiting_room'),    # 1007
        SceneLabel(class_name='walk_in_freezer'),    # 1008
        SceneLabel(class_name='walkway'),    # 1009
        SceneLabel(class_name='war_room'),    # 1010
        SceneLabel(class_name='warehouse_indoor'),    # 1011
        SceneLabel(class_name='warehouse_outdoor'),    # 1012
        SceneLabel(class_name='washhouse_indoor'),    # 1013
        SceneLabel(class_name='washhouse_outdoor'),    # 1014
        SceneLabel(class_name='washroom'),    # 1015
        SceneLabel(class_name='watchtower'),    # 1016
        SceneLabel(class_name='water'),    # 1017
        SceneLabel(class_name='water_fountain'),    # 1018
        SceneLabel(class_name='water_gate'),    # 1019
        SceneLabel(class_name='water_mill'),    # 1020
        SceneLabel(class_name='water_park'),    # 1021
        SceneLabel(class_name='water_tower'),    # 1022
        SceneLabel(class_name='water_treatment_plant_indoor'),    # 1023
        SceneLabel(class_name='water_treatment_plant_outdoor'),    # 1024
        SceneLabel(class_name='watering_hole'),    # 1025
        SceneLabel(class_name='waterscape'),    # 1026
        SceneLabel(class_name='waterway'),    # 1027
        SceneLabel(class_name='wave'),    # 1028
        SceneLabel(class_name='weighbridge'),    # 1029
        SceneLabel(class_name='western'),    # 1030
        SceneLabel(class_name='wet_bar'),    # 1031
        SceneLabel(class_name='wetland'),    # 1032
        SceneLabel(class_name='wharf'),    # 1033
        SceneLabel(class_name='wheat_field'),    # 1034
        SceneLabel(class_name='whispering_gallery'),    # 1035
        SceneLabel(class_name='widows_walk_indoor'),    # 1036
        SceneLabel(class_name='widows_walk_interior'),    # 1037
        SceneLabel(class_name='wild'),    # 1038
        SceneLabel(class_name='wind_farm'),    # 1039
        SceneLabel(class_name='windmill'),    # 1040
        SceneLabel(class_name='window_seat'),    # 1041
        SceneLabel(class_name='windstorm'),    # 1042
        SceneLabel(class_name='winery'),    # 1043
        SceneLabel(class_name='witness_stand'),    # 1044
        SceneLabel(class_name='woodland'),    # 1045
        SceneLabel(class_name='workroom'),    # 1046
        SceneLabel(class_name='workshop'),    # 1047
        SceneLabel(class_name='wrestling_ring_indoor'),    # 1048
        SceneLabel(class_name='wrestling_ring_outdoor'),    # 1049
        SceneLabel(class_name='yard'),    # 1050
        SceneLabel(class_name='youth_hostel'),    # 1051
        SceneLabel(class_name='zen_garden'),    # 1052
        SceneLabel(class_name='ziggurat'),    # 1053
        SceneLabel(class_name='zoo'),    # 1054
    ))
