# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os

from .. import KNOWN_DATASETS
from ..version import get_version


DATASET_BASEPATH = os.environ.get(
    'NICR_SA_DATASET_BASEPATH',
    os.path.join('/datasets_nas/nicr_scene_analysis_datasets/',
                 'version_{}{}{}'.format(*get_version(with_suffix=False)))
)

DATASET_PATH_DICT = {
    key: os.path.join(DATASET_BASEPATH, key)
    for key in KNOWN_DATASETS
}
