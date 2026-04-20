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

_DATASET_PATH_DICT = {
    key: os.path.join(DATASET_BASEPATH, key)
    for key in KNOWN_DATASETS
}

class DatasetPathDict(dict):
    def __getitem__(self, key):
        value = super().__getitem__(key)
        print(f"Getting test dataset path for '{key}': '{value}'")
        return value

DATASET_PATH_DICT = DatasetPathDict(_DATASET_PATH_DICT)
