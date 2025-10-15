# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Union

from ...utils.misc import partial_class
from .alpha_clip import AlphaCLIPEmbeddingEstimator


_EMBEDDING_ESTIMATORS = {
    # Alpha-CLIP
    AlphaCLIPEmbeddingEstimator.NAME: AlphaCLIPEmbeddingEstimator,
}
# add all variants of each base class as well
for cls in list(_EMBEDDING_ESTIMATORS.values()):
    if issubclass(cls, AlphaCLIPEmbeddingEstimator):
        for model_name in cls.MODEL_LOOKUP_DICT.keys():
            n = f"{cls.NAME}__{model_name}"
            _EMBEDDING_ESTIMATORS[n] = partial_class(cls, model_name=model_name)

KNOWN_EMBEDDING_ESTIMATORS = tuple(sorted(_EMBEDDING_ESTIMATORS.keys()))

EMBEDDING_ESTIMATOR_TYPE = Union[
    AlphaCLIPEmbeddingEstimator,
]


def get_embedding_estimator_class(name: str) -> EMBEDDING_ESTIMATOR_TYPE:
    # force lowercase
    name = name.lower()

    cls = _EMBEDDING_ESTIMATORS.get(name, None)
    if cls is None:
        raise ValueError(f"Unknown embedding estimator: '{name}'")

    return cls
