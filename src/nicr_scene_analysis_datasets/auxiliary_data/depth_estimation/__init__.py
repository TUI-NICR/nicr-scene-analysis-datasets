# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Union

from ...utils.misc import partial_class
from .hugging_face import DepthAnythingV2DepthEstimator
from .hugging_face import DinoV2DPTDepthEstimator
from .hugging_face import ZoeDepthDepthEstimator
from .hugging_face import _HuggingFaceDepthEstimator


_DEPTH_ESTIMATORS = {
    # DepthAnything V2
    DepthAnythingV2DepthEstimator.NAME: DepthAnythingV2DepthEstimator,
    # ZoeDepth
    ZoeDepthDepthEstimator.NAME: ZoeDepthDepthEstimator,
    # Dino V2 with Dense Prediction Transformer (DPT) head for depth estimation
    DinoV2DPTDepthEstimator.NAME: DinoV2DPTDepthEstimator
}
# add all variants of each base class as well
for cls in list(_DEPTH_ESTIMATORS.values()):
    if issubclass(cls, _HuggingFaceDepthEstimator):
        for model_name in cls.MODEL_LOOKUP_DICT.keys():
            n = f"{cls.NAME}__{model_name}"
            _DEPTH_ESTIMATORS[n] = partial_class(cls, model_name=model_name)

KNOWN_DEPTH_ESTIMATORS = tuple(sorted(_DEPTH_ESTIMATORS.keys()))

DEPTH_ESTIMATOR_TYPE = Union[
    DepthAnythingV2DepthEstimator,
    ZoeDepthDepthEstimator,
    DinoV2DPTDepthEstimator
]


def get_depth_estimator_class(name: str) -> DEPTH_ESTIMATOR_TYPE:
    # force lowercase
    name = name.lower()

    cls = _DEPTH_ESTIMATORS.get(name, None)
    if cls is None:
        raise ValueError(f"Unknown depth estimator: '{name}'")

    return cls
