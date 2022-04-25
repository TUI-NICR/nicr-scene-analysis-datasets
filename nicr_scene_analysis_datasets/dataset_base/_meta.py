# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthStats:
    min: float
    max: float
    mean: float
    std: float
