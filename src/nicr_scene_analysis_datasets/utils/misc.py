# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from functools import lru_cache
from functools import partialmethod


@lru_cache()
def partial_class(cls, *args, **kwargs):
    # modified version of: https://stackoverflow.com/a/38911383
    if args or kwargs:

        class PartialClass(cls):
            __init__ = partialmethod(cls.__init__, *args, **kwargs)

        return PartialClass
    else:
        return cls
