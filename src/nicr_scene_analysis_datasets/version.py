# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import subprocess
import warnings

_VERSION_MAJOR = 0
_VERSION_MINOR = 8
_VERSION_MICRO = 3


def get_version(with_suffix=False):    # pragma no cover
    if with_suffix:
        try:
            suffix = subprocess.check_output(
                ['git', 'describe', '--always', '--dirty'],
                cwd=os.path.abspath(os.path.dirname(__file__))
            )
            suffix = suffix.decode().strip()
            # replace - with . to be PEP440 compliant,
            # e.g., d2c4396-dirty -> d2c4396.dirty
            suffix = suffix.replace('-', '.')
        except Exception:
            warnings.warn("Cannot determine version suffix using git.")
            suffix = ''

        return _VERSION_MAJOR, _VERSION_MINOR, _VERSION_MICRO, suffix

    else:
        return _VERSION_MAJOR, _VERSION_MINOR, _VERSION_MICRO


__version__ = '{}.{}.{}'.format(*get_version(with_suffix=False))
