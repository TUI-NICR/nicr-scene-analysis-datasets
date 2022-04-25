# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import cv2

from PythonImageWrapper import Img


def to_mira_img(img, rgb2bgr=False):
    if 3 == img.ndim:
        h, w, n = img.shape
    elif 2 == img.ndim:
        h, w = img.shape
        n = 1
    else:
        raise ValueError(f"Unknown shape: {img.shape}")

    if rgb2bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if img.dtype == 'uint8':
        t = '8U'
    elif img.dtype == 'uint16':
        t = '16U'
    elif img.dtype == 'float32':
        t = '32F'
    else:
        raise ValueError(f"Unknown dtype: {img.dtype}")
    cv_type = getattr(cv2, f'CV_{t}C{n}')

    img_mira = Img(w, h, cv_type, n)
    img_mira.setMat(img)

    return img_mira


def parse_list(comma_sep_str, cast_to=str):
    if cast_to == bool:
        cast_to = lambda x: x.lower() in ['true', '1']

    return [cast_to(e.strip())
            for e in comma_sep_str.strip().split(',')
            if e.strip()]


class AutoGetterSetter:
    def __getattr__(self, name):
        """Generic getter and setter methods for reflection"""
        if name.startswith(('_rget', '_rset')):
            member = name[5:]
            if member not in self.__dict__:
                raise AttributeError(
                    "{} has no attribute '{}'".format(self, name)
                )

            if name.startswith('_rset'):
                # make setter
                def _cb_set(value):
                    setattr(self, member, value)
                return _cb_set
            elif name.startswith('_rget'):
                # make getter
                def _cb_get():
                    return getattr(self, member)
                return _cb_get
        else:
            return super().__getattr__(name)
