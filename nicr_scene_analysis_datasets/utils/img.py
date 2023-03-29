# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import numpy as np
from PIL import Image


from ._colormaps import COLORMAP_VISUALLY_DISTINCT_VOID_PLUS_256
from ._colormaps import COLORMAP_VISUALLY_DISTINCT_256


def dimshuffle(input_img, from_axes, to_axes):
    # check axes parameter
    if from_axes.find('0') == -1 or from_axes.find('1') == -1:
        raise ValueError("`from_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if to_axes.find('0') == -1 or to_axes.find('1') == -1:
        raise ValueError("`to_axes` must contain both axis0 ('0') and"
                         "axis 1 ('1')")
    if len(from_axes) != len(input_img.shape):
        raise ValueError("Number of axis given by `from_axes` does not match "
                         "the number of axis in `input_img`")

    # handle special cases for channel axis
    to_axes_c = to_axes.find('c')
    from_axes_c = from_axes.find('c')
    # remove channel axis (only grayscale image)
    if to_axes_c == -1 and from_axes_c >= 0:
        if input_img.shape[from_axes_c] != 1:
            raise ValueError('Cannot remove channel axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_c)
        from_axes = from_axes.replace('c', '')

    # handle special cases for batch axis
    to_axes_b = to_axes.find('b')
    from_axes_b = from_axes.find('b')
    # remove batch axis
    if to_axes_b == -1 and from_axes_b >= 0:
        if input_img.shape[from_axes_b] != 1:
            raise ValueError('Cannot remove batch axis because size is not '
                             'equal to 1')
        input_img = input_img.squeeze(axis=from_axes_b)
        from_axes = from_axes.replace('b', '')

    # add new batch axis (in front)
    if to_axes_b >= 0 and from_axes_b == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'b' + from_axes

    # add new channel axis (in front)
    if to_axes_c >= 0 and from_axes_c == -1:
        input_img = input_img[np.newaxis]
        from_axes = 'c' + from_axes

    return np.transpose(input_img, [from_axes.find(a) for a in to_axes])


def get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def get_visual_distinct_colormap(with_void: bool = True) -> np.ndarray:
    # useful for visualizing instances
    if with_void:
        return COLORMAP_VISUALLY_DISTINCT_VOID_PLUS_256
    else:
        return COLORMAP_VISUALLY_DISTINCT_256


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath, 'PNG')


def blend_images(img1, img2, alpha=0.5):
    """
    Function to alpha composite two images. The output image is calculated
    by img_out = ( 1 - ( alpha*( img2 > 0 ) ) )*img1 + alpha*img2.

    Parameters
    ----------
    img1 : {numpy.ndarray, list, tuple}
        The first image with axes '01' or '01c' and of dtype 'uintX' or
        'floatX'. (background image).
    img2 : {numpy.ndarray, list, tuple}
        The second image with axes '01' or '01c' and of dtype 'uintX' or
        'floatX' (foreground image).
    alpha : {float}
        The alpha value to use: 0.0 <= alpha <= 1.0.

    Returns
    -------
    img_out : numpy.ndarray
        The resulting image.

    """
    # ensure that img is a numpy object
    img1 = np.asanyarray(img1)
    img2 = np.asanyarray(img2)
    assert img1.dtype == img2.dtype
    assert img1.ndim == img2.ndim

    # alpha composite images
    if img2.ndim == 3:
        mask = np.any(img2 > 0, axis=2)
    else:
        mask = img2 > 0

    result = img1.copy()
    result[mask, ...] = \
        ((1-alpha)*img1[mask, ...] + alpha*img2[mask, ...]).astype(img1.dtype)

    return result
