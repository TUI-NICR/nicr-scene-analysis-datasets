# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>

"""
import argparse as ap
from glob import glob
import inspect
import os
import warnings

import cv2
import numpy as np

from ..utils.img import blend_images

from .common import AVAILABLE_COLORMAPS
from .common import get_colormap
from .common import print_section


WINDOW_NAME = 'Semantic/Instance Viewer'

USAGE = inspect.cleandoc(
    """
    ARROW_RIGHT / n / d: next image
    ARROW_LEFT / p / a:  previous image
    +:                   increase alpha for blending (see `--color-path`)
    -:                   decrease alpha for blending (see `--color-path`)
    CTRL + s:            save current image (save screenshot at current scale)
    q:                   quit
    CLICK ON IMAGE:      show label (and opt. name if meaningful colormap is
                         given) at cursor position
    """
)


def _parse_args():
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Viewer for semantic/instance/panoptic annotations."
    )

    parser.add_argument(
        'path_or_filepath',
        type=str,
        help="Path to image(s) to load."
    )
    parser.add_argument(
        '--file-extension',
        type=str,
        default='png',
        help="File extension for images to search for."
    )
    parser.add_argument(
        '--rgb-format',
        type=str,
        choices=(
            'r', 'g', 'b',         # display only a single channel
            'g*256+b',             # display as g+b channel as uint16
            'r*256*256+g*256+b'    # display as r+g+b channel as uint32
        ),
        default=None,
        help="How to interpret the channels if the input is an RGB image, "
             "i.e., use 'r' / 'g' / 'b' to display only a single channel, use "
             "'g*256+b' to display g+b channel as uint16 - useful for uint16 "
             "instance ids stored in 2 channels, or use 'r*256*256+g*256+b' to "
             "to display r+g+b channel as uint32 - useful for uint32 instances "
             "or panoptic labels."
    )
    parser.add_argument(
        '-w', '--without_void',
        action='store_true',
        default=False,
        help="Ignore void class (actually do value+1 before)."
    )
    parser.add_argument(
        '-c', '--colormap',
        type=str,
        default='auto_n',
        choices=AVAILABLE_COLORMAPS,
        help="Colormap to use for visualization (all include the void class)"
    )
    parser.add_argument(
        '-n', '--n-colors',
        type=int,
        default=256,
        help="Number of colors when 'colormap' is `auto_n`."
    )
    parser.add_argument(
        '--color-path',
        default=None,
        type=str,
        help="Optional path to the directory with the corresponding color "
             "images. If given, the corresponding color image is overlaid with "
             "current image to be displayed."
    )
    parser.add_argument(
        '--color-alpha',
        default=0.5,
        type=float,
        help="Alpha value in [0,1] for blending the color image with the "
             "current image."
    )
    parser.add_argument(
        '--auto-size',
        action='store_true',
        help="Open the window with CV_WINDOW_AUTOSIZE flag, i.e., displaying "
             "the image at the original size. This option is useful if you "
             "want to save a screenshot at the original size."
    )

    return parser.parse_args()


def _load_and_show(filepaths,
                   idx,
                   cmap,
                   names,
                   color_path=None,
                   color_alpha=0.5,
                   rgb_format=None,
                   without_void=False):
    # load image
    filepath = filepaths[idx]
    if os.path.splitext(filepath)[1] == '.npy':
        img = np.load(filepath).astype('uint8')
    else:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

    if img.ndim == 3:
        # handle RGB images
        assert rgb_format is not None, "RGB image but no `--rgb-format` given"

        # image is BGR !
        if rgb_format == 'b':
            img = img[..., 0]
        if rgb_format == 'g':
            img = img[..., 1]
        elif rgb_format == 'r':
            img = img[..., 2]
        elif rgb_format == 'g*256+b':
            img = img.astype('uint16')
            img = img[..., 1]*256 + img[..., 0]
        elif rgb_format == 'r*256*256+g*256+b':
            img = img.astype('uint32')
            img = img[..., 2]*256*256 + img[..., 1]*256 + img[..., 0]
        else:
            raise ValueError(f"Unknown `rgb_format`: {rgb_format}")

    print(f"[{idx}/{len(filepaths)-1}] {os.path.basename(filepath)} "
          f"(WxH: {img.shape[1]}x{img.shape[0]})")

    if img.ndim != 2:
        print(f"Unknown shape: {img.shape}")

    if without_void:
        img += 1

    if img.max() >= len(cmap):
        warnings.warn(
            "Selected colormap has less colors than the maximum value in the "
            f"image: {len(cmap)} vs. {img.max()}. Using modulo to map values "
            "to colors."
        )
        img %= len(cmap)

    img_colored = cmap[img]
    img_colored_bgr = cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)

    # potentially overlay with rgb image
    if color_path:
        basename = os.path.splitext(os.path.basename(filepaths[idx]))[0]
        # get corresponding color filepaths
        color_filepaths = glob(os.path.join(color_path, f'{basename}.*'))

        if not color_filepaths:
            print("No corresponding color image found. "
                  "Using a black image instead. ")
            color_img = np.zeros_like(img_colored_bgr)
        else:
            if len(color_filepaths) > 1:
                print("Multiple corresponding color images found. "
                      f"Taking the first one: {color_filepaths[0]}")
            color_img = cv2.imread(color_filepaths[0])

        # overlay color image with semantic/ instance image
        img_colored_bgr = blend_images(
            color_img,
            img_colored_bgr,
            alpha=color_alpha
        )

    cv2.imshow(WINDOW_NAME, img_colored_bgr)

    def mouse_callback(event, x, y, flags, params):
        # 0: mouse move, 1: left button down, 2: right button down
        if 1 == event:
            idx = img[y, x]
            print(f"(x={x}, y={y}) = {idx} (name: {names[idx]})")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)


def main():
    args = _parse_args()

    # determine colors and names for visualization
    colors, names = get_colormap(
        name=args.colormap, n=args.n_colors, return_names=True
    )

    max_len = len(max(names, key=len)) + 1
    cmap_str = '\n'.join(("{: <%ds} {}" % max_len).format(n+':', c)
                         for n, c in zip(names, colors.tolist()))
    print_section("Colormap",
                  f"Using: '{args.colormap}' with {len(colors)} colors:\n"
                  f"{cmap_str}")

    # get filepaths
    path = os.path.abspath(os.path.expanduser(args.path_or_filepath))

    if os.path.isfile(path):
        path = os.path.dirname(args.path_or_filepath)
        extension = os.path.splitext(args.path_or_filepath)[1][1:]
    else:
        extension = args.file_extension

    files = sorted(glob(os.path.join(path, f'*.{extension}')))
    if os.path.isfile(args.path_or_filepath):
        idx = files.index(args.path_or_filepath)
    else:
        idx = 0

    print_section("Files",
                  f"Found {len(files)} '*.{extension}' file(s) in: {path}")

    if len(files) == 0:
        return

    print_section("Usage", USAGE)

    # display image(s)
    if args.auto_size:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    else:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    reload = True
    color_alpha = args.color_alpha
    while True:
        key = cv2.waitKey(100) & 0xFF

        if key in (27, ord('q')):
            # exit (ESC or q)
            break
        elif key in (83, ord('n'), ord('d')):
            # next image (ARROW_RIGHT, n, or d)
            idx = (idx + 1) % len(files)
            reload = True
        elif key in (81, ord('p'), ord('a')):
            # previous image (ARROW_LEFT, p, or a)
            idx = (idx - 1) % len(files)
            reload = True
        elif key in (43, ord('+')):
            # increase color alpha (PLUS)
            color_alpha = min(1.0, color_alpha + 0.05)
            print(f"Changed color alpha to: {color_alpha:0.2f}")
            reload = True
        elif key in (45, ord('-')):
            # decrease color alpha (MINUS)
            color_alpha = max(0.0, color_alpha - 0.05)
            print(f"Changed color alpha to: {color_alpha:0.2f}")
            reload = True

        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) == 0:
            # window closed
            break

        if reload:
            # show image
            _load_and_show(files,
                           idx,
                           cmap=colors,
                           names=names,
                           color_path=args.color_path,
                           color_alpha=color_alpha,
                           rgb_format=args.rgb_format,
                           without_void=args.without_void)
            reload = False

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
