# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
"""
import argparse as ap
from glob import glob
import inspect
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ..utils.img import blend_images
from .common import print_section


WINDOW_NAME = 'Depth Viewer'

USAGE = inspect.cleandoc(
    """
    ARROW_RIGHT / n / d: next image
    ARROW_LEFT / p / a:  previous image
    +:                   increase alpha for blending (see `--color-path`)
    -:                   decrease alpha for blending (see `--color-path`)
    CTRL + s:            save current image (save screenshot at current scale)
    q:                   quit
    CLICK ON IMAGE:      show depth value at cursor position
    """
)


def _parse_args():
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Viewer for depth images."
    )

    parser.add_argument(
        'path_or_filepath',
        type=str,
        help="Path to prediction image(s) to load."
    )
    parser.add_argument(
        '--file-extension',
        type=str,
        default='png',
        help="File extension for images to search for."
    )
    parser.add_argument(
        '-m', '--mode',
        type=str,
        choices=('image', 'image_nonzero', 'limits'),
        default='limits',
        help="Mode for scaling depth values before applying colormap."
    )
    parser.add_argument(
        '-l', '--min-depth',
        type=float,
        default=0,
        help="Lower limit for scaling depth values, if `scale-mode` is "
             "'limits'."
    )
    parser.add_argument(
        '-u', '--max-depth',
        type=float,
        default=6000,
        help="Upper limit for scaling depth values, if `scale-mode` is "
        "'limits'."
    )
    parser.add_argument(
        '-c', '--colormap',
        type=str,
        default='gray',
        choices=tuple(plt.colormaps()),
        help="Colormap to use for visualization"
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
                   color_path=None,
                   color_alpha=0.5,
                   scale_mode='limits',
                   min_depth=0,
                   max_depth=10000,
                   colormap='gray'):

    # load image
    filepath = filepaths[idx]
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    img_raw = img.copy()
    img = img.astype(np.float32)
    img_min = img.min()
    img_min_nonzero = img[img > 0].min()
    img_max = img.max()
    img_mean = img.mean()
    img_std = img.std()

    print(f"[{idx}/{len(filepaths)-1}] {os.path.basename(filepath)} "
          f"(WxH: {img.shape[1]}x{img.shape[0]}, "
          f"Min: {img_min:0.2f}, Min (nonzero): {img_min_nonzero:0.2f}, "
          f"Max: {img_max:0.2f}, "
          f"Mean: {img_mean:0.2f}, "
          f"Std: {img_std:0.2f})")

    if img.ndim != 2:
        print(f"Unknown shape: {img.shape}")

    # scale depth values
    if 'limits' == scale_mode:
        img -= min_depth
        img /= max_depth - min_depth
        # ensure [0, 1] range
        img = np.clip(img, 0, 1)
    elif 'image_nonzero' == scale_mode:
        img -= img_min_nonzero
        img /= img_max - img_min_nonzero
        # ensure [0, 1] range
        img = np.clip(img, 0, 1)    # set invalid pixels back to zero
    else:
        img -= img_min
        img /= img_max - img_min

    # get color map from matplotlib
    cmap = plt.get_cmap(colormap)
    img_colored = cmap(img, bytes=True)
    img_colored_bgr = cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)

    # reset invalid pixels back to zero
    img_colored_bgr[img == 0] = (0, 0, 0)

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

        # overlay color image with depth image
        img_colored_bgr = blend_images(
            color_img,
            img_colored_bgr,
            alpha=color_alpha
        )

    # show image
    cv2.imshow(WINDOW_NAME, img_colored_bgr)

    def mouse_callback(event, x, y, flags, params):
        # 0: mouse move, 1: left button down, 2: right button down
        if 1 == event:
            depth_value = img_raw[y, x]
            print(f"(x={x}, y={y}) = {depth_value}")

    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)


def main():
    # parse args
    args = _parse_args()

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
        key = cv2.waitKey(10) & 0xFF

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
                           color_path=args.color_path,
                           color_alpha=color_alpha,
                           scale_mode=args.mode,
                           min_depth=args.min_depth,
                           max_depth=args.max_depth,
                           colormap=args.colormap)
            reload = False

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
