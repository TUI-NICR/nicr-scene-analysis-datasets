# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import os

import numpy as np
import cv2
import shutil

from ..nyuv2.dataset import NYUv2
from .dataset import SUNRGBD


class NYUv2InstancesMatcher:
    def __init__(self, sunrgbd_path, nyuv2_path):
        self.sunrgbd_path = sunrgbd_path
        self.sun_cam = "kv1"
        self.nyuv2_path = nyuv2_path
        self.splits = ["train", "test"]
        # Images in SUNRGBD contains cropped images of NYUv2. SUNRGBD does not
        # seem to contain the cropping parameters, so we estimated them with the
        # calc_crop function. Cropping parameters seemed to be constants, so
        # we hardcoded them.
        self.crop = (44, 40, 471, 601)

    def get_sunrgbd_path_for_nyuv2(self, split, folder):
        path = os.path.join(self.sunrgbd_path,
                            split,
                            folder,
                            'kv1',
                            'NYUdata')
        return path

    def get_nyuv2_path(self, split, folder):
        # Map NYUv2 samples in SUNRGBD to index
        path = os.path.join(self.nyuv2_path, split, folder)
        return path

    def do_matching(self, verbose=False, dry_run=False):
        # note that the semantic labels in SUNRGBD and NYUv2 are not completely
        # equal, besides the last three other* classes in NYUv2, there are
        # classes (e.g. 'shower curtain', 'floor mat'), which are labeled in
        # NYUv2 but are assigned to void in SUNRGBD, thus, not all instances
        # can be matched, nevertheless, we copy the instance ids without
        # considering the semantic label, which means that we also assign
        # instance ids to void labels !!!
        def debug_print(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        for split in self.splits:
            debug_print(f"Processing split: {split}")

            # note that sorting the content in both directories is already
            # enough to get a valid matching -> so do not change the filenames
            nyuv2_image_dir = self.get_nyuv2_path(split, 'rgb')
            nyuv2_image_content = sorted(os.listdir(nyuv2_image_dir))

            sun_image_dir = self.get_sunrgbd_path_for_nyuv2(split, 'rgb')
            sun_image_content = sorted(os.listdir(sun_image_dir))

            nyuv2_instance_dir = self.get_nyuv2_path(split, 'instance')
            nyuv2_instances = sorted(os.listdir(nyuv2_instance_dir))

            sun_instance_dir = self.get_sunrgbd_path_for_nyuv2(split,
                                                               'instance')
            sun_instances = sorted(os.listdir(sun_instance_dir))

            nyuv2_orientation_dir = self.get_nyuv2_path(split, 'orientations')
            nyuv2_orientations = sorted(os.listdir(nyuv2_orientation_dir))

            sun_orientation_dir = self.get_sunrgbd_path_for_nyuv2(
                split, 'orientations'
            )
            sun_orientations = sorted(os.listdir(sun_orientation_dir))

            for idx, (sun_fn, nyuv2_fn) in enumerate(zip(sun_image_content,
                                                         nyuv2_image_content)):
                debug_print(f"Matching instances: nyuv2: {nyuv2_fn} -> "
                            f"sunrgbd: {sun_fn}")

                sun_instance = cv2.imread(os.path.join(sun_instance_dir,
                                                       sun_instances[idx]),
                                          cv2.IMREAD_UNCHANGED)

                nyuv2_instance = cv2.imread(os.path.join(nyuv2_instance_dir,
                                                         nyuv2_instances[idx]),
                                            cv2.IMREAD_UNCHANGED)
                # Indexing is to keep datatype
                sun_instance[:, :] = nyuv2_instance[self.crop[0]:self.crop[2],
                                                    self.crop[1]:self.crop[3]]
                if dry_run:
                    debug_print("-> dry run enabled: nothing matched")
                    # stop here, do not adapt data
                    continue

                cv2.imwrite(os.path.join(sun_instance_dir,
                                         sun_instances[idx]),
                            sun_instance)

                shutil.copy(os.path.join(nyuv2_orientation_dir,
                                         nyuv2_orientations[idx]),
                            os.path.join(sun_orientation_dir,
                                         sun_orientations[idx]))

    def calc_crop(self, nyuv2_rgb, sun_rgb):
        n_h, n_w = nyuv2_rgb.shape[:2]
        s_h, s_w = sun_rgb.shape[:2]
        t_h = n_h - s_h
        t_w = n_w - s_w

        min_err = np.infty
        b_h = 0
        b_w = 0
        for c_h in range(t_h):
            for c_w in range(t_w):
                nyuv_patch = nyuv2_rgb[c_h:c_h+s_h, c_w:c_w+s_w]
                err = np.abs(nyuv_patch-sun_rgb).sum()
                if err < min_err:
                    min_err = err
                    b_h = c_h
                    b_w = c_w
        print(min_err)
        return b_h, b_w, b_h+s_h, b_w+s_w


if __name__ == '__main__':
    # python -m nicr_scene_analysis_datasets.datasets.sunrgbd.match_nyuv2_instances \
    #     path/to/sunrgbd \
    #     path/to/nyuv2/ \
    #     [--no-dry-run]

    # argument parser
    parser = ap.ArgumentParser()
    parser.add_argument('sunrgbd_path',
                        type=str,
                        help="Path to SUNRGBD dataset.")
    parser.add_argument('nyuv2_path',
                        type=str,
                        help="Path to NYUv2 dataset.")
    parser.add_argument('--no-dry-run',
                        default=False,
                        action='store_true',
                        help="Adapt data. Use this argument with caution.")
    args = parser.parse_args()
    print(args)

    # run matching
    matcher = NYUv2InstancesMatcher(
        sunrgbd_path=args.sunrgbd_path,
        nyuv2_path=args.nyuv2_path
    )

    matcher.do_matching(verbose=True, dry_run=not args.no_dry_run)
