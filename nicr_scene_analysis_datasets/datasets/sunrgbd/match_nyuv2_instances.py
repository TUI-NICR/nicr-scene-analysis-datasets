# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os

import numpy as np
import cv2
import shutil

from ..nyuv2.dataset import NYUv2
from .dataset import SUNRGBD


class NYUv2InstancesMatcher:
    def __init__(self, sun_rgbd_path, nyuv2_path):
        self.sun_rgbd_path = sun_rgbd_path
        self.sun_cam = "kv1"
        self.nyuv2_path = nyuv2_path
        self.splits = ["train", "test"]
        # Images in SUNRGBD contains cropped images of NYUv2. SUNRGBD does not
        # seem to contain the cropping parameters, so we estimated them with the
        # calc_crop function. Cropping parameters seemed to be constants, so
        # we hardcoded them.
        self.crop = (44, 40, 471, 601)

    def get_sun_dir_nyuv2(self, split, folder):
        dir = os.path.join(self.sun_rgbd_path,
                           split,
                           folder,
                           'kv1',
                           'NYUdata')
        return dir

    def get_nyuv2_dir(self, split, folder):
        # Map NYUv2 samples in SUNRGBD to index
        nyuv2_image_dir = os.path.join(self.nyuv2_path,
                                       split,
                                       folder)
        return nyuv2_image_dir

    def do_matching(self):
        for split in self.splits:
            nyuv2_image_dir = self.get_nyuv2_dir(split, "rgb")
            nyuv2_image_content = sorted(os.listdir(nyuv2_image_dir))

            sun_image_dir = self.get_sun_dir_nyuv2(split, "rgb")
            sun_image_content = sorted(os.listdir(sun_image_dir))

            nyuv2_instance_dir = self.get_nyuv2_dir(split, "instance")
            nyuv2_instances = sorted(os.listdir(nyuv2_instance_dir))

            sun_instance_dir = self.get_sun_dir_nyuv2(split, "instance")
            sun_instances = sorted(os.listdir(sun_instance_dir))

            nyuv2_orientation_dir = self.get_nyuv2_dir(split, "orientations")
            nyuv2_orientations = sorted(os.listdir(nyuv2_orientation_dir))

            sun_orientation_dir = self.get_sun_dir_nyuv2(split, "orientations")
            sun_orientations = sorted(os.listdir(sun_orientation_dir))

            for idx, (_, _) in enumerate(
                    zip(sun_image_content, nyuv2_image_content)):

                sun_instance = cv2.imread(os.path.join(sun_instance_dir,
                                                       sun_instances[idx]),
                                          cv2.IMREAD_UNCHANGED)

                nyuv2_instance = cv2.imread(os.path.join(nyuv2_instance_dir,
                                                         nyuv2_instances[idx]),
                                            cv2.IMREAD_UNCHANGED)
                # Indexing is to keep datatype
                sun_instance[:, :] = nyuv2_instance[self.crop[0]:self.crop[2],
                                                    self.crop[1]:self.crop[3]]
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
