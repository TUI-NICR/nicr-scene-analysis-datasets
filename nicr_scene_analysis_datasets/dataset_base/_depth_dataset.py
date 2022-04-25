# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Dict, Tuple

import abc
from dataclasses import asdict
from functools import partial
import warnings

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from ._annotation import IntrinsicCameraParametersNormalized
from ._base_dataset import DatasetBase


class DepthDataset(DatasetBase):
    def __init__(
        self,
        depth_mode: str = 'raw',
        sample_keys: Tuple[str] = ('depth', 'semantic'),
        use_cache: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            sample_keys=sample_keys,
            use_cache=use_cache,
            **kwargs
        )

        self._depth_mode = depth_mode

    @abc.abstractmethod
    def _load_depth(self, idx) -> np.array:
        pass

    def _load_depth_intrinsics(self, idx) -> IntrinsicCameraParametersNormalized:
        # so far, only few datasets support intrinsics, thus, we define a
        # default here
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def depth_mode(self) -> str:
        pass

    @property
    def depth_stats(self) -> Dict[str, float]:
        return asdict(self.config.depth_stats)

    @property
    def depth_mean(self) -> float:
        return self.config.depth_stats.mean

    @property
    def depth_std(self) -> float:
        return self.config.depth_stats.std

    @property
    def depth_max(self) -> float:
        return self.config.depth_stats.max

    @property
    def depth_min(self) -> float:
        return self.config.depth_stats.min

    def depth_compute_stats(
        self,
        n_threads: int = 1,
        debug: bool = False
    ) -> Dict[str, float]:
        if debug:
            warnings.warn("Returning default depth stats from <dataset>.py as "
                          "debug mode is enabled")
            return self.depth_stats

        if self._disable_prints:
            def print_(*args, **kwargs):
                pass
        else:
            print_ = print

        pixel_sum = np.float64(0)
        pixel_cnt = np.uint64(0)
        pixel_square_sum = np.float64(0)
        max_depth = np.float64(-np.inf)
        min_depth = np.float64(np.inf)

        print_("Computing mean, std, min, and max for depth images ...")

        def compute_helper1(sample_idx):
            depth = self.load('depth', sample_idx)
            if 'raw' == self.depth_mode:
                # we do not count invalid depth measurements
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()

            if 0 == depth_valid.size:
                # only invalid values
                return None

            max_depth_sample = depth_valid.max()
            min_depth_sample = depth_valid.min()
            pixel_sum_sample = np.sum(depth_valid)
            pixel_cnt_sample = np.uint64(len(depth_valid))

            return (max_depth_sample, min_depth_sample, pixel_sum_sample,
                    pixel_cnt_sample)

        def compute_helper2(sample_idx, mean):
            depth = self.load('depth', sample_idx)
            if 'raw' == self.depth_mode:
                # we do not count invalid depth measurements
                depth_valid = depth[depth > 0]
            else:
                depth_valid = depth.flatten()

            if 0 == depth_valid.size:
                # only invalid values
                return None

            pixel_square_sum_sample = np.sum(np.square(depth_valid - mean))

            return pixel_square_sum_sample

        print_("[1/2] Computing mean, min, and max ...")

        # compute max, min, total sum, number of pixel
        if n_threads == 1:
            for i in tqdm(range(len(self)),
                          total=len(self),
                          disable=self._disable_prints):
                # process current image at index i
                res = compute_helper1(i)
                if res is None:
                    continue

                # update stats
                cur_max_depth, cur_min_depth, cur_pixel_sum, cur_pixel_cnt = res
                max_depth = np.maximum(cur_max_depth, max_depth)
                min_depth = np.minimum(cur_min_depth, min_depth)
                pixel_sum += cur_pixel_sum
                pixel_cnt += cur_pixel_cnt
        else:
            # process images using multiple threads
            results = thread_map(compute_helper1,
                                 range(len(self)),
                                 total=len(self),
                                 max_workers=n_threads,
                                 disable=self._disable_prints)
            # update stats
            for res in results:
                if res is None:
                    continue

                cur_max_depth, cur_min_depth, cur_pixel_sum, cur_pixel_cnt = res
                max_depth = np.maximum(cur_max_depth, max_depth)
                min_depth = np.minimum(cur_min_depth, min_depth)
                pixel_sum += cur_pixel_sum
                pixel_cnt += cur_pixel_cnt

        # compute mean
        mean_depth = pixel_sum / pixel_cnt

        print_("[2/2] Computing std ...")
        # compute total squared sum
        if n_threads == 1:
            for i in tqdm(range(len(self)),
                          total=len(self),
                          disable=self._disable_prints):
                res = compute_helper2(i, mean=mean_depth)
                if res is None:
                    continue

                pixel_square_sum += res
        else:
            # process images using multiple threads
            results = thread_map(partial(compute_helper2, mean=mean_depth),
                                 range(len(self)),
                                 total=len(self),
                                 max_workers=n_threads)
            # update stats
            for res in results:
                if res is None:
                    continue

                pixel_square_sum += res

        # compute std
        std_depth = np.sqrt(pixel_square_sum / pixel_cnt)

        depth_stats = {'mean': mean_depth,
                       'std': std_depth,
                       'min': min_depth,
                       'max': max_depth}

        return depth_stats
