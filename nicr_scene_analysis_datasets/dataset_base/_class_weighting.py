# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import numpy as np
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


KNOWN_CLASS_WEIGHTINGS = (
    'median-frequency',     # median frequency balancing
    'logarithmic',     # logarithmic weighting with: 1 / ln(c+p_class)
    'linear',    # 1 - p_class
    'none'    # no weighting (ones for all classes)
)


def compute_class_weights(
    dataset,
    sample_key,
    n_classes,
    ignore_first_class: bool = True,    # ignore void class
    weight_mode: str = 'median-frequency',
    c: float = 1.02,
    n_threads: int = 1,
    debug: bool = False,
    verbose: bool = True
) -> np.array:
    assert weight_mode in KNOWN_CLASS_WEIGHTINGS

    if verbose:
        print_ = print
    else:
        def print_(*args, **kwargs):
            pass

    if debug:
        warnings.warn(
            "Weight mode 'none' is forced as debug mode is enabled, i.e., "
            "ones are used as class weights."
        )
        weight_mode = 'none'

    print_(f"Computing class weights for '{sample_key}' ...")

    if 'none' == weight_mode:
        # equal weights for all classes -> disables class weighting
        if ignore_first_class:
            return np.ones(n_classes-1)
        else:
            return np.ones(n_classes)

    def count_helper(sample_idx):
        data = dataset.load(sample_key, sample_idx)
        h, w = data.shape
        n_pixels_per_class_sample = np.bincount(
            data.flatten(),
            minlength=n_classes
        )

        # for median frequency, we need the pixel sum of the images where
        # the specific class is present. (it only matters if the class is
        # present in the image and not how many pixels it occupies.)
        class_in_image = n_pixels_per_class_sample > 0
        n_image_pixels_with_class_sample = class_in_image * h * w

        return n_pixels_per_class_sample, n_image_pixels_with_class_sample

    n_pixels_per_class = np.zeros(n_classes, dtype=np.int64)
    n_image_pixels_with_class = np.zeros(n_classes, dtype=np.int64)

    if n_threads == 1:
        for i in tqdm(range(len(dataset)),
                      total=len(dataset),
                      disable=not verbose):
            # process current image at index i
            cur_n_pixels_per_class, cur_n_image_pixels_with_class = \
                count_helper(i)

            # update stats
            n_pixels_per_class += cur_n_pixels_per_class
            n_image_pixels_with_class += cur_n_image_pixels_with_class
    else:
        # process images using multiple threads
        res = thread_map(count_helper, range(len(dataset)),
                         total=len(dataset),
                         max_workers=n_threads,
                         disable=not verbose)
        # update stats
        for cur_n_pixels_per_class, cur_n_image_pixels_with_class in res:
            n_pixels_per_class += cur_n_pixels_per_class
            n_image_pixels_with_class += cur_n_image_pixels_with_class

    # remove first class (void)
    if ignore_first_class:
        n_pixels_per_class = n_pixels_per_class[1:]
        n_image_pixels_with_class = n_image_pixels_with_class[1:]

    if weight_mode == 'linear':
        probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
        class_weights = 1 - probabilities

    elif weight_mode == 'median-frequency':
        frequency = n_pixels_per_class / n_image_pixels_with_class
        class_weights = np.nanmedian(frequency) / frequency

    elif weight_mode == 'logarithmic':
        probabilities = n_pixels_per_class / np.sum(n_pixels_per_class)
        class_weights = 1 / np.log(c + probabilities)

    nan_indices = np.argwhere(np.isnan(class_weights))
    if len(nan_indices) != 0:
        print_(f"class_weights:\n{class_weights}")
        warnings.warn(
            f"Classweights contain NaNs at positions: {nan_indices}, "
            "setting NaNs to 0."
        )
        print_(f"n_pixels_per_class:\n{n_pixels_per_class}")
        print_(f"n_image_pixels_with_class:\n{n_image_pixels_with_class}")
        class_weights[nan_indices] = 0
        print_(f"fixed class_weights:\n{class_weights}")

    return class_weights
