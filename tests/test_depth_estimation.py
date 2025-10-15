# -*- coding: utf-8 -*-
"""
Simple (interface) tests for depth estimation

.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import shutil

import cv2
import pytest

from nicr_scene_analysis_datasets.auxiliary_data.depth_estimation import get_depth_estimator_class
from nicr_scene_analysis_datasets.utils.io import download_file
from nicr_scene_analysis_datasets.scripts import viewer_depth


# EXAMPLE_IMAGE = 'http://images.cocodataset.org/val2017/000000039769.jpg'
EXAMPLE_IMAGE = 'https://dl.fbaipublicfiles.com/dinov2/images/example.jpg'
# EXAMPLE_IMAGE = '/local/dase6070/datasets/ade20k/tmp/ADEChallengeData2016/images/training/ADE_train_00006921.jpg'  # 2100x2100 image
SHOW_RESULTS = False

# true: always use the same default transformers path and, thus, speed up
# consecutive test runs
USE_DEFAULT_CACHE_PATH = True


def _get_example_img(tmp_path):
    fn = 'example.jpg'
    fp = os.path.join(tmp_path, fn)
    if not os.path.exists(fp):
        if EXAMPLE_IMAGE.startswith('http'):
            download_file(EXAMPLE_IMAGE, fp)
        else:
            shutil.copy(EXAMPLE_IMAGE, fp)

    img = cv2.imread(fp, cv2.IMREAD_UNCHANGED)
    assert img is not None
    assert img.ndim == 3

    return img


def _show_result(img, tmp_path):
    if not SHOW_RESULTS:
        return

    # dump file and show
    output_path = os.path.join(tmp_path, 'prediction')
    os.makedirs(output_path, exist_ok=True)
    cv2.imwrite(os.path.join(output_path, 'example.png'), img)
    args = [
        output_path,
        '--color-path', str(tmp_path),
        '--color-alpha', '0.9',
    ]
    viewer_depth.main(args)


@pytest.mark.parametrize(
    'estimator__model', (
        'depthanything_v2__indoor_small',
        'depthanything_v2__indoor_base',
        'depthanything_v2__indoor_large',
        'depthanything_v2__outdoor_small',
        'depthanything_v2__outdoor_base',
        'depthanything_v2__outdoor_large',
        'zoedepth__indoor',
        'zoedepth__outdoor',
        'zoedepth__indoor_outdoor',
        'dino_v2_dpt__indoor_small',
        'dino_v2_dpt__indoor_base',
        'dino_v2_dpt__indoor_large',
        'dino_v2_dpt__indoor_giant',
        'dino_v2_dpt__outdoor_small',
        'dino_v2_dpt__outdoor_base',
        'dino_v2_dpt__outdoor_large',
        'dino_v2_dpt__outdoor_giant',
    )
)
@pytest.mark.parametrize('max_pixels', (1920 * 1080, None))
def test_depth_estimator(estimator__model, max_pixels, tmp_path):
    # get image
    img = _get_example_img(tmp_path)

    # get model
    Estimator = get_depth_estimator_class(estimator__model)
    estimator = Estimator(
        device='cpu',
        max_pixels=max_pixels,
        auto_set_up=True,
        cache_basepath=tmp_path if not USE_DEFAULT_CACHE_PATH else None,
    )

    # predict
    depth = estimator.predict(img)

    # perform some basic tests
    assert depth.ndim == 2
    assert depth.shape == img.shape[:2]

    # optional: show result
    _show_result(depth, tmp_path)
