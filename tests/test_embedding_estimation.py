# -*- coding: utf-8 -*-
"""
Simple (interface) tests for embedding estimation

.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import os
import shutil

import cv2
import numpy as np
import pytest

from nicr_scene_analysis_datasets.auxiliary_data.embedding_estimation import get_embedding_estimator_class
from nicr_scene_analysis_datasets.utils.io import download_file


EXAMPLE_IMAGE = 'https://dl.fbaipublicfiles.com/dinov2/images/example.jpg'

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


@pytest.mark.parametrize(
    'estimator__model', (
        'alpha_clip__b16-grit-1m',
        'alpha_clip__l14-grit-1m',
        'alpha_clip__l14-336-grit-1m',
        'alpha_clip__b16-grit-20m',
        'alpha_clip__l14-grit-20m',
        'alpha_clip__l14-336-grit-20m',
        'alpha_clip__b16-combined',
        'alpha_clip__l14-combined',
    )
)
def test_embedding_estimator(estimator__model, tmp_path):
    # Get example image
    img = _get_example_img(tmp_path)

    # Initialize the embedding estimator
    Estimator = get_embedding_estimator_class(estimator__model)
    estimator = Estimator(
        device='cpu',
        auto_set_up=True,
        cache_basepath=tmp_path if not USE_DEFAULT_CACHE_PATH else None,
    )

    # Generate mask for the whole image, same size as input
    mask = np.ones_like(img, dtype=np.uint8)
    #  The mask should only have one channel
    mask = mask[:, :, 0][:, :, None]

    # Get embeddings
    embeddings = estimator.predict(img, mask)

    # Basic assertions
    assert embeddings is not None
    assert isinstance(embeddings, np.ndarray)

    # Embedding should have batch dimension and embedding dimension
    assert embeddings.ndim == 2

    # We only have on input, so the batch dimension should be 1
    assert embeddings.shape[0] == 1
