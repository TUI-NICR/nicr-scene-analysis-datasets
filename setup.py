# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

"""
import os
from setuptools import find_packages
from setuptools import setup
import sys


def run_setup():
    # get version
    version_namespace = {}
    version_fp = os.path.join('nicr_scene_analysis_datasets', 'version.py')
    with open(version_fp) as version_file:
        exec(version_file.read(), version_namespace)
    version = version_namespace['get_version'](with_suffix=False)

    requirements_general = [
        'cityscapesScripts==1.5.0',
        'numpy',
        'pillow',
        'scipy',
        'tqdm>=4.42.0',
    ]

    # OpenCV might be installed using another name
    try:
        import cv2
    except ImportError:
        requirements_general.append('opencv-python')

    if sys.version_info <= (3, 7):
        # python 3.6 does not support dataclasses
        requirements_general.append('dataclasses')

    requirements_prepare = [
        'h5py',
        'numba',
        'pandas',
        'panopticapi @ git+https://github.com/cocodataset/panopticapi.git',
        'protobuf',
        'termcolor',
    ]

    requirements_3d = [
        'open3d',
        'plyfile',
    ]

    # setup
    setup(name='nicr_scene_analysis_datasets',
          version='{}.{}.{}'.format(*version),
          description='Package to prepare and use common datasets for scene '
                      'analysis.',
          author='Daniel Seichter, Soehnke Fischedick',
          author_email='daniel.seichter@tu-ilmenau.de, '
                       'soehnke-benedikt.fischedick@tu-ilmenau.de',
          license='Copyright 2020-2023, Neuroinformatics and Cognitive Robotics'
                  'Lab TU Ilmenau, Ilmenau, Germany',
          packages=find_packages(),
          install_requires=requirements_general,
          extras_require={
              'withpreparation': requirements_prepare,
              'with3d': requirements_3d,
              'test': [
                  'pytest>=3.0.2',
                  'torch'    # should be installed using conda
              ]
          },
          entry_points={
              'console_scripts': [
                  'nicr_sa_prepare_dataset=nicr_scene_analysis_datasets.scripts.prepare_dataset:main',
                  'nicr_sa_prepare_labeled_point_clouds=nicr_scene_analysis_datasets.scripts.prepare_labeled_point_clouds:main',
                  'nicr_sa_depth_viewer=nicr_scene_analysis_datasets.scripts.viewer_depth:main',
                  'nicr_sa_semantic_instance_viewer=nicr_scene_analysis_datasets.scripts.viewer_semantic_instance:main',
                  'nicr_sa_labeled_pc_viewer=nicr_scene_analysis_datasets.scripts.viewer_labeled_point_cloud:main'
              ],
          },
          include_package_data=True)


if __name__ == '__main__':
    run_setup()
