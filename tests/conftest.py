# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import os
import shutil

import pytest


def pytest_addoption(parser):
    parser.addoption('--keep-files', action='store_true')


@pytest.fixture(scope='session')
def keep_files(request):
    return request.config.getoption('--keep-files')


@pytest.fixture(scope='session')
def tmp_path(tmpdir_factory, keep_files):
    # see: https://docs.pytest.org/en/6.2.x/reference.html#tmpdir-factory
    # use '--basetemp' to change default path
    # -> BE AWARE <- --basetemp is cleared on start !!!

    path = tmpdir_factory.mktemp('nicr_scene_analysis_datasets')
    print(f"\nWriting temporary files to '{path}'")
    if keep_files:
        print("Files are kept and require to be deleted manually!")

    yield path

    # teardown (delete if it was created)
    if os.path.exists(path) and not keep_files:
        shutil.rmtree(path)
