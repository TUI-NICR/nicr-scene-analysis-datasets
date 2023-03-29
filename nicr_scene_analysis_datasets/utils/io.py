# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Any, Dict, List, Union

from collections import OrderedDict
from datetime import datetime
import getpass
import json
import os
import sys
from time import time
import urllib.request
import zipfile

from tqdm import tqdm


from ..version import get_version


CREATION_META_FILENAME = 'creation_meta.json'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def extract_zip(zip_filepath: str, output_dirpath: str) -> None:
    with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
        for m in tqdm(zip_file.infolist(), desc='Extracting'):
            zip_file.extract(m, output_dirpath)


def download_file(
    url: str,
    output_filepath: str,
    display_progressbar: bool = False
) -> None:
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)


def create_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_files_by_extension(
    path: str,
    extension: str = '.png',
    flat_structure: bool = False,
    recursive: bool = False,
    follow_links: bool = True
) -> Union[List, Dict]:
    # check input args
    if not os.path.exists(path):
        raise IOError("No such file or directory: '{}'".format(path))

    if flat_structure:
        filelist = []
    else:
        filelist = {}

    # path is a file
    if os.path.isfile(path):
        basename = os.path.basename(path)
        if extension is None or basename.lower().endswith(extension):
            if flat_structure:
                filelist.append(path)
            else:
                filelist[os.path.dirname(path)] = [basename]
        return filelist

    # get filelist
    filter_func = lambda f: extension is None or f.lower().endswith(extension)
    for root, _, filenames in os.walk(path, topdown=True,
                                      followlinks=follow_links):
        filenames = list(filter(filter_func, filenames))
        if filenames:
            if flat_structure:
                filelist.extend((os.path.join(root, f) for f in filenames))
            else:
                filelist[root] = sorted(filenames)
        if not recursive:
            break

    # return
    if flat_structure:
        return sorted(filelist)
    else:
        return OrderedDict(sorted(filelist.items()))


def create_or_update_creation_metafile(
    dataset_basepath: str,
    **additional_meta
) -> None:
    filepath = os.path.join(dataset_basepath, CREATION_META_FILENAME)

    # load existing file
    if os.path.exists(filepath):
        with open(filepath) as f:
            meta = json.load(f)
    else:
        meta = []

    # update file
    ts = time()
    meta.append({
        'command': ' '.join(sys.argv),
        'timestamp': int(ts),
        'local_time': datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
        'user': getpass.getuser(),
        'version': '{}.{}.{}-{}'.format(*get_version(with_suffix=True)),
        'additional_meta': additional_meta or None
    })
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)


def load_creation_metafile(dataset_basepath: str) -> Dict[str, Any]:
    filepath = os.path.join(dataset_basepath, CREATION_META_FILENAME)
    with open(filepath) as f:
        return json.load(f)
