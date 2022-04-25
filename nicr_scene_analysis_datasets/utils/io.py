# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
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


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def extract_zip(zip_filepath, output_dirpath):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_file:
        for m in tqdm(zip_file.infolist(), desc='Extracting'):
            zip_file.extract(m, output_dirpath)


def download_file(url, output_filepath, display_progressbar=False):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1],
                             disable=not display_progressbar) as t:
        urllib.request.urlretrieve(url,
                                   filename=output_filepath,
                                   reporthook=t.update_to)


def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def get_files_by_extension(path,
                           extension='.png',
                           flat_structure=False,
                           recursive=False,
                           follow_links=True):
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


def create_or_update_creation_metafile(dataset_basepath):
    filepath = os.path.join(dataset_basepath, 'creation_meta.json')

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
        'version': '{}.{}.{}-{}'.format(*get_version(with_suffix=True))
    })
    with open(filepath, 'w') as f:
        json.dump(meta, f, indent=4)
