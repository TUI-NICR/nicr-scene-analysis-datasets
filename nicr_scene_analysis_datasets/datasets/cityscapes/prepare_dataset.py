# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de
"""
import argparse as ap
import json
import os
import shutil

import numpy as np
from tqdm import tqdm
import cv2

from .cityscapes import CityscapesMeta
from ...utils.img import save_indexed_png
from ...utils.io import create_or_update_creation_metafile
from ...utils.io import get_files_by_extension


RGB_DIR = 'leftImg8bit'
PARAMETERS_RAW_DIR = 'camera'
DISPARITY_RAW_DIR = 'disparity'
LABEL_DIR = 'gtFine'


def main():
    # argument parser
    parser = ap.ArgumentParser(description='Prepare Cityscapes dataset.')
    parser.add_argument('output_path', type=str,
                        help="Path where to store dataset.")
    parser.add_argument('cityscapes_filepath', type=str,
                        help="Filepath to downloaded (and uncompressed) "
                             "Cityscapes files.")
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)

    # create output path if not exist
    os.makedirs(output_path, exist_ok=True)

    # write meta file
    create_or_update_creation_metafile(output_path)

    def get_filepaths(path, extension):
        # skip folders such as 'demoVideo'
        subfolders = ['train', 'val', 'test']

        filepaths = []
        for f in subfolders:
            filepaths.extend(get_files_by_extension(os.path.join(path, f),
                                                    extension=extension,
                                                    flat_structure=True,
                                                    recursive=True))
        return filepaths

    rgb_filepaths = get_filepaths(
        os.path.join(args.cityscapes_filepath, RGB_DIR),
        extension='.png',
    )

    label_filepaths = get_filepaths(
        os.path.join(args.cityscapes_filepath, LABEL_DIR),
        extension='.png',
    )
    label_filepaths = [fp for fp in label_filepaths
                       if os.path.basename(fp).find('labelIds') > -1]

    disparity_raw_filepaths = get_filepaths(
        os.path.join(args.cityscapes_filepath, DISPARITY_RAW_DIR),
        extension='.png',
    )

    parameters_filepaths = get_filepaths(
        os.path.join(args.cityscapes_filepath, PARAMETERS_RAW_DIR),
        extension='.json',
    )

    # check for consistency
    assert all(len(path_list) == 5000 for path_list in [rgb_filepaths,
                                                        label_filepaths,
                                                        disparity_raw_filepaths,
                                                        parameters_filepaths])

    def get_basename(fp):
        # e.g. berlin_000000_000019_camera.json -> berlin_000000_000019
        return '_'.join(os.path.basename(fp).split('_')[:3])

    basenames = [get_basename(f) for f in rgb_filepaths]
    for li in [label_filepaths, disparity_raw_filepaths, parameters_filepaths]:
        assert basenames == [get_basename(f) for f in li]

    filelists = {s: {'rgb': [],
                     'depth_raw': [],
                     'disparity_raw': [],
                     'labels_33': [],
                     'labels_19': []}
                 for s in CityscapesMeta.SPLITS}

    # copy rgb images
    print("Copying rgb files")
    for rgb_fp in tqdm(rgb_filepaths):
        basename = os.path.basename(rgb_fp)
        city = os.path.basename(os.path.dirname(rgb_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(rgb_fp)))
        subset = 'valid' if subset == 'val' else subset

        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.RGB_DIR, city)
        os.makedirs(dest_path, exist_ok=True)

        # print(rgb_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(rgb_fp, os.path.join(dest_path, basename))
        filelists[subset]['rgb'].append(os.path.join(city, basename))

    # copy depth images
    print("Copying disparity files and creating depth files")
    for d_fp, p_fp in tqdm(zip(disparity_raw_filepaths,
                               parameters_filepaths),
                           total=len(disparity_raw_filepaths)):
        basename = os.path.basename(d_fp)
        city = os.path.basename(os.path.dirname(d_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(d_fp)))
        subset = 'valid' if subset == 'val' else subset

        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.DISPARITY_RAW_DIR, city)
        os.makedirs(dest_path, exist_ok=True)

        # print(d_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(d_fp, os.path.join(dest_path, basename))
        filelists[subset]['disparity_raw'].append(os.path.join(city, basename))

        # load disparity file and camera parameters
        disp = cv2.imread(d_fp, cv2.IMREAD_UNCHANGED)
        with open(p_fp, 'r') as f:
            camera_parameters = json.load(f)
        baseline = camera_parameters['extrinsic']['baseline']
        fx = camera_parameters['intrinsic']['fx']

        # convert disparity to depth (im m?)
        # see: https://github.com/mcordts/cityscapesScripts/issues/55#issuecomment-411486510
        disp_mask = disp > 0
        depth = disp.astype('float32')
        depth[disp_mask] = (depth[disp_mask] - 1) / 256
        disp_mask = depth > 0    # avoid divide by zero
        depth[disp_mask] = (baseline * fx) / depth[disp_mask]

        # cast to float16
        depth = depth.astype('float16')

        # save depth image
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.DEPTH_RAW_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        depth_basename = basename.replace('.png', '.npy')
        depth_basename = depth_basename.replace('disparity', 'depth')
        np.save(os.path.join(dest_path, depth_basename), depth)
        filelists[subset]['depth_raw'].append(os.path.join(city,
                                                           depth_basename))

    print("Processing label files")
    mapping_1plus33_to_1plus19 = np.array(
        [CityscapesMeta.SEMANTIC_CLASS_MAPPING_REDUCED[i]
         for i in range(1+33)], dtype='uint8'
    )

    for l_fp in tqdm(label_filepaths):
        basename = os.path.basename(l_fp)
        city = os.path.basename(os.path.dirname(l_fp))
        subset = os.path.basename(os.path.dirname(os.path.dirname(l_fp)))
        subset = 'valid' if subset == 'val' else subset

        # load label with 1+33 classes
        label_full = cv2.imread(l_fp, cv2.IMREAD_UNCHANGED)

        # full: 1+33 classes (original label file -> just copy file)
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.SEMANTIC_FULL_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        # print(l_fp, '->', os.path.join(dest_path, basename))
        shutil.copy(l_fp, os.path.join(dest_path, basename))
        filelists[subset]['labels_33'].append(os.path.join(city, basename))

        # full: 1+33 classes colored
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.SEMANTIC_FULL_COLORED_DIR,
                                 city)
        os.makedirs(dest_path, exist_ok=True)
        save_indexed_png(os.path.join(dest_path, basename), label_full,
                         colormap=CityscapesMeta.SEMANTIC_LABEL_LIST_FULL.colors_array)

        # map full to reduced: 1+33 classes -> 1+19 classes
        label_reduced = mapping_1plus33_to_1plus19[label_full]

        # reduced: 1+19 classes
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.SEMANTIC_REDUCED_DIR, city)
        os.makedirs(dest_path, exist_ok=True)
        cv2.imwrite(os.path.join(dest_path, basename), label_reduced)
        filelists[subset]['labels_19'].append(os.path.join(city, basename))

        # reduced: 1+19 classes colored
        dest_path = os.path.join(args.output_path, subset,
                                 CityscapesMeta.SEMANTIC_REDUCED_COLORED_DIR,
                                 city)
        os.makedirs(dest_path, exist_ok=True)
        save_indexed_png(os.path.join(dest_path, basename), label_reduced,
                         colormap=CityscapesMeta.SEMANTIC_LABEL_LIST_REDUCED.colors_array)

    # ensure that filelists are valid and faultless
    def get_identifier(filepath):
        return '_'.join(filepath.split('_')[:3])

    n_samples = 0
    for subset in CityscapesMeta.SPLITS:
        identifier_lists = []
        for filelist in filelists[subset].values():
            identifier_lists.append([get_identifier(fp) for fp in filelist])

        assert all(li == identifier_lists[0] for li in identifier_lists[1:])
        n_samples += len(identifier_lists[0])

    assert n_samples == 5000

    # save meta files
    print("Writing meta files")
    np.savetxt(os.path.join(output_path, 'class_names_1+33.txt'),
               CityscapesMeta.SEMANTIC_LABEL_LIST_FULL.class_names,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+33.txt'),
               CityscapesMeta.SEMANTIC_LABEL_LIST_FULL.colors_array,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+19.txt'),
               CityscapesMeta.SEMANTIC_LABEL_LIST_REDUCED.class_names,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+19.txt'),
               CityscapesMeta.SEMANTIC_LABEL_LIST_REDUCED.colors_array,
               delimiter=',', fmt='%s')

    for subset in CityscapesMeta.SPLITS:
        subset_dict = filelists[subset]
        for key, filelist in subset_dict.items():
            np.savetxt(os.path.join(output_path, f'{subset}_{key}.txt'),
                       filelist,
                       delimiter=',', fmt='%s')


if __name__ == '__main__':
    main()
