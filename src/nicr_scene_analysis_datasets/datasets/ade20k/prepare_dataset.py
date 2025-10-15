# -*- coding: utf-8 -*-
"""
.. codeauthor:: Robin Schmidt <robin.schmidt@tu-ilmenau.de>
.. codeauthor:: Söhnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import os
import shutil
from functools import partial
from glob import glob
from tarfile import TarFile
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from ...utils.io import download_file
from ._class_mappings import MAPPING_INSTANCE_100_TO_SCENE_PARSE_150
from .ade20k import ADE20KMeta

CHALLENGE_URLS = {
    # 2016 Scene Parse Benchmark Challenge
    'ADEChallengeData2016.zip': 'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',
    # 2017 Places Challenge (instance segmenation part)
    'annotations_instance.tar': 'http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar'
}


COLORS = {
    150: ADE20KMeta.SEMANTIC_LABEL_LIST_CHALLENGE_150.colors_array,
}


def _parse_args(args=None):
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Prepare ADE20K dataset."
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path where to store dataset."
    )
    parser.add_argument(
        '--challenge-2016-filepath',
        type=str,
        default=None,
        help="Path to the '2016 Scene Parse Benchmark Challenge' zip file "
             "(ADEChallengeData2016.zip)."
    )
    parser.add_argument(
        '--challenge-2017-instances-filepath',
        type=str,
        default=None,
        help="Path to the tar file containing the instance annotations of the "
             "'2017 Places Challenge' tar file (annotations_instance.tar)."
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=1,
        help="Number of processes to use."
    )

    return parser.parse_args(args)


def _get_image_size_str(*, w, h):
    return f'{w}x{h}'


def _save_png(fp, img):
    create_dir(os.path.dirname(fp))
    cv2.imwrite(fp, img)


def _challenge_prepare_data(challenge_raw_path, output_path, n_processes=1):
    # parse 'sceneCategories.txt' to get filenames and scene classes ----------
    filenames_scenes = []
    fp = os.path.join(challenge_raw_path, 'sceneCategories.txt')
    with open(fp, 'r') as f:
        # format: space separated filename and scene class, each line one entry
        for line in f:
            fn, sc = line.strip().split(' ')
            filenames_scenes.append((fn, sc))

    assert len(filenames_scenes) == 22210

    # process samples ---------------------------------------------------------
    if 1 == n_processes:
        for fn_sc in tqdm(filenames_scenes):
            _challenge_process_one_sample(
                filename_scene_class=fn_sc,
                challenge_raw_path=challenge_raw_path,
                output_path=output_path
            )
    else:
        fun_call = partial(
            _challenge_process_one_sample,
            challenge_raw_path=challenge_raw_path,
            output_path=output_path
        )
        process_map(fun_call, filenames_scenes, chunksize=20)

    # define splits for symlink and split file generation
    challenge_splits = [
        ADE20KMeta.SPLIT_TRAIN_CHALLENGE_2016,
        ADE20KMeta.SPLIT_VALID_CHALLENGE_2016,
    ]
    panoptic_splits = [
        ADE20KMeta.SPLIT_TRAIN_PANOPTIC_2017,
        ADE20KMeta.SPLIT_VALID_PANOPTIC_2017,
    ]

    # create symlinks for panoptic splits
    for split_challenge, split_panoptic in zip(challenge_splits, panoptic_splits):
        challenge_img_dir = os.path.join(
            output_path, split_challenge, ADE20KMeta.IMAGE_DIR
        )
        panoptic_img_dir = os.path.join(
            output_path, split_panoptic, ADE20KMeta.IMAGE_DIR
        )
        create_dir(os.path.dirname(panoptic_img_dir))
        relative_path = os.path.relpath(
            challenge_img_dir, os.path.dirname(panoptic_img_dir)
        )
        os.symlink(relative_path, panoptic_img_dir)
    # write split files (based on rgb images) ---------------------------------
    n = 0
    for split in challenge_splits:
        rgb_path = os.path.join(output_path, split, ADE20KMeta.IMAGE_DIR)
        filepaths = glob(os.path.join(rgb_path, '**/*.jpg'))
        filepaths = sorted(
            filepaths,
            key=lambda x: os.path.basename(x)  # order by filename, not path
        )

        split_txt_fp = os.path.join(
            output_path, ADE20KMeta.get_split_filelist_filename(split)
        )
        with open(split_txt_fp, 'w') as f:
            for fp in filepaths:
                entry = os.path.splitext(os.path.relpath(fp, rgb_path))[0]
                f.write(f'{entry}\n')
                n += 1

    # split files are the same, as the images are the same
    for split_challenge, split_panoptic in zip(challenge_splits, panoptic_splits):
        src = os.path.join(output_path, ADE20KMeta.get_split_filelist_filename(split_challenge))
        dst = os.path.join(output_path, ADE20KMeta.get_split_filelist_filename(split_panoptic))
        shutil.copy(src, dst)
    # sanity check
    assert n == len(filenames_scenes)    # same files for challenge and panoptic splits


def _challenge_process_one_sample(filename_scene_class,
                                  challenge_raw_path,
                                  output_path):
    fn, sc = filename_scene_class    # unpack tuple

    # determine split and path
    if 'train' in fn:
        split_raw = 'training'
        split_challenge = ADE20KMeta.SPLIT_TRAIN_CHALLENGE_2016
        split_panoptic = ADE20KMeta.SPLIT_TRAIN_PANOPTIC_2017
    elif 'val' in fn:
        split_raw = 'validation'
        split_challenge = ADE20KMeta.SPLIT_VALID_CHALLENGE_2016
        split_panoptic = ADE20KMeta.SPLIT_VALID_PANOPTIC_2017
    else:
        # currently no test split
        raise ValueError(f"Unknown split for sample '{fn}'")

    split_path_challenge = os.path.join(output_path, split_challenge)
    split_path_panoptic = os.path.join(output_path, split_panoptic)

    # get spatial shape from rgb image
    rgb_filepath = os.path.join(
        challenge_raw_path, 'images', split_raw, f'{fn}.jpg'
    )
    w, h = Image.open(rgb_filepath).size
    wxh_str = _get_image_size_str(w=w, h=h)

    # copy rgb image (part of challenge and panoptic splits) ------------------
    src_rgb_filepath = rgb_filepath
    # first, copy to the challenge directory
    dst_rgb_challenge = os.path.join(
        split_path_challenge, ADE20KMeta.IMAGE_DIR, wxh_str, f'{fn}.jpg'
    )
    create_dir(os.path.dirname(dst_rgb_challenge))
    shutil.copy(src_rgb_filepath, dst_rgb_challenge)

    # copy semantic image (part of challenge splits only) ---------------------
    src_semantic_filepath = os.path.join(
        challenge_raw_path, 'annotations', split_raw, f'{fn}.png'
    )
    dst_semantic_filepath = os.path.join(
        split_path_challenge, ADE20KMeta.SEMANTIC_DIR, wxh_str,
        f'{fn}.png'
    )
    create_dir(os.path.dirname(dst_semantic_filepath))
    shutil.copy(src_semantic_filepath, dst_semantic_filepath)

    # handle panoptic stuff - combine semantic and instance -------------------
    src_instance_filepath = os.path.join(
        challenge_raw_path, 'annotations_instance', split_raw, f'{fn}.png'
    )
    semantic = cv2.imread(src_semantic_filepath, cv2.IMREAD_UNCHANGED)
    instance = cv2.imread(src_instance_filepath, cv2.IMREAD_UNCHANGED)
    instance = cv2.cvtColor(instance, cv2.COLOR_BGR2RGB)
    panoptic_semantic, panoptic_instance = \
        _challenge_combine_semantic_and_instance(semantic, instance)
    dst_semantic_filepath = os.path.join(
        split_path_panoptic, ADE20KMeta.SEMANTIC_DIR, wxh_str,
        f'{fn}.png'
    )
    dst_instance_filepath = os.path.join(
        split_path_panoptic, ADE20KMeta.INSTANCES_DIR, wxh_str, f'{fn}.png'
    )
    _save_png(dst_semantic_filepath, panoptic_semantic)
    _save_png(dst_instance_filepath, panoptic_instance)

    # When combining semantic from challange 2016 with the instance images
    # from 2017, it can happen that an instance has multiple semantic classes.
    # When using the semantic from 2017 (i.e. from instance annotations),
    # this does not happen.
    # # small sanity check
    # dst_semantic_filepath = src_semantic_filepath
    # dst_instance_filepath = src_instance_filepath
    # semantic = cv2.imread(dst_semantic_filepath, cv2.IMREAD_UNCHANGED)
    # semantic_instance = cv2.imread(dst_instance_filepath, cv2.IMREAD_UNCHANGED)
    # semantic_instance = cv2.cvtColor(semantic_instance, cv2.COLOR_BGR2RGB)
    # # see: https://github.com/CSAILVision/placeschallenge/blob/master/instancesegmentation/evaluation/convert_anns_to_json_dataset.py#L47C3-L47C5
    # instance = semantic_instance[..., 1]
    # for instance_id in np.unique(instance):
    #     if instance_id == 0:
    #         continue   # no instance label

    #     # semantic from semantic image for classes
    #     vals, counts = np.unique(semantic[instance == instance_id],
    #                              return_counts=True)
    #     if len(vals) > 1:
    #         # this happens quite often
    #         print(f"Multiple semantic classes {vals}: {counts} for "
    #               f"instance id {instance_id} in semantic annotation for {fn}")

    #     # semantic from instance image for classes
    #     vals, counts = np.unique(
    #         semantic_instance[...,0][instance == instance_id],
    #         return_counts=True
    #     )
    #     if len(vals) > 1:
    #         # this does not happen
    #         print(f"Multiple semantic classes {vals}: {counts} for "
    #               f"instance id {instance_id} in instance annotation for {fn}")

    # scene class (part of challenge and panoptic splits) ---------------------
    for split_path in (split_path_challenge, split_path_panoptic):
        dst_scene_path = os.path.join(
            split_path, ADE20KMeta.SCENE_CLASS_DIR, wxh_str, f'{fn}.txt'
        )
        create_dir(os.path.dirname(dst_scene_path))
        with open(dst_scene_path, 'w') as file:
            file.write(sc)


def _challenge_combine_semantic_and_instance(semantic, instance_rgb):
    # combine semantic and instance information as done in many papers, e.g.,
    # Panoptic SegFormer, MaskFormer, Mask2Former, or OneFormer
    # see: https://github.com/facebookresearch/MaskFormer/blob/main/datasets/prepare_ade20k_pan_seg.py

    assert semantic.ndim == 2
    assert instance_rgb.ndim == 3   # R: semantic, G: instance, B: 0

    # get stuff class ids
    is_thing = np.array(
        ADE20KMeta.SEMANTIC_LABEL_LIST_CHALLENGE_150.classes_is_thing,
        dtype='bool'
    )
    stuff_ids = np.where(~is_thing)[0]

    # start with empty semantic and instance images
    panoptic_semantic = np.zeros_like(semantic, dtype='uint8')
    panoptic_instance = np.zeros_like(semantic, dtype='uint8')

    # add stuff classes to semantic
    mask_stuff = np.isin(semantic, stuff_ids)
    panoptic_semantic[mask_stuff] = semantic[mask_stuff]

    # add thing classes to semantic, enumerate instances
    instance_semantic, instance_ids = instance_rgb[..., 0], instance_rgb[..., 1]
    i = 0
    for instance_id in np.unique(instance_ids):
        if instance_id == 0:
            continue    # no instance label

        # get mask for current instance
        mask_instance = instance_ids == instance_id

        # there should be only one class for each instance
        semantic_classes = np.unique(instance_semantic[mask_instance])
        assert len(semantic_classes) == 1
        semantic_class = semantic_classes[0]

        # get mapped semantic class
        semantic_class = MAPPING_INSTANCE_100_TO_SCENE_PARSE_150[semantic_class]

        # copy semantic of from instance image to panoptic semantic
        panoptic_semantic[mask_instance] = semantic_class
        # get new instance id and assign to panoptic instance
        i += 1
        panoptic_instance[mask_instance] = i

    return panoptic_semantic, panoptic_instance


def main(args=None):
    # argument parser ---------------------------------------------------------
    args = _parse_args(args)

    # handle data filepath args -----------------------------------------------
    if args.challenge_2016_filepath is not None:
        challenge_2016_filepath = os.path.expanduser(
            args.challenge_2016_filepath
        )
        assert os.path.exists(challenge_2016_filepath), (
            f"Challenge 2016 filepath '{challenge_2016_filepath}' does not "
            "exist!"
        )
    if args.challenge_2017_instances_filepath is not None:
        challenge_2017_instances_filepath = os.path.expanduser(
            args.challenge_2017_instances_filepath
        )
        assert os.path.exists(challenge_2017_instances_filepath), (
            "Challenge 2017 instances filepath "
            f"'{challenge_2017_instances_filepath}' does not exist!"
        )

    output_path = os.path.expanduser(args.output_path)
    os.makedirs(output_path, exist_ok=True)
    tmp_path = os.path.join(output_path, 'tmp')
    os.makedirs(tmp_path, exist_ok=True)

    # download and extract challenge data --------------------------------------
    # -> semantic segmentations and scene labels
    filename = 'ADEChallengeData2016.zip'
    if args.challenge_2016_filepath is None:
        print(f"Downloading 2016 challenge data: '{filename}' ...")
        challenge_2016_filepath = os.path.join(tmp_path, filename)
        download_file(CHALLENGE_URLS[filename], challenge_2016_filepath,
                      display_progressbar=True)

    print(f"Extracting 2016 challenge data: '{filename}' ...")
    with ZipFile(challenge_2016_filepath, 'r') as zf:
        for member in tqdm(zf.infolist(), desc='', position=1):
            zf.extract(member, tmp_path)
    challenge_raw_path = os.path.join(tmp_path, 'ADEChallengeData2016')

    # -> instances
    filename = 'annotations_instance.tar'
    if args.challenge_2017_instances_filepath is None:
        print(f"Downloading 2017 challenge instance data: '{filename}'")
        challenge_2017_instances_filepath = os.path.join(tmp_path, filename)
        download_file(CHALLENGE_URLS[filename],
                      challenge_2017_instances_filepath,
                      display_progressbar=True)

    print(f"Extracting 2017 challenge instance data: '{filename}' ...")
    with TarFile(challenge_2017_instances_filepath, 'r') as tf:
        for member in tqdm(tf.getmembers(), desc=''):
            # files in tar do not have a top-level 'ADEChallengeData2016'
            # folder, so the files should be extracted to the challenge
            # folder instead of the tmp folder
            tf.extract(member, challenge_raw_path)

    # prepare 2016 challenge version ------------------------------------------
    print('Preparing challenge files ...')
    _challenge_prepare_data(challenge_raw_path,
                            output_path,
                            n_processes=args.n_processes)

    # write meta file
    create_or_update_creation_metafile(output_path)

    # clean up ----------------------------------------------------------------
    print('Cleaning up ...')


if __name__ == '__main__':
    main()
