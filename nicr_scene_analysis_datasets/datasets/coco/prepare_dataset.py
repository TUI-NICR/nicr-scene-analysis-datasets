# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
from functools import partial
import json
import os
import shutil
from zipfile import ZipFile

import cv2
import numpy as np
from panopticapi.utils import rgb2id
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ...utils.img import save_indexed_png
from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from ...utils.io import download_file
from .coco import COCOMeta


# see: https://cocodataset.org/#download in section images and annotation
DATASET_URL_TRAIN = 'http://images.cocodataset.org/zips/train2017.zip'
DATASET_URL_VALID = 'http://images.cocodataset.org/zips/val2017.zip'
DATASET_URL_TEST = 'http://images.cocodataset.org/zips/test2017.zip'
DATASET_URL_TRAIN_VAL_ANNOTATIONS = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
DATASET_URL_TRAIN_VAL_STUFF_ANNOTATIONS = 'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip'
DATASET_URL_TRAIN_VAL_PANOPTIC_ANNOTATIONS = 'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip'


def _extract_semantic_and_instance_annotation(annotation,
                                              panoptic_path,
                                              output_path):

    # load panoptic image
    panoptic_fp = os.path.join(panoptic_path, annotation['file_name'])
    panoptic_img = cv2.imread(panoptic_fp, cv2.IMREAD_UNCHANGED)
    panoptic_img = cv2.cvtColor(panoptic_img, cv2.COLOR_BGR2RGB)

    # extract semantic and instance
    # see https://github.com/cocodataset/panopticapi/blob/master/converters/panoptic2semantic_segmentation.py
    pan = rgb2id(panoptic_img)
    semantic = np.zeros(pan.shape, dtype=np.uint8)
    instance = np.zeros(pan.shape, dtype=np.uint8)
    instance_ctr = 1
    for segm_info in annotation['segments_info']:
        cat_id = segm_info['category_id']

        # convert to contiguous indices
        cat_id = COCOMeta.COCO_ID.index(cat_id)
        mask = pan == segm_info['id']
        semantic[mask] = cat_id
        instance[mask] = instance_ctr
        instance_ctr += 1

    # store output images
    wxh_str = _get_image_size_str_as_w_x_h(pan)

    # semantic
    path = os.path.join(output_path, COCOMeta.SEMANTIC_DIR, wxh_str)
    create_dir(path)
    cv2.imwrite(os.path.join(path, annotation['file_name']), semantic)

    # semantic colored
    class_colors = COCOMeta.SEMANTIC_LABEL_LIST.colors_array
    path = os.path.join(output_path, COCOMeta.SEMANTIC_COLORED_DIR, wxh_str)
    create_dir(path)
    save_indexed_png(os.path.join(path, annotation['file_name']),
                     semantic, class_colors)

    # instance
    path = os.path.join(output_path, COCOMeta.INSTANCES_DIR, wxh_str)
    create_dir(path)
    cv2.imwrite(os.path.join(path, annotation['file_name']), instance)


def _get_image_size_str_as_w_x_h(value):
    if isinstance(value, np.ndarray):
        # value is a (loaded) image
        assert value.ndim in (2, 3)
        return '{w}x{h}'.format(w=value.shape[1], h=value.shape[0])

    elif isinstance(value, str):
        # assume value is a path
        assert os.path.exists(value)
        img = cv2.imread(value, cv2.IMREAD_UNCHANGED)
        return _get_image_size_str_as_w_x_h(img)

    raise ValueError()


def main():
    # argument parser
    parser = ap.ArgumentParser(description="Prepare COCO dataset.")
    parser.add_argument('output_path', type=str,
                        help="Path where to store dataset.")
    parser.add_argument('--do-not-delete-zip', default=False,
                        action="store_true",
                        help="Whether to delete the zips after extraction.")
    parser.add_argument('--download-path', type=str, default=None,
                        help="Path where to store the downloaded dataset files"
                             "(i.e. before extraction). If this is not set, "
                             "the files will be downloaded into the "
                             "`output_path`.")

    args = parser.parse_args()

    # expand user
    output_path = os.path.expanduser(args.output_path)
    os.makedirs(output_path, exist_ok=True)

    # write meta file
    create_or_update_creation_metafile(output_path)

    if args.download_path is None:
        download_path = output_path
    else:
        download_path = os.path.expanduser(args.download_path)
    os.makedirs(download_path, exist_ok=True)

    datasets_url = {
        'training images': DATASET_URL_TRAIN,
        'validation images': DATASET_URL_VALID,
        # 'test images': DATASET_URL_TEST
        'annotations': DATASET_URL_TRAIN_VAL_ANNOTATIONS,
        'stuff annotations': DATASET_URL_TRAIN_VAL_STUFF_ANNOTATIONS,
        'panoptic annotations': DATASET_URL_TRAIN_VAL_PANOPTIC_ANNOTATIONS
    }

    # download and extract dataset files ---------------------------------------
    for name, url in datasets_url.items():
        filename = url.split('/')[-1]
        zip_file_path = os.path.join(download_path, filename)
        if not os.path.exists(zip_file_path):
            print(f"Downloading {name}: {url}")
            download_file(url, zip_file_path,
                          display_progressbar=True)

        print(f"Extracting {name}")
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(output_path, 'extracted'))

        if not args.do_not_delete_zip:
            print(f"Deleting '{filename}'.")
            os.remove(zip_file_path)

    # extract panoptic annotations
    annotations_path = os.path.join(output_path, 'extracted', 'annotations')
    annotations_zips = [
        os.path.join(annotations_path, 'panoptic_train2017.zip'),
        os.path.join(annotations_path, 'panoptic_val2017.zip')
    ]
    for zip_file_path in annotations_zips:
        with ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.join(os.path.dirname(zip_file_path),
                                            'extracted'))

    # prepare dataset ----------------------------------------------------------

    # note, we split the dataset in several parts, i.e. virtual cameras, based
    # on the image size as this simplifies processing in our pipelines
    vcam_lookup_dict = {s: {} for s in COCOMeta.SPLITS}

    for split, split_coco in zip(COCOMeta.SPLITS,
                                 ('train2017', 'val2017')):

        print(f"Preparing '{split}' split")

        # create directories
        split_path = os.path.join(output_path, split)
        create_dir(split_path)

        for sample_dir in (COCOMeta.IMAGE_DIR,
                           COCOMeta.SEMANTIC_DIR,
                           COCOMeta.SEMANTIC_COLORED_DIR,
                           COCOMeta.INSTANCES_DIR):
            create_dir(os.path.join(split_path, sample_dir))

        # move images to the correct location and fill vcam lookup dict
        print(f"-> moving images")

        tmp_path = os.path.join(output_path, 'extracted', split_coco)
        img_path = os.path.join(split_path, COCOMeta.IMAGE_DIR)

        for fn in tqdm(sorted(os.listdir(tmp_path))):
            if fn.endswith('.jpg'):
                fp = os.path.join(tmp_path, fn)

                # get subfolder name based on image size
                wxh_str = _get_image_size_str_as_w_x_h(fp)

                # store image size in lookup dict
                vcam_lookup_dict[split][os.path.splitext(fn)[0]] = wxh_str

                create_dir(os.path.join(img_path, wxh_str))
                shutil.move(fp, os.path.join(img_path, wxh_str, fn))

        # semantic and instance annotations
        print("-> converting panoptic annotations to semantic and instance "
              "annotations")
        panoptic_json_fp = os.path.join(annotations_path,
                                        f'panoptic_{split_coco}.json')
        panoptic_path = os.path.join(annotations_path, 'extracted',
                                     f'panoptic_{split_coco}')

        with open(panoptic_json_fp, 'r') as f:
            d_coco = json.load(f)
        annotations = d_coco['annotations']

        cpu_cnt = os.cpu_count()
        fn = partial(_extract_semantic_and_instance_annotation,
                     panoptic_path=panoptic_path,
                     output_path=split_path)

        process_map(fn, annotations,
                    max_workers=cpu_cnt,
                    chunksize=100,
                    total=len(annotations))

        # perform small sanity check
        for sample_dir in (COCOMeta.IMAGE_DIR,
                           COCOMeta.SEMANTIC_DIR,
                           COCOMeta.SEMANTIC_COLORED_DIR,
                           COCOMeta.INSTANCES_DIR):
            ext = '.jpg' if COCOMeta.IMAGE_DIR == sample_dir else '.png'
            for fn, wxh_str in vcam_lookup_dict[split].items():
                fp = os.path.join(split_path, sample_dir, wxh_str, f'{fn}{ext}')
                assert os.path.exists(fp), fp

        # write filelist
        print("-> writing filelist")
        fp = os.path.join(output_path, COCOMeta.SPLIT_FILELIST_FILENAMES[split])
        with open(fp, 'w') as f:
            for fn, wxh_str in vcam_lookup_dict[split].items():
                f.write(f'{os.path.join(wxh_str, fn)}\n')

    # clean up
    extraction_path = os.path.join(output_path, 'extracted')
    shutil.rmtree(extraction_path)


if __name__ == '__main__':
    main()
