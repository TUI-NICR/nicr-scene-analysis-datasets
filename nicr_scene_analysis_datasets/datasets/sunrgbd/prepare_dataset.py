# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
import json
import os
import shutil
from zipfile import ZipFile

import cv2
import h5py
import numpy as np
import scipy.io
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ...utils.img import save_indexed_png
from ...utils.io import download_file
from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from ...utils.io import extract_zip
from . import prepare_instances
from .match_nyuv2_instances import NYUv2InstancesMatcher
from .sunrgbd import SUNRGBDMeta


# see: http://rgbd.cs.princeton.edu/ in section Data and Annotation
DATASET_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
DATASET_TOOLBOX_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip'
DATASET_BOX3D_URL = 'https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat'


TMP_DIR = 'tmp'


def main():
    # argument parser
    parser = ap.ArgumentParser(
        description='Prepare SUNRGB-D dataset.')
    parser.add_argument('output_path', type=str,
                        help="Path where to store dataset")
    parser.add_argument('--toolbox-filepath', default=None,
                        help="Filepath to SUNRGBD toolbox file.")
    parser.add_argument('--data-filepath', default=None,
                        help="Filepath to SUNRGBD data zip file.")
    parser.add_argument('--box-filepath', default=None,
                        help="filepath to SUNRGBD 3d bounding boxes")
    parser.add_argument('--create-instances', default=False,
                        action="store_true",
                        help="Whether the instances should be created by "
                             "matching 3d boxes into a pointcloud.")
    parser.add_argument('--copy-instances-from-nyuv2', default=False,
                        action="store_true",
                        help="whether the orientations of SUNRGBD should "
                             "be matched to NYUv2.")
    parser.add_argument('--nyuv2-path', default="", type=str,
                        help="Path to NYUv2 dataset for matching.")

    args = parser.parse_args()

    # output path
    output_path = os.path.expanduser(args.output_path)
    create_dir(output_path)

    # path where to store stuff during creation
    tmp_path = os.path.join(output_path, TMP_DIR)
    create_dir(tmp_path)

    # update or write metafile
    create_or_update_creation_metafile(output_path)

    # download and extract data
    # toolbox
    if args.toolbox_filepath is None:
        # we need to download file
        zip_file_path = os.path.join(tmp_path, 'SUNRGBDtoolbox.zip')
        download_file(DATASET_TOOLBOX_URL, zip_file_path,
                      display_progressbar=True)
    else:
        zip_file_path = args.toolbox_filepath
    print(f"Extracting toolbox from: '{zip_file_path}'")
    extract_zip(zip_file_path, tmp_path)

    # data
    if args.data_filepath is None:
        zip_file_path = os.path.join(tmp_path, 'SUNRGBD.zip')
        download_file(DATASET_URL, zip_file_path, display_progressbar=True)
    else:
        zip_file_path = args.data_filepath
    print(f"Extracting images from '{zip_file_path}'")
    extract_zip(zip_file_path, tmp_path)
    data_path = os.path.join(tmp_path, 'SUNRGBD')

    # 3d boxes
    if args.box_filepath is None:
        boxes_3d_path = os.path.join(tmp_path, 'SUNRGBDMeta3DBB_v2.mat')
        download_file(DATASET_BOX3D_URL, boxes_3d_path,
                      display_progressbar=True)
    else:
        boxes_3d_path = args.box_filepath

    toolbox_path = os.path.join(tmp_path, 'SUNRGBDtoolbox')
    SUNRGBDMeta_fp = os.path.join(toolbox_path, 'Metadata', 'SUNRGBDMeta.mat')
    allsplit_fp = os.path.join(toolbox_path, 'traintestSUNRGBD', 'allsplit.mat')
    SUNRGBD2Dseg_fp = os.path.join(toolbox_path, 'Metadata', 'SUNRGBD2Dseg.mat')

    train_dirs = []
    test_dirs = []

    SUNRGBD2Dseg = h5py.File(SUNRGBD2Dseg_fp, mode='r', libver='latest')

    # load the data from the matlab file
    SUNRGBDMetaMat = scipy.io.loadmat(SUNRGBDMeta_fp, squeeze_me=True,
                                      struct_as_record=False)['SUNRGBDMeta']

    SUNRGBDMeta3DBoxes = \
        scipy.io.loadmat(boxes_3d_path, squeeze_me=True,
                         struct_as_record=False)['SUNRGBDMeta']

    split = scipy.io.loadmat(allsplit_fp, squeeze_me=True,
                             struct_as_record=False)
    split_train = split['alltrain']

    seglabel = SUNRGBD2Dseg['SUNRGBD2Dseg']['seglabel']

    instances_converter = prepare_instances.SUNRGBDInstances()

    print("Processing files")
    for i, (meta, meta_3d) in tqdm(enumerate(zip(SUNRGBDMetaMat,
                                                 SUNRGBDMeta3DBoxes)),
                                   total=len(SUNRGBDMetaMat)):

        assert meta.rgbname == meta_3d.rgbname

        # e.g., /n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg
        rgb_filepath_split = meta.rgbpath.split('/')

        # e.g., /n/fs/sun3d/data/SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize
        meta_path = '/'.join(meta.rgbpath.split('/')[:-2])
        is_train = meta_path in split_train
        split_dir = 'train' if is_train else 'test'

        # e.g., kv2/kinect2data
        cam_path = os.path.join(*rgb_filepath_split[6:8])

        # e.g., kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize
        sample_path = os.path.join(*rgb_filepath_split[6:-2])

        # Depth images ---------------------------------------------------------
        # refined depth image
        depth_bfx_path_tmp = os.path.join(data_path, sample_path, 'depth_bfx',
                                          meta.depthname)
        depth_bfx_path = os.path.join(output_path, split_dir,
                                      SUNRGBDMeta.DEPTH_DIR,
                                      cam_path, f'{i:05d}.png')

        create_dir(os.path.dirname(depth_bfx_path))
        os.replace(depth_bfx_path_tmp, depth_bfx_path)

        # raw depth image
        depth_raw_path_tmp = os.path.join(data_path, sample_path, 'depth',
                                          meta.depthname)
        depth_raw_path = os.path.join(output_path, split_dir,
                                      SUNRGBDMeta.DEPTH_DIR_RAW,
                                      cam_path, f'{i:05d}.png')
        create_dir(os.path.dirname(depth_raw_path))
        os.replace(depth_raw_path_tmp, depth_raw_path)

        # RGB image ------------------------------------------------------------
        rgb_path_tmp = os.path.join(data_path, sample_path, 'image',
                                    meta.rgbname)
        rgb_path = os.path.join(output_path, split_dir, SUNRGBDMeta.IMAGE_DIR,
                                cam_path, f'{i:05d}.jpg')
        create_dir(os.path.dirname(rgb_path))
        os.replace(rgb_path_tmp, rgb_path)

        # Semantic label -------------------------------------------------------
        # note, we store the labels in SUNRGBD and NYUv2 colormap
        semantic_path = os.path.join(output_path, split_dir,
                                     SUNRGBDMeta.SEMANTIC_DIR, cam_path,
                                     f'{i:05d}.png')
        create_dir(os.path.dirname(semantic_path))
        semantic = np.array(SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0))
        semantic = semantic.astype(np.uint8)
        cv2.imwrite(semantic_path, semantic)

        semantic_path_colored_sun = os.path.join(
            output_path, split_dir, SUNRGBDMeta.SEMANTIC_COLORED_DIR_SUN,
            cam_path, f'{i:05d}.png'
        )
        create_dir(os.path.dirname(semantic_path_colored_sun))
        save_indexed_png(semantic_path_colored_sun, semantic,
                         np.array(SUNRGBDMeta.SEMANTIC_CLASS_COLORS,
                                  dtype='uint8'))

        semantic_path_colored_nyuv2 = os.path.join(
            output_path, split_dir, SUNRGBDMeta.SEMANTIC_COLORED_DIR_NYUV2,
            cam_path, f'{i:05d}.png'
        )
        create_dir(os.path.dirname(semantic_path_colored_nyuv2))
        save_indexed_png(semantic_path_colored_nyuv2, semantic,
                         np.array(SUNRGBDMeta.SEMANTIC_CLASS_COLORS_NYUV2,
                                  dtype='uint8'))

        # Scene class ----------------------------------------------------------
        scene_class_path_tmp = os.path.join(data_path, sample_path,
                                            'scene.txt')
        scene_class_path = os.path.join(output_path,
                                        split_dir,
                                        SUNRGBDMeta.SCENE_CLASS_DIR,
                                        cam_path,
                                        f'{i:05d}.txt')

        create_dir(os.path.dirname(scene_class_path))
        os.replace(scene_class_path_tmp, scene_class_path)

        # 3D boxes and orientations --------------------------------------------
        boxes_3d_path = os.path.join(output_path, split_dir,
                                     SUNRGBDMeta.BOX_DIR, cam_path,
                                     f'{i:05d}.json')

        boxes_3d_dict = {}

        create_dir(os.path.dirname(boxes_3d_path))

        extrinsic = meta_3d.anno_extrinsics.tolist()
        boxes_3d_dict['extrinsics'] = extrinsic
        # save all box information
        boxes_3d_dict['basis'] = []
        boxes_3d_dict['centroid'] = []
        boxes_3d_dict['class'] = []
        boxes_3d_dict['coeffs'] = []
        boxes_3d_dict['orientations'] = []
        orientations_list = []
        if not isinstance(meta_3d.groundtruth3DBB, np.ndarray):
            meta_3d.groundtruth3DBB = [meta_3d.groundtruth3DBB]

        for box in meta_3d.groundtruth3DBB:
            boxes_3d_dict['basis'].append(box.basis.tolist())
            boxes_3d_dict['centroid'].append(box.centroid.tolist())
            boxes_3d_dict['class'].append(box.classname)
            boxes_3d_dict['coeffs'].append(box.coeffs.tolist())
            boxes_3d_dict['orientations'] = box.orientation.tolist()
            tau = 2*np.pi
            ang = (np.arctan2(boxes_3d_dict['orientations'][0],
                              -boxes_3d_dict['orientations'][1])+tau) % tau
            orientations_list.append(ang)
        boxes_3d_dict = instances_converter.convert_boxes_3d(boxes_3d_dict)

        # store extrinsics for later and delete it from box dict
        extrinsics = boxes_3d_dict['extrinsics']
        del boxes_3d_dict['extrinsics']

        # save 3d boxes
        with open(boxes_3d_path, "w") as j:
            # convert ndarray to list
            for k, v in boxes_3d_dict.items():
                if isinstance(v, np.ndarray):
                    boxes_3d_dict[k] = v.tolist()
            json.dump(boxes_3d_dict, j, indent=4)

        # Extrinsics -----------------------------------------------------------
        extrinsics_path = os.path.join(output_path, split_dir,
                                       SUNRGBDMeta.EXTRINSICS_DIR, cam_path,
                                       f'{i:05d}.json')
        create_dir(os.path.dirname(extrinsics_path))

        extrinsic_quat = R.from_matrix(extrinsic).as_quat()
        quat_x, quat_y, quat_z, quat_w = extrinsic_quat
        extrinsics_dict = {
            'x': 0,
            'y': 0,
            'z': 0,
            'quat_x': quat_x,
            'quat_y': quat_y,
            'quat_z': quat_z,
            'quat_w': quat_w
        }

        with open(extrinsics_path, 'w') as outfile:
            json.dump(extrinsics_dict, outfile, indent=4)

        # Instances ------------------------------------------------------------
        instance_path = os.path.join(output_path,
                                     split_dir,
                                     SUNRGBDMeta.INSTANCES_DIR,
                                     cam_path,
                                     f'{i:05d}.png')
        intrinsics_path_tmp = os.path.join(data_path, sample_path,
                                           'intrinsics.txt')

        intrinsics = instances_converter.load_intrinsics(intrinsics_path_tmp)

        if args.create_instances and not os.path.isfile(instance_path):
            depth_img = cv2.imread(depth_bfx_path, cv2.IMREAD_UNCHANGED)

            instance_img, box_mapping = \
                instances_converter.get_instance(boxes_3d_dict,
                                                 intrinsics,
                                                 extrinsics,
                                                 depth_img,
                                                 semantic)

            create_dir(os.path.dirname(instance_path))
            cv2.imwrite(instance_path, instance_img)

            boxes_3d_dict["instance_numbers"] = box_mapping.tolist()
        else:
            # Only imread if file exists. Else set instance_img to None
            # so no orientations are created.
            if os.path.isfile(instance_path):
                instance_img = cv2.imread(instance_path, cv2.IMREAD_UNCHANGED)
            else:
                instance_img = None

        # Intrinsics -----------------------------------------------------------
        intrinsics_path = os.path.join(output_path, split_dir,
                                       SUNRGBDMeta.INTRINSICS_DIR, cam_path,
                                       f'{i:05d}.json')
        create_dir(os.path.dirname(intrinsics_path))

        # Normalize and save the intrinsics
        # load rgb image to get the shape
        rgb_img = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        h, w, _ = rgb_img.shape
        normalized_intrinsics = {
            'fx': intrinsics['Fx'] / w,
            'fy': intrinsics['Fy'] / h,
            'cx': intrinsics['Cx'] / w,
            'cy': intrinsics['Cy'] / h
        }
        with open(intrinsics_path, "w") as j:
            json.dump(normalized_intrinsics, j, indent=4)

        # Orientations ---------------------------------------------------------

        # The orientations of the instance can only be saved, if instances are
        # created or loaded.
        # instance_img will be None, if it couldn't be loaded
        if args.create_instances or instance_img is not None:
            orientations_path = os.path.join(output_path, split_dir,
                                             SUNRGBDMeta.ORIENTATIONS_DIR,
                                             cam_path, f'{i:05d}.json')
            create_dir(os.path.dirname(orientations_path))

            orientations = np.array(orientations_list)
            orientations_dict = {}
            for key, value in enumerate(orientations):
                # +1 as 0 indicates void/no instance
                current_key = key + 1
                mask = instance_img == current_key
                # This happens when no pixel was inside a 3d box, which leads
                # to no instance being created
                if mask.sum() == 0:
                    continue

                orientations_dict[current_key] = value

            with open(orientations_path, "w") as j:
                json.dump(orientations_dict, j, indent=4)

            if is_train:
                train_dirs.append(os.path.join(cam_path, f"{i:05d}"))
            else:
                test_dirs.append(os.path.join(cam_path, f"{i:05d}"))

    # write file lists
    def _write_list_to_file(list_, filepath):
        with open(os.path.join(output_path, filepath), 'w') as f:
            f.write('\n'.join(list_))
        print(f"Written file {filepath}")

    _write_list_to_file(train_dirs, 'train.txt')
    _write_list_to_file(test_dirs, 'test.txt')

    if args.copy_instances_from_nyuv2 and args.create_instances:
        print("Matching NYUv2 instance masks to SUNRGBD")
        matcher = NYUv2InstancesMatcher(args.output_path,
                                        args.nyuv2_path)
        matcher.do_matching()

    # cleanup
    SUNRGBD2Dseg.close()
    print("Removing temporary files")
    shutil.rmtree(tmp_path)


if __name__ == '__main__':
    main()
