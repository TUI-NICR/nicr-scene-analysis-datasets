# -*- coding: utf-8 -*-
"""
.. codeauthor:: Marius Engelhardt <marius.engelhardt@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import argparse as ap
import datetime
import json
import os

import cv2
import h5py
import re
import numpy as np
import pandas as pd
import functools
from scipy.spatial.transform import Rotation
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from ...utils.io import create_or_update_creation_metafile
from ...utils.io import download_file
from ...utils.img import save_indexed_png as save_indexed_png_
from .hypersim import HypersimMeta


SPLITS_CSV_URL = 'https://raw.githubusercontent.com/apple/ml-hypersim/6cbaa80207f44a312654e288cf445016c84658a1/' \
                 'evermotion_dataset/analysis/metadata_images_split_scene_v1.csv'

SCENE_CLASS_URL = 'https://raw.githubusercontent.com/apple/ml-hypersim/6cbaa80207f44a312654e288cf445016c84658a1/' \
                  'evermotion_dataset/analysis/metadata_camera_trajectories.csv'


def color_semgentation(label, with_void=True):
    class_colors = HypersimMeta.CLASS_COLORS
    if with_void:
        colors = class_colors
    else:
        colors = class_colors[1:]
    cmap = np.asarray(colors, dtype='uint8')

    return cmap[label], cmap


def save_indexed_png(filepath, label, colormap):
    assert label.min() >= 0 and label.max() <= 40, f'Label should be in [0, 40]' \
                                                   f' but is in [{label.min()}, {label.max()}]'
    return save_indexed_png_(filepath, label, colormap)


def load_depth(scene_path, cam='cam_00', frame=0, correct_focal=True, in_mm=True):
    depth_path = os.path.join(scene_path, 'images', f'scene_{cam}_geometry_hdf5',
                              f'frame.{frame:04d}.depth_meters.hdf5')

    with h5py.File(depth_path, 'r') as f:
        depth = f['dataset'][:].astype(np.float32)

    if correct_focal:
        # code from https://github.com/apple/ml-hypersim/issues/9#issuecomment-754935697
        int_width = depth.shape[1]
        int_height = depth.shape[0]
        flt_focal = 886.81

        npy_imageplane_x = np.linspace((-0.5 * int_width) + 0.5, (0.5 * int_width) - 0.5, int_width)
        npy_imageplane_x = npy_imageplane_x.reshape(1, int_width).repeat(int_height, 0).astype(np.float32)[:, :, None]
        npy_imageplane_y = np.linspace((-0.5 * int_height) + 0.5, (0.5 * int_height) - 0.5, int_height)
        npy_imageplane_y = npy_imageplane_y.reshape(int_height, 1).repeat(int_width, 1).astype(np.float32)[:, :, None]
        npy_imageplane_z = np.full([int_height, int_width, 1], flt_focal, np.float32)
        npy_imageplane = np.concatenate([npy_imageplane_x, npy_imageplane_y, npy_imageplane_z], 2)

        depth = depth / np.linalg.norm(npy_imageplane, 2, 2) * flt_focal

    if in_mm:
        depth = depth * 1000
        depth[depth >= 2 ** 16] = 0

    return depth.astype('uint16')


def load_normal(scene_path, cam='cam_00', frame=0):
    normal_path = os.path.join(scene_path,
                               'images',
                               f'scene_{cam}_geometry_hdf5',
                               f'frame.{frame:04d}.normal_bump_cam.hdf5')

    with h5py.File(normal_path, 'r') as f:
        normal = f['dataset'][:].astype(np.float32)

    # note: every valid normal vector in the dataset is already normalized to
    # have unit length

    # convert back to uint8, clipping is necessary to avoid over/-underflows
    normal = ((normal + 1)*127).clip(0, 255).astype('uint8')
    # convert xyz to zyz as imwrite will later swap it again
    normal = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
    return normal


def load_semantic(scene_path, cam='cam_00', frame=0, orig_mapping=False):
    # note, by default, we replace -1 (void in original dataset) with 0
    # (void in other datasets).
    gt_path = os.path.join(scene_path,
                           'images',
                           f'scene_{cam}_geometry_hdf5',
                           f'frame.{frame:04d}.semantic.hdf5')

    with h5py.File(gt_path, 'r') as f:
        gt = f['dataset'][:].astype(np.int32)

    assert 0 == np.sum(gt == 0)

    if not orig_mapping:
        gt[gt == -1] = 0

    return gt


def load_instance(scene_path, cam='cam_00', frame=0):
    gt_path = os.path.join(scene_path,
                           'images',
                           f'scene_{cam}_geometry_hdf5',
                           f'frame.{frame:04d}.semantic_instance.hdf5')

    with h5py.File(gt_path, 'r') as f:
        gt = f['dataset'][:]

    # Min value is -1 and max value is 4166, so we shift it to store it in an
    # uint16
    gt[gt == -1] = 0    # note there is no instance with id 0
    gt = gt.astype(np.uint16)

    return gt


def load_box_3d(scene_path, cam='cam_00', frame=0):
    extents = os.path.join(
        scene_path,
        '_detail',
        'mesh',
        'metadata_semantic_instance_bounding_box_object_aligned_2d_extents.hdf5'
    )

    orientations = os.path.join(
        scene_path,
        '_detail',
        'mesh',
        'metadata_semantic_instance_bounding_box_object_aligned_2d_orientations.hdf5'
    )
    positions = os.path.join(
        scene_path,
        '_detail',
        'mesh',
        'metadata_semantic_instance_bounding_box_object_aligned_2d_positions.hdf5'
    )
    r_dict = {}
    dict_keys = ['extents', 'orientations', 'positions']
    for path, name in zip([extents, orientations, positions], dict_keys):
        with h5py.File(path, 'r') as f:
            gt = f['dataset'][:].astype(np.float32)
            r_dict[name] = gt.tolist()

    return r_dict


def tonemap_image(scene_path, cam='cam_00', frame=0):
    # see: https://github.com/apple/ml-hypersim/blob/master/code/python/tools/scene_generate_images_tonemap.py
    rgb_hdf5_path = os.path.join(
        scene_path,
        'images',
        f'scene_{cam}_final_hdf5',
        f'frame.{frame:04d}.color.hdf5'
    )
    rendr_id_path = os.path.join(
        scene_path,
        'images',
        f'scene_{cam}_geometry_hdf5',
        f'frame.{frame:04d}.render_entity_id.hdf5'
    )

    with h5py.File(rgb_hdf5_path, 'r') as f:
        rgb_color = f['dataset'][:].astype(np.float32)

    with h5py.File(rendr_id_path, 'r') as f:
        render_id = f['dataset'][:].astype(np.int32)

    #
    # compute brightness according to "CCIR601 YIQ" method, use CGIntrinsics strategy for tonemapping, see [1,2]
    # [1] https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py
    # [2] https://landofinterruptions.co.uk/manyshades
    #

    assert np.all(render_id != 0)

    gamma = 1.0 / 2.2  # standard gamma correction exponent
    inv_gamma = 1.0 / gamma
    percentile = 90  # we want this percentile brightness value in the unmodified image...
    brightness_nth_percentile_desired = 0.8  # ...to be this bright after scaling

    valid_mask = render_id != -1

    if np.count_nonzero(valid_mask) == 0:
        scale = 1.0  # if there are no valid pixels, then set scale to 1.0
    else:
        brightness = 0.3 * rgb_color[:, :, 0] + \
                     0.59 * rgb_color[:, :, 1] + \
                     0.11 * rgb_color[:, :, 2]  # "CCIR601 YIQ" method for computing brightness
        brightness_valid = brightness[valid_mask]

        eps = 0.0001  # if the kth percentile brightness value in the unmodified image
        # is less than this, set the scale to 0.0 to avoid divide-by-zero
        brightness_nth_percentile_current = np.percentile(brightness_valid, percentile)

        if brightness_nth_percentile_current < eps:
            scale = 0.0
        else:

            # Snavely uses the following expression in the code at
            # https://github.com/snavely/pbrs_tonemapper/blob/master/tonemap_rgbe.py:
            # scale = np.exp(np.log(brightness_nth_percentile_desired)*inv_gamma -\
            #         np.log(brightness_nth_percentile_current))
            #
            # Our expression below is equivalent, but is more intuitive, because it
            # follows more directly from the expression:
            # (scale*brightness_nth_percentile_current)^gamma = brightness_nth_percentile_desired

            scale = np.power(brightness_nth_percentile_desired, inv_gamma) / brightness_nth_percentile_current

    rgb_color_tm = np.power(np.maximum(scale * rgb_color, 0), gamma)

    # clip image?
    rgb = (np.clip(rgb_color_tm, 0, 1) * 255).astype(np.uint8)

    return rgb


def load_extrinsics(scene_path, cam='cam_00', frame=0):
    # translation
    assets_to_meters_path = os.path.join(scene_path, '_detail',
                                         'metadata_scene.csv')
    assets_to_meters_csv = pd.read_csv(assets_to_meters_path)
    assets_to_meters = assets_to_meters_csv['parameter_value'][0]

    translation_path = os.path.join(scene_path, '_detail', cam,
                                    'camera_keyframe_positions.hdf5')
    with h5py.File(translation_path, 'r') as f:
        camera_position = f['dataset'][frame]
    camera_position *= assets_to_meters

    # rotation
    rotation_path = os.path.join(scene_path, '_detail', cam,
                                 'camera_keyframe_orientations.hdf5')
    with h5py.File(rotation_path, 'r') as f:
        camera_orientations = f['dataset'][frame]
    rotation_quat = Rotation.from_matrix(camera_orientations).as_quat()
    quat_x, quat_y, quat_z, quat_w = rotation_quat

    return {
        'x': camera_position[0],
        'y': camera_position[1],
        'z': camera_position[2],
        'quat_x': quat_x,
        'quat_y': quat_y,
        'quat_z': quat_z,
        'quat_w': quat_w
    }


def generate_split_txts(split_df: pd.DataFrame, destination_path,
                        splits=None, subsample=1, use_blacklist=True):
    assert subsample > 0, "Undefined behaviour for subsample < 1"
    if splits is None:
        splits = list(split_df['split_partition_name'].unique())

    f_names = HypersimMeta.get_split_filelist_filenames(subsample)

    if use_blacklist:
        for scene, camera in HypersimMeta.BLACKLIST.items():
            if camera == '*':  # exclude all trajectories of scene
                old_len = len(split_df)
                split_df = split_df[~(split_df['scene_name'] == scene)]
                print(f"Excluded scene {scene}. "
                      f"{old_len - len(split_df)} samples removed.")
            else:  # only exclude certain trajectories in scene
                old_len = len(split_df)
                split_df = split_df[~((split_df['scene_name'] == scene) & (split_df['camera_name'] == camera))]
                print(f"Excluded {camera} from scene {scene}. "
                      f"{old_len - len(split_df)} samples removed.")

    for split in splits:
        f_name = f_names[split]

        # get filepaths
        filepaths = split_df[split_df['split_partition_name'] == split]['path']
        # remove file extension
        filepaths = [fp.replace('.png', '') for fp in filepaths]
        # apply subsampling
        filepaths = filepaths[::subsample]

        np.savetxt(os.path.join(destination_path, f_name),
                   filepaths,
                   fmt="%s")
        print(f"Created filelist: '{f_name}'")


def process_one_sample(destination_path, dataset_path,
                       overwrite, cv_write_flags, index_entry):
    index, entry = index_entry

    # rgb image with default tonemapping ---------------------------------------
    rgb_path = os.path.join(destination_path,
                            entry['split_partition_name'],
                            HypersimMeta.RGB_DIR,
                            entry['path'])

    if overwrite['rgb'] or not os.path.exists(rgb_path):
        rgb = tonemap_image(os.path.join(dataset_path, entry['scene_name']),
                            cam=entry['camera_name'],
                            frame=entry['frame_id'])
        os.makedirs(os.path.dirname(rgb_path), exist_ok=True)
        assert cv2.imwrite(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                           cv_write_flags), f'Failed to save {rgb_path}'

    # depth image with mm as int16 ---------------------------------------------
    depth_path = os.path.join(destination_path, entry['split_partition_name'],
                              HypersimMeta.DEPTH_DIR,
                              entry['path'])

    if overwrite['depth'] or not os.path.exists(depth_path):
        depth = load_depth(os.path.join(dataset_path, entry['scene_name']),
                           cam=entry['camera_name'],
                           frame=entry['frame_id'])
        os.makedirs(os.path.dirname(depth_path), exist_ok=True)

        assert cv2.imwrite(depth_path, depth, cv_write_flags), f'Failed to save {depth}'

    # normal ------------------------------------------------------------------
    normal_path = os.path.join(destination_path, entry['split_partition_name'],
                               HypersimMeta.NORMAL_DIR,
                               entry['path'])
    if overwrite['normal'] or not os.path.exists(normal_path):
        normal = load_normal(os.path.join(dataset_path, entry['scene_name']),
                             cam=entry['camera_name'],
                             frame=entry['frame_id'])
        os.makedirs(os.path.dirname(normal_path), exist_ok=True)

        assert cv2.imwrite(normal_path, normal), f'Failed to save {normal}'

    # semantic (indexed and colored png) ---------------------------------------
    colored_path = os.path.join(destination_path,
                                entry['split_partition_name'],
                                HypersimMeta.SEMANTIC_COLORED_DIR,
                                entry['path'])
    semantic_path = os.path.join(destination_path,
                                 entry['split_partition_name'],
                                 HypersimMeta.SEMANTIC_DIR,
                                 entry['path'])

    if overwrite['semantic'] or not os.path.exists(semantic_path):
        semantic = load_semantic(os.path.join(dataset_path, entry['scene_name']),
                                 cam=entry['camera_name'],
                                 frame=entry['frame_id'])
        os.makedirs(os.path.dirname(colored_path), exist_ok=True)
        os.makedirs(os.path.dirname(semantic_path), exist_ok=True)
        save_indexed_png(colored_path, semantic,
                         colormap=HypersimMeta.CLASS_COLORS)
        assert cv2.imwrite(semantic_path, semantic, cv_write_flags)
    else:
        # see instance below
        semantic = None

    # extrinsic camera parameters ----------------------------------------------
    extrinsic_path = os.path.join(destination_path,
                                  entry['split_partition_name'],
                                  HypersimMeta.EXTRINSICS_DIR,
                                  entry['path'].replace('png', 'json'))
    if overwrite['extrinsics'] or not os.path.exists(extrinsic_path):
        extrinsics = load_extrinsics(
            os.path.join(dataset_path, entry['scene_name']),
            cam=entry['camera_name'],
            frame=entry['frame_id']
        )

        os.makedirs(os.path.dirname(extrinsic_path), exist_ok=True)
        with open(extrinsic_path, 'w') as outfile:
            json.dump(extrinsics, outfile, indent=4)
    else:
        # see orientation below
        extrinsics = None

    # instance -----------------------------------------------------------------
    instances_path = os.path.join(destination_path,
                                  entry['split_partition_name'],
                                  HypersimMeta.INSTANCES_DIR,
                                  entry['path'])

    if overwrite['instance'] or not os.path.exists(instances_path):
        instance = load_instance(os.path.join(dataset_path,
                                 entry['scene_name']),
                                 cam=entry['camera_name'],
                                 frame=entry['frame_id'])

        os.makedirs(os.path.dirname(instances_path), exist_ok=True)
        # In hypersim not every instance allways just match to one semantic label
        # We want to ensure that, thats why we combine both information
        if semantic is None:
            raise Exception("Cannot create instances without loaded "
                            "semantic label!")

        # max value of the semantic label is 40 (min. 6 bits) and the max
        # value of instances is 4166 (min. 13 bits), so we need 19 bits in
        # total, an easy way to do it is to just use an image with three
        # channels (3*uint8 = 24 Bit): the first channel can be used for
        # gt (uint8) and the following two can be used for the instance
        # label (uint16)
        instance_img = np.zeros((*instance.shape, 3), np.uint8)
        instance_img[:, :, 0] = semantic
        instance_img[:, :, 1:] = np.expand_dims(instance,
                                                axis=2).view(np.uint8)

        # images get converted to BGR as we write them using OpenCV
        instance_img = cv2.cvtColor(instance_img, cv2.COLOR_RGB2BGR)
        assert cv2.imwrite(instances_path, instance_img, cv_write_flags)

    # 3D bounding boxes --------------------------------------------------------
    boxes_3d_path = os.path.join(destination_path,
                                 entry['split_partition_name'],
                                 HypersimMeta.BOXES_3D_DIR,
                                 entry['path'].replace('png', 'json'))

    if overwrite['boxes_3d+orientations'] or not os.path.exists(boxes_3d_path):
        boxes_3d = load_box_3d(os.path.join(dataset_path, entry['scene_name']),
                               cam=entry['camera_name'],
                               frame=entry['frame_id'])

        boxes_3d_new = {}
        # the loaded box dict has orientations for instances that are part of
        # the scene but not in the current image, we surpress these
        # orientations by only considering the unique ids based on the current
        # image
        # moreover, the datastructure is changed (we extract the box parameters
        # for each instance id)
        for box_idx in np.unique(instance):
            if box_idx == 0:    # void
                continue

            new_value = {}
            for key, value in boxes_3d.items():
                new_value[key] = value[box_idx]

            # Also we need to use the changed instance value
            boxes_3d_new[int(box_idx)] = new_value

        os.makedirs(os.path.dirname(boxes_3d_path), exist_ok=True)
        with open(boxes_3d_path, 'w') as outfile:
            json.dump(boxes_3d_new, outfile, indent=4)

        if extrinsics is None:
            raise Exception("Cannot determine orientation without loading"
                            "extrinsics!")

        # calculate orientation
        cam_rot = Rotation.from_quat((extrinsics['quat_x'],
                                      extrinsics['quat_y'],
                                      extrinsics['quat_z'],
                                      extrinsics['quat_w']))

        orientation_dict = {}
        for key, box in boxes_3d_new.items():
            box_orientation = Rotation.from_matrix(box["orientations"])
            # only take rotation around z-axis and convert to rad
            obj_rot_deg = np.rad2deg(box_orientation.as_rotvec()[2])
            cam_rot_deg = np.rad2deg(cam_rot.as_rotvec()[2])
            obj_rot_in_cam = ((obj_rot_deg-cam_rot_deg)-180) % 360
            orientation_dict[key] = np.deg2rad(obj_rot_in_cam)

        orientation_path = os.path.join(destination_path,
                                        entry['split_partition_name'],
                                        HypersimMeta.ORIENTATIONS_DIR,
                                        entry['path'].replace('png', 'json'))
        os.makedirs(os.path.dirname(orientation_path), exist_ok=True)
        with open(orientation_path, 'w') as outfile:
            json.dump(orientation_dict, outfile, indent=4)

    # scene class --------------------------------------------------------------
    scene_class_path = os.path.join(destination_path,
                                    entry['split_partition_name'],
                                    HypersimMeta.SCENE_CLASS_DIR,
                                    entry['path'].replace('png', 'txt'))
    if overwrite['scene_class'] or not os.path.exists(scene_class_path):
        os.makedirs(os.path.dirname(scene_class_path), exist_ok=True)
        scene_class = entry['scene_class']
        with open(scene_class_path, 'w') as f:
            f.write(scene_class + "\n")


def main():
    # argument parser
    parser = ap.ArgumentParser(description='Prepare Hypersim dataset.')
    parser.add_argument(
        'output_path',
        type=str,
        help="Path where to store dataset."
    )
    parser.add_argument(
        'hypersim_filepath',
        type=str,
        help="Filepath to downloaded (and uncompressed) Hypersim files."
    )
    parser.add_argument(
        '--additional-subsamples',
        type=int,
        nargs='*',
        default=[10],
        help="Create subsampled versions with every N samples of the the "
             "Hypersim dataset as well. Note that the subsampling is applied "
             "across all samples and NOT per scene or camera."
    )
    parser.add_argument(
        '--multiprocessing',
        action="store_true",
        default=False,
        help='Whether multiprocessing should be used.'
    )
    args = parser.parse_args()

    # prevent overwriting already existing outputs
    overwrite = {'rgb': True,
                 'depth': True,
                 'semantic': True,
                 'normal': True,
                 'extrinsics': True,
                 'instance': True,
                 'boxes_3d+orientations': True,
                 'scene_class': True}

    dataset_path = args.hypersim_filepath
    destination_path = args.output_path
    additional_subsamples = args.additional_subsamples

    os.makedirs(destination_path, exist_ok=True)

    # write meta file
    create_or_update_creation_metafile(destination_path)

    cv_write_flags = [cv2.IMWRITE_PNG_COMPRESSION, 9]

    splits_csv_path = os.path.join(dataset_path,
                                   'metadata_images_split_scene_v1.csv')
    if not os.path.exists(splits_csv_path):
        print(f"Downloading splits csv file to: `{splits_csv_path}`")
        download_file(SPLITS_CSV_URL, splits_csv_path,
                      display_progressbar=True)

    split_df = pd.read_csv(splits_csv_path)
    # remove excluded images
    split_df = split_df[split_df['included_in_public_release']]
    split_df.reset_index(drop=True, inplace=True)

    splits = HypersimMeta.SPLITS
    split_df['path'] = split_df["scene_name"] + '/' + split_df["camera_name"]
    split_df['path'] += '/' + split_df["frame_id"].map('{:04d}'.format) + '.png'

    # map val to valid (= HypersimBase.SPLITS[1])
    split_df.loc[split_df['split_partition_name'] == 'val',
                 'split_partition_name'] = 'valid'

    generate_split_txts(split_df, destination_path, splits)
    for subsample in additional_subsamples:
        generate_split_txts(split_df, destination_path, splits,
                            subsample=subsample)

    # create subfolders for each split
    for split in splits:
        for sub_f in [HypersimMeta.RGB_DIR,
                      HypersimMeta.DEPTH_DIR,
                      HypersimMeta.SEMANTIC_DIR,
                      HypersimMeta.SEMANTIC_COLORED_DIR,
                      HypersimMeta.INSTANCES_DIR,
                      HypersimMeta.BOXES_3D_DIR,
                      HypersimMeta.SCENE_CLASS_DIR]:
            os.makedirs(
                os.path.join(destination_path,
                             HypersimMeta.SPLIT_DIRS[split],
                             sub_f),
                exist_ok=True
            )

    # download and prepare data for scene class
    scene_class_csv_path = os.path.join(dataset_path,
                                        'metadata_camera_trajectories.csv')
    if not os.path.exists(scene_class_csv_path):
        print(f"Downloading splits csv file to: `{scene_class_csv_path}`")
        download_file(SCENE_CLASS_URL, scene_class_csv_path,
                      display_progressbar=True)

    scene_class_df = pd.read_csv(scene_class_csv_path)
    scene_class_animation_list = scene_class_df['Animation'].tolist()
    scene_class_animation_list = [re.split("(_cam_[0-9][0-9])", x)[:2]
                                  for x in scene_class_animation_list]
    # remove leading underscore
    scene_class_animation_list = [[a, b[1:]]
                                  for a, b in scene_class_animation_list]
    scene_class_list = scene_class_df['Scene type'].tolist()

    # add scene class column to existing df
    split_df.insert(3, 'scene_class', 'None')
    scene_class_dict = {}
    for idx, (scene_name, camera_name) in enumerate(scene_class_animation_list):
        if scene_name not in scene_class_dict:
            scene_class_dict[scene_name] = {camera_name: scene_class_list[idx]}
        else:
            scene_class_dict[scene_name][camera_name] = scene_class_list[idx]
    # match scene class to split_df
    for idx, row in split_df.iterrows():
        row_scene_name = row['scene_name']
        row_camera_name = row['camera_name']
        scene_class = None
        if row_scene_name not in scene_class_dict:
            raise ValueError()
        if row_camera_name not in scene_class_dict[row_scene_name]:
            raise ValueError()
        split_df.loc[idx, ('scene_class')] = \
            scene_class_dict[row_scene_name][row_camera_name]

    start_time = datetime.datetime.now()
    if not args.multiprocessing:
        for index, entry in tqdm(split_df.iterrows(),
                                 total=len(split_df)):
            process_one_sample(destination_path, dataset_path, overwrite,
                               cv_write_flags, (index, entry))
    else:
        max_workers = os.cpu_count()
        fun_call = functools.partial(process_one_sample,
                                     destination_path,
                                     dataset_path,
                                     overwrite,
                                     cv_write_flags)

        process_map(fun_call, split_df.iterrows(),
                    max_workers=max_workers,
                    total=len(split_df))

    print(f"Preparing Hypersim Dataset was successful. "
          f"Dataset path: {destination_path}. "
          f"It took {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()
