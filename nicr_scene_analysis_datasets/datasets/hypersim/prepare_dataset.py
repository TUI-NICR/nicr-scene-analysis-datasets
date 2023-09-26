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

CAMERA_PARAMETER_CSV_URL = 'https://raw.githubusercontent.com/apple/ml-hypersim/76dbac9b2ee15faf2d677e87db2c0203fc3cc024/' \
                           'contrib/mikeroberts3000/metadata_camera_parameters.csv'

CV_WRITE_FLAGS = (cv2.IMWRITE_PNG_COMPRESSION, 9)


def _parse_args(args=None):
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description='Prepare Hypersim dataset.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path where to store dataset."
    )
    parser.add_argument(
        'hypersim_path',
        type=str,
        help="Path to downloaded (and uncompressed) Hypersim files."
    )
    parser.add_argument(
        '--additional-subsamples',
        type=int,
        nargs='*',
        default=[10],
        help="Create additional versions with every N samples of the the "
             "Hypersim dataset as well. Note that the subsample is applied "
             "across all samples and NOT per scene or camera. However, as each "
             "trajectory comprises 100 samples, a lot of factors are possible "
             "to subsample each trajectory independently."
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=1,
        help='Number of worker processes to spawn.'
    )
    parser.add_argument(
        '--no-tilt-shift-conversion',
        action='store_true',
        default=False,
        help="Disable projecting the data/annotations back to a standard "
             "camera ignoring tilt-shift parameters. Use this argument to "
             "prepare a dataset compatible with < v050)."
    )

    return parser.parse_args(args)


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


def save_pc(filepath, pc):
    height, width = pc.shape[:2]

    file_extension = os.path.splitext(filepath)[1]

    if file_extension == '.npz':
        np.savez_compressed(filepath, pc)

    elif file_extension == '.pcd':

        with open(filepath, 'w') as f:
            # write header
            f.write('VERSION 0.7\n')
            f.write('FIELDS x y z\n')
            f.write('SIZE 4 4 4\n')
            f.write('TYPE f f f\n')
            f.write('COUNT 1 1 1\n')
            f.write(f'WIDTH {width}\n')
            f.write(f'HEIGHT {height}\n')
            f.write('VIEWPOINT 0 0 0 1 0 0 0\n')
            f.write(f'POINTS {height * width}\n')
            f.write('DATA ascii\n')

            # write point data
            for point in pc.reshape(-1, 3):
                f.write(f'{point[0]} {point[1]} {point[2]}\n')
    else:
        raise ValueError('Filetype for point cloud not understood')


def load_pc(scene_path, cam='cam_00', frame=0):
    pc_path = os.path.join(scene_path, 'images', f'scene_{cam}_geometry_hdf5',
                           f'frame.{frame:04d}.position.hdf5')

    assets_to_meters_path = os.path.join(scene_path, '_detail',
                                         'metadata_scene.csv')
    assets_to_meters_csv = pd.read_csv(assets_to_meters_path)
    assets_to_meters = assets_to_meters_csv['parameter_value'][0]

    with h5py.File(pc_path, 'r') as f:
        pc = f['dataset'][:].astype(np.float32) * assets_to_meters

    height, width = pc.shape[:2]

    # load extrinsic as the point cloud is given in world coordinate frame
    extrinsics = load_extrinsics(scene_path, cam, frame)

    rotation = Rotation.from_quat(
        [extrinsics[q] for q in ['quat_x', 'quat_y', 'quat_z', 'quat_w']]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = [extrinsics[t] for t in ['x', 'y', 'z']]

    # flatten point cloud for transformation
    pc = pc.reshape(-1, 3)

    # change to homogenous coordinates
    pc = np.concatenate([pc, np.ones((len(pc), 1))], axis=-1)

    # transform points into camera coordinate frame
    pc = np.linalg.inv(transform) @ pc.T
    pc = pc.T[:, :3]

    pc = pc.reshape(height, width, 3)

    return pc


def compute_point_cloud(scene_path, cam, frame, camera_params):
    # code adapted from:
    # https://github.com/apple/ml-hypersim/blob/main/contrib/mikeroberts3000/jupyter/01_casting_rays_that_match_hypersim_images.ipynb

    # note, we do not use the precomputed point clouds as they are of type
    # float16 and, thus, not precise enough for our purposes

    # distance image (note its the distance along the ray, not the distance to
    # the camera)
    filepath = os.path.join(scene_path, 'images', f'scene_{cam}_geometry_hdf5',
                            f'frame.{frame:04d}.depth_meters.hdf5')
    with h5py.File(filepath, 'r') as f:
        distance_img_meters = f['dataset'][:]

    # camera parameters
    width_pixels = int(camera_params["settings_output_img_width"])
    height_pixels = int(camera_params["settings_output_img_height"])

    # camera matrix
    cp = camera_params
    M_cam_from_uv = np.array(
        [[cp["M_cam_from_uv_00"], cp["M_cam_from_uv_01"], cp["M_cam_from_uv_02"]],
         [cp["M_cam_from_uv_10"], cp["M_cam_from_uv_11"], cp["M_cam_from_uv_12"]],
         [cp["M_cam_from_uv_20"], cp["M_cam_from_uv_21"], cp["M_cam_from_uv_22"]]]
    )

    # compute rays
    u_min = -1.0
    u_max = 1.0
    v_min = -1.0
    v_max = 1.0
    half_du = 0.5 * (u_max - u_min) / width_pixels
    half_dv = 0.5 * (v_max - v_min) / height_pixels
    u, v = np.meshgrid(
        np.linspace(u_min+half_du, u_max-half_du, width_pixels),
        np.linspace(v_min+half_dv, v_max-half_dv, height_pixels)[::-1]
    )
    uvs_2d = np.dstack((u, v, np.ones_like(u)))
    # rays = np.dot(M_cam_from_uv, uvs_2d.reshape(-1, 3).T).T
    rays = np.dot(uvs_2d.reshape(-1, 3), M_cam_from_uv.T)   # (h*w, 3)

    # compute point cloud
    normed_rays = rays / np.linalg.norm(rays, ord=2, axis=-1, keepdims=True)
    points_cam = normed_rays * distance_img_meters.reshape(-1, 1)   # (h*w, 3)

    # flip axes to fix projection side
    points_cam *= np.array([1, -1, -1]).reshape(-1, 3)

    # reshape back to image
    points_cam = points_cam.reshape(height_pixels, width_pixels, 3)

    return points_cam


def load_depth_old(scene_path, cam, frame, correct_focal=True, in_mm=True):
    # do not use this code anymore

    depth_path = os.path.join(scene_path,
                              'images',
                              f'scene_{cam}_geometry_hdf5',
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


def load_normal(scene_path, cam, frame):
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

    # format is xyz
    return normal


def load_semantic(scene_path, cam, frame, orig_mapping=False):
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


def load_instance(scene_path, cam, frame):
    gt_path = os.path.join(scene_path,
                           'images',
                           f'scene_{cam}_geometry_hdf5',
                           f'frame.{frame:04d}.semantic_instance.hdf5')

    with h5py.File(gt_path, 'r') as f:
        gt = f['dataset'][:]

    # Min value is -1 and max value is 4166, so we shift it to store it in an
    # uint16
    assert 0 == np.sum(gt == 0)
    gt[gt == -1] = 0    # note there is no instance with id 0
    gt = gt.astype(np.uint16)

    return gt


def load_box_3d(scene_path, cam, frame):
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

    # the boxes are stored in assets coordinates, we need to convert them to
    # meters
    assets_to_meters_path = os.path.join(scene_path, '_detail',
                                         'metadata_scene.csv')
    assets_to_meters_csv = pd.read_csv(assets_to_meters_path)
    assets_to_meters = assets_to_meters_csv['parameter_value'][0]

    new_extents = []
    for extent in r_dict['extents']:
        new_extents.append([extent[0] * assets_to_meters,
                            extent[1] * assets_to_meters,
                            extent[2] * assets_to_meters])
    r_dict['extents'] = new_extents

    new_positions = []
    for position in r_dict['positions']:
        new_positions.append([position[0] * assets_to_meters,
                              position[1] * assets_to_meters,
                              position[2] * assets_to_meters])
    r_dict['positions'] = new_positions

    return r_dict


def tonemap_image(scene_path, cam, frame):
    # see: https://github.com/apple/ml-hypersim/blob/master/code/python/tools/scene_generate_images_tonemap.py
    rgb_hdf5_path = os.path.join(
        scene_path,
        'images',
        f'scene_{cam}_final_hdf5',
        f'frame.{frame:04d}.color.hdf5'
    )
    render_id_path = os.path.join(
        scene_path,
        'images',
        f'scene_{cam}_geometry_hdf5',
        f'frame.{frame:04d}.render_entity_id.hdf5'
    )

    with h5py.File(rgb_hdf5_path, 'r') as f:
        rgb_color = f['dataset'][:].astype(np.float32)

    with h5py.File(render_id_path, 'r') as f:
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


def load_extrinsics(scene_path, cam, frame):
    # changed in v051:
    # apply 180 degree rotation around x-axis, i.e., flipping y-axis and z-axis
    constant_rot = Rotation.from_euler('zyx', [0, 0, 180], degrees=True)

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
    rotation = Rotation.from_matrix(
        np.dot(camera_orientations, constant_rot.as_matrix())
    )
    quat_x, quat_y, quat_z, quat_w = rotation.as_quat()

    return {
        'x': camera_position[0],
        'y': camera_position[1],
        'z': camera_position[2],
        'quat_x': quat_x,
        'quat_y': quat_y,
        'quat_z': quat_z,
        'quat_w': quat_w
    }


def generate_split_txt_files(split_df: pd.DataFrame, destination_path,
                             splits=None, subsample=1, use_blacklist=True):
    assert subsample > 0, "Undefined behavior for subsample < 1"
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
        # apply subsample
        filepaths = filepaths[::subsample]

        np.savetxt(os.path.join(destination_path, f_name),
                   filepaths,
                   fmt="%s")
        print(f"Created filelist: '{f_name}'")


def _write_image(filepath, img):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not cv2.imwrite(filepath, img, CV_WRITE_FLAGS):
        raise RuntimeError(f"Could not write image: '{filepath}'")


def process_one_sample(
    index_entry,
    destination_path,
    dataset_path,
    camera_params,
    apply_tilt_shift_conversion=True
):
    _, entry = index_entry

    scene_path = os.path.join(dataset_path, entry['scene_name'])

    # process data -------------------------------------------------------------
    # -> extrinsics ------------------------------------------------------------
    extrinsics = load_extrinsics(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )

    # -> rgb image with default tonemapping ------------------------------------
    rgb = tonemap_image(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )

    # -> normal ----------------------------------------------------------------
    normal = load_normal(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )

    # -> semantic --------------------------------------------------------------
    semantic = load_semantic(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )

    # -> instance --------------------------------------------------------------
    instance_raw = load_instance(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )
    # for hypersim, instances may not have a single semantic class,
    # we combine semantic and instance to ensure a unique label,
    # max value of the semantic label is 40 (min. 6 bits) and the maximum value
    # for instances is 4166 (min. 13 bits), so we need 19 bits in total, an
    # easy way to store is to just use an image with three channels
    # (3*uint8 = 24 Bit): the first channel can be used for semantic (uint8)
    # and the following two can be used for the instance label (uint16)
    instance = np.empty((*instance_raw.shape, 3), np.uint8)
    instance[:, :, 0] = semantic
    instance[:, :, 1] = (instance_raw >> 8).astype(np.uint8)
    instance[:, :, 2] = instance_raw.astype(np.uint8)

    # -> scene class -----------------------------------------------------------
    scene_class = entry['scene_class']

    #
    #
    #

    # --------------------------------------------------------------------------
    # convert to standard perspective projection with cx, cy, fx, fy

    # note, Hypersim uses non-standard perspective projection matrices (with
    # tilt-shift photography parameters) in most scenes.
    # as common frameworks, such as MIRA or ROS, do not support this
    # projection, we convert the parameters if possible or project the points
    # back to a standard camera ignoring the tilt-shift parameters
    # note that this is not a perfect conversion and introduces artifacts
    # however, rendering full images with a standard perspective projection
    # requires buying the dataset meshes
    # for more details, see https://github.com/apple/ml-hypersim/issues/24
    if apply_tilt_shift_conversion:
        # get camera parameters ------------------------------------------------
        cp = camera_params.loc[entry['scene_name']]
        width_pixels = int(cp["settings_output_img_width"])
        height_pixels = int(cp["settings_output_img_height"])

        M_cam_from_uv = np.array(
            [[cp["M_cam_from_uv_00"], cp["M_cam_from_uv_01"], cp["M_cam_from_uv_02"]],
             [cp["M_cam_from_uv_10"], cp["M_cam_from_uv_11"], cp["M_cam_from_uv_12"]],
             [cp["M_cam_from_uv_20"], cp["M_cam_from_uv_21"], cp["M_cam_from_uv_22"]]]
        )

        # TODO(dase): is this fine? -> double-check this
        # ensure -1 / 1 for last parameter
        # M_cam_from_uv = M_cam_from_uv / np.abs(cp["M_cam_from_uv_22"])

        # convert parameters to standard perspective projection parameters -----
        # u, v in range [0, 1]
        # default camera origin is at the center of the image
        intrinsics = {
            'fx': (1 / M_cam_from_uv[0, 0]) / 2,
            'fy': (1 / M_cam_from_uv[1, 1]) / 2,
            'cx': 0.5 - 0.5*M_cam_from_uv[0, 2]/M_cam_from_uv[0, 0],
            'cy': 0.5 + 0.5*M_cam_from_uv[1, 2]/M_cam_from_uv[1, 1],
            'a': 1/1000.,     # unit of depth images is mm
            'b': -1
        }
        rgb_intrinsics = {k: v
                          for k, v in intrinsics.items()
                          if k not in ('a', 'b')}
        depth_intrinsics = intrinsics

        # get point cloud ------------------------------------------------------
        # note, we do not use the precomputed point cloud as they is of type
        # float16 and, thus, not precise enough for our purposes (i.e,
        # introduces more artifacts in the back-projection)
        points_cam = compute_point_cloud(
            scene_path,
            cam=entry['camera_name'],
            frame=entry['frame_id'],
            camera_params=cp
        )

        # get depth image as z coordinate from point cloud ---------------------
        depth = points_cam[:, :, -1]    # use z as depth
        depth = depth * 1000    # convert to mm
        depth[depth > np.iinfo('uint16').max] = 0    # clip to 65.535m
        depth = depth.astype('uint16')

        # project points back to standard perspective camera -------------------
        height = height_pixels    # render at same resolution
        width = width_pixels    # render at same resolution
        points_cam_flat = points_cam.reshape((-1, 3))
        points_cam_flat_uv = points_cam_flat / points_cam_flat[:, -1:]  # z = 1
        points_cam_flat_uv = points_cam_flat_uv[:, :2]  # remove z
        points_cam_flat_uv *= np.array([[intrinsics['fx'], intrinsics['fy']]])
        points_cam_flat_uv += np.array([[intrinsics['cx'], intrinsics['cy']]])
        points_cam_flat_uv *= np.array([[width-0.5, height-0.5]])

        # scale to pixels and clip points to image
        points_cam_flat_uv = points_cam_flat_uv.astype('int16')
        np.clip(points_cam_flat_uv[:, 0], 0, width-1,
                out=points_cam_flat_uv[:, 0])
        np.clip(points_cam_flat_uv[:, 1], 0, height-1,
                out=points_cam_flat_uv[:, 1])

        # remove non-unique elements (multiple points may project to same pixel)
        unique_uv, indices, counts = np.unique(points_cam_flat_uv,
                                               axis=0,
                                               return_index=True,
                                               return_counts=True)

        unique_uv = unique_uv[counts == 1]
        indices = indices[counts == 1]

        # create mapped images -------------------------------------------------
        # TODO(dase): do not compute mapping for each sample, it is fixed for
        # all samples of a camera trajectory (and an entire scene)
        rgb_mapped = np.zeros((height, width, 3), dtype='uint8')
        depth_mapped = np.zeros((height, width), dtype='uint16')
        normal_mapped = np.zeros((height, width, 3), dtype='uint8')
        semantic_mapped = np.zeros((height, width), dtype='uint8')
        instance_mapped = np.zeros((height, width, 3), dtype='uint8')

        u, v = points_cam_flat_uv[indices, 0], points_cam_flat_uv[indices, 1]
        rgb_mapped[v, u] = rgb.reshape(-1, 3)[indices]
        depth_mapped[v, u] = depth.reshape(-1)[indices]
        normal_mapped[v, u] = normal.reshape(-1, 3)[indices]
        semantic_mapped[v, u] = semantic.reshape(-1)[indices]
        instance_mapped[v, u] = instance.reshape(-1, 3)[indices]

        rgb = rgb_mapped
        depth = depth_mapped
        normal = normal_mapped
        semantic = semantic_mapped
        instance = instance_mapped

    # --------------------------------------------------------------------------
    else:
        # OLD way (< 0.5.0)
        # note that code below do not allow correct projection into 3D, however,
        # training in 2D may be fine
        # see: https://github.com/apple/ml-hypersim/issues/24
        width_pixels = 1024
        height_pixels = 768
        fov_x = np.pi/3.0
        fov_y = fov_x*(height_pixels/width_pixels)
        f_x = (width_pixels/np.tan(fov_x/2))/2
        f_y = (height_pixels/np.tan(fov_y/2))/2
        intrinsics = {
            'fx': f_x/width_pixels,
            'fy': f_y/height_pixels,
            'cx': 0.5,
            'cy': 0.5,
            'a': 1/1000.,     # unit of depth images is mm
            'b': -1
        }
        rgb_intrinsics = {k: v
                          for k, v in intrinsics.items()
                          if k not in ('a', 'b')}
        depth_intrinsics = intrinsics
        depth = load_depth_old(
            scene_path,
            cam=entry['camera_name'],
            frame=entry['frame_id']
        )
    #
    #
    #

    # -> 3D bounding boxes and orientations ------------------------------------
    boxes_3d_raw = load_box_3d(
        scene_path,
        cam=entry['camera_name'],
        frame=entry['frame_id']
    )

    # the loaded box dict has orientations for instances that are part of the
    # scene but not in the current image, we suppress these orientations by
    # only considering the unique ids based on the current image
    # moreover, the data structure is changed (we extract the box parameters
    # for each instance id)
    boxes_3d = {}
    instance_ids_image = instance[:, :, 1].astype('uint16') << 8
    instance_ids_image += instance[:, :, 2].astype('uint16')

    for box_idx in np.unique(instance_ids_image):
        if box_idx == 0:    # void
            continue

        new_value = {}
        for key, value in boxes_3d_raw.items():
            new_value[key] = value[box_idx]

        # also we need to use the changed instance value
        boxes_3d[int(box_idx)] = new_value

    # calculate orientation
    cam_rot = Rotation.from_quat((extrinsics['quat_x'],
                                  extrinsics['quat_y'],
                                  extrinsics['quat_z'],
                                  extrinsics['quat_w']))
    orientations = {}
    for key, box in boxes_3d.items():
        box_orientation = Rotation.from_matrix(box["orientations"])
        # only take rotation around z-axis and convert to rad
        obj_rot_deg = np.rad2deg(box_orientation.as_rotvec()[2])
        cam_rot_deg = np.rad2deg(cam_rot.as_rotvec()[2])
        obj_rot_in_cam = ((obj_rot_deg-cam_rot_deg)-180) % 360
        orientations[key] = np.deg2rad(obj_rot_in_cam)

    # write stuff to disk ------------------------------------------------------
    # -> extrinsics ------------------------------------------------------------
    extrinsic_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.EXTRINSICS_DIR,
        entry['path'].replace('png', 'json')
    )
    os.makedirs(os.path.dirname(extrinsic_path), exist_ok=True)
    with open(extrinsic_path, 'w') as f:
        json.dump(extrinsics, f, indent=4)

    # -> intrinsics ------------------------------------------------------------
    for intrinsics, dirname in zip((rgb_intrinsics, depth_intrinsics),
                                   (HypersimMeta.RGB_INTRINSICS_DIR,
                                    HypersimMeta.DEPTH_INTRINSICS_DIR)):
        intrinsic_path = os.path.join(
            destination_path,
            entry['split_partition_name'],
            dirname,
            entry['scene_name'],
            f'{entry["camera_name"]}.json'
        )
        os.makedirs(os.path.dirname(intrinsic_path), exist_ok=True)
        with open(intrinsic_path, 'w') as f:
            json.dump(intrinsics, f, indent=4)

    # -> rgb image with default tonemapping ------------------------------------
    rgb_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.RGB_DIR,
        entry['path']
    )
    # image gets converted to BGR as we write it using OpenCV
    _write_image(rgb_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # -> depth -----------------------------------------------------------------
    depth_path = os.path.join(
        destination_path, entry['split_partition_name'],
        HypersimMeta.DEPTH_DIR,
        entry['path']
    )
    _write_image(depth_path, depth)

    # -> normal ----------------------------------------------------------------
    normal_path = os.path.join(
        destination_path, entry['split_partition_name'],
        HypersimMeta.NORMAL_DIR,
        entry['path']
    )
    # convert xyz to zyx as cv2.imwrite will swap it later again
    _write_image(normal_path, cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))

    # -> semantic --------------------------------------------------------------
    semantic_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.SEMANTIC_DIR,
        entry['path']
    )
    semantic_colored_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.SEMANTIC_COLORED_DIR,
        entry['path']
    )
    _write_image(semantic_path, semantic)
    # additionally save a colored version as indexed png
    os.makedirs(os.path.dirname(semantic_colored_path), exist_ok=True)
    save_indexed_png(semantic_colored_path, semantic,
                     colormap=HypersimMeta.CLASS_COLORS)

    # -> instance --------------------------------------------------------------
    instances_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.INSTANCES_DIR,
        entry['path']
    )
    # image gets converted to BGR as we write it using OpenCV
    _write_image(instances_path, cv2.cvtColor(instance, cv2.COLOR_RGB2BGR))
    instance = cv2.cvtColor(instance, cv2.COLOR_RGB2BGR)

    # -> scene class -----------------------------------------------------------
    scene_class_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.SCENE_CLASS_DIR,
        entry['path'].replace('png', 'txt')
    )
    os.makedirs(os.path.dirname(scene_class_path), exist_ok=True)
    with open(scene_class_path, 'w') as f:
        f.write(scene_class + "\n")

    # -> 3D bounding boxes and orientations ------------------------------------
    # write boxes
    boxes_3d_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.BOXES_3D_DIR,
        entry['path'].replace('png', 'json')
    )
    os.makedirs(os.path.dirname(boxes_3d_path), exist_ok=True)
    with open(boxes_3d_path, 'w') as f:
        json.dump(boxes_3d, f, indent=4)

    # write orientations
    orientation_path = os.path.join(
        destination_path,
        entry['split_partition_name'],
        HypersimMeta.ORIENTATIONS_DIR,
        entry['path'].replace('png', 'json')
    )
    os.makedirs(os.path.dirname(orientation_path), exist_ok=True)
    with open(orientation_path, 'w') as f:
        json.dump(orientations, f, indent=4)


def main(args=None):
    # args ---------------------------------------------------------------------
    args = _parse_args(args)
    dataset_path = args.hypersim_path
    destination_path = args.output_path
    additional_subsamples = args.additional_subsamples

    os.makedirs(destination_path, exist_ok=True)

    # write meta file ----------------------------------------------------------
    create_or_update_creation_metafile(
        destination_path,
        additional_subsamples=args.additional_subsamples,
        apply_tilt_shift_conversion=not args.no_tilt_shift_conversion
    )

    # load splits csv file and filter scenes and cameras -----------------------
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

    generate_split_txt_files(split_df, destination_path, splits)
    for subsample in additional_subsamples:
        generate_split_txt_files(split_df, destination_path, splits,
                                 subsample=subsample)

    # create folders for each split -----------------------------------------
    for split in splits:
        for sub_f in [HypersimMeta.RGB_DIR,
                      HypersimMeta.RGB_INTRINSICS_DIR,
                      HypersimMeta.DEPTH_DIR,
                      HypersimMeta.DEPTH_INTRINSICS_DIR,
                      HypersimMeta.EXTRINSICS_DIR,
                      HypersimMeta.SEMANTIC_DIR,
                      HypersimMeta.SEMANTIC_COLORED_DIR,
                      HypersimMeta.INSTANCES_DIR,
                      HypersimMeta.BOXES_3D_DIR,
                      HypersimMeta.ORIENTATIONS_DIR,
                      HypersimMeta.SCENE_CLASS_DIR]:
            os.makedirs(
                os.path.join(destination_path,
                             HypersimMeta.SPLIT_DIRS[split],
                             sub_f),
                exist_ok=True
            )

    # download and prepare data for scene class --------------------------------
    scene_class_csv_path = os.path.join(dataset_path,
                                        'metadata_camera_trajectories.csv')
    if not os.path.exists(scene_class_csv_path):
        print(f"Downloading scene csv file to: `{scene_class_csv_path}`")
        download_file(SCENE_CLASS_URL, scene_class_csv_path,
                      display_progressbar=True)

    scene_class_df = pd.read_csv(scene_class_csv_path)
    scene_class_animation_list = scene_class_df['Animation'].tolist()
    scene_class_animation_list = [
        re.split("(_cam_[0-9][0-9])", x)[:2]
        for x in scene_class_animation_list
    ]
    # remove leading underscore
    scene_class_animation_list = [
        [a, b[1:]]
        for a, b in scene_class_animation_list
    ]
    scene_class_list = scene_class_df['Scene type'].tolist()

    # match scene class to split_df
    scene_class_dict = {}
    for idx, (scene_name, camera_name) in enumerate(scene_class_animation_list):
        if scene_name not in scene_class_dict:
            scene_class_dict[scene_name] = {camera_name: scene_class_list[idx]}
        else:
            scene_class_dict[scene_name][camera_name] = scene_class_list[idx]

    scene_names = split_df['scene_name']
    camera_names = split_df['camera_name']
    scene_classes = [
        scene_class_dict[scene_names[idx]][camera_names[idx]]
        for idx in range(len(split_df))
    ]

    # add scene class column to existing df
    split_df.insert(3, 'scene_class', scene_classes)

    # load splits csv file and filter scenes and cameras -----------------------
    camera_parameters_csv_path = os.path.join(
        dataset_path,
        'metadata_camera_parameters.csv.csv'
    )
    if not os.path.exists(camera_parameters_csv_path):
        print(f"Downloading splits csv file to: `{camera_parameters_csv_path}`")
        download_file(CAMERA_PARAMETER_CSV_URL, camera_parameters_csv_path,
                      display_progressbar=True)
    camera_parameters_df = pd.read_csv(camera_parameters_csv_path,
                                       index_col='scene_name')

    # for testing
    # split_df = split_df[split_df['scene_name'] == 'ai_021_002']

    # process samples ----------------------------------------------------------
    start_time = datetime.datetime.now()
    if args.n_processes == 1:
        for index, entry in tqdm(split_df.iterrows(), total=len(split_df)):
            process_one_sample(
                index_entry=(index, entry),
                destination_path=destination_path,
                dataset_path=dataset_path,
                camera_params=camera_parameters_df,
                apply_tilt_shift_conversion=not args.no_tilt_shift_conversion
            )
    else:
        fun_call = functools.partial(
            process_one_sample,
            destination_path=destination_path,
            dataset_path=dataset_path,
            camera_params=camera_parameters_df,
            apply_tilt_shift_conversion=not args.no_tilt_shift_conversion
        )

        process_map(fun_call, split_df.iterrows(),
                    max_workers=args.n_processes,
                    total=len(split_df))

    print(f"Preparing Hypersim dataset was successful. "
          f"Dataset path: {destination_path}. "
          f"It took {datetime.datetime.now() - start_time}")


if __name__ == '__main__':
    main()
