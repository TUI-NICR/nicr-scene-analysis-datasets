# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>

"""
import argparse as ap
from functools import lru_cache
import json
import os
from pkg_resources import resource_string
import shutil

import cv2
import h5py
from numba import jit
from numba import prange
import numpy as np
from numpy import matlib
import scipy.io
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

from ...utils.img import save_indexed_png
from ...utils.io import download_file
from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from ...utils.io import extract_zip
from .match_nyuv2_instances import NYUv2InstancesMatcher
from ..nyuv2.nyuv2 import NYUv2Meta
from ..nyuv2 import prepare_dataset as nyuv2_prepare_dataset
from .sunrgbd import SUNRGBDMeta


# see: http://rgbd.cs.princeton.edu/ in section Data and Annotation
DATASET_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBD.zip'
DATASET_TOOLBOX_URL = 'http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip'
DATASET_BOX3D_URL = 'https://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat'


TMP_DIR = 'tmp'


def convert_boxes_3d_to_corners(boxes,
                                oversize_factor=1.0,
                                dict_key='corners'):
    # SUNRGB-D stores boxes not in corner notation, we convert them to corner
    # format here to enable easier handling

    for box in boxes:
        box_raw = box['raw']

        basis = np.array(box_raw['basis'])
        coeffs = np.array(box_raw['coeffs'])
        centroid = np.array(box_raw['centroid'])

        # oversize boxes a little (easier to do in SUNRGB-D notation)
        coeffs *= oversize_factor

        # create the output array for boxes in corner notation
        corners = np.zeros((8, 3))

        # the following code is taken from the SUNRGB-D toolbox
        #   SUNRGBDtoolbox/SUNRGBDtoolbox/mBB/get_corners_of_bb3d.m
        # and adapted to python - it is used to create the 3d box corners
        # from the basis and coeffs
        idx = np.argsort(np.abs(basis[:, 0]))[::-1]
        basis = basis[idx, :]
        coeffs = coeffs[idx]

        idx = np.argsort(np.abs(basis[1:2, 1]))[::-1]
        if idx[0] == 2:
            basis[1:2, :] = np.flip(basis[1:2, :], 0)
            coeffs[1:2] = np.flip(coeffs[1:2], 1)

        basis = flip_towards_viewer(basis, matlib.repmat(centroid, 3, 1))

        coeffs = np.abs(coeffs)

        # calculate the corners
        corners[0, :] = -basis[0, :] * coeffs[0] \
            + basis[1, :] * coeffs[1] \
            + basis[2, :] * coeffs[2]

        corners[1, :] = basis[0, :] * coeffs[0] \
            + basis[1, :] * coeffs[1] \
            + basis[2, :] * coeffs[2]

        corners[2, :] = basis[0, :] * coeffs[0]  \
            + -basis[1, :] * coeffs[1] \
            + basis[2, :] * coeffs[2]

        corners[3, :] = -basis[0, :] * coeffs[0] \
            + -basis[1, :] * coeffs[1] \
            + basis[2, :] * coeffs[2]

        corners[4, :] = -basis[0, :] * coeffs[0] \
            + basis[1, :] * coeffs[1] \
            + -basis[2, :] * coeffs[2]

        corners[5, :] = basis[0, :] * coeffs[0] \
            + basis[1, :] * coeffs[1] \
            + -basis[2, :] * coeffs[2]

        corners[6, :] = basis[0, :] * coeffs[0] \
            + -basis[1, :] * coeffs[1] \
            + -basis[2, :] * coeffs[2]

        corners[7, :] = -basis[0, :] * coeffs[0] \
            + -basis[1, :] * coeffs[1] \
            + -basis[2, :] * coeffs[2]

        corners += matlib.repmat(centroid, 8, 1)

        # corners above are given as:
        #  7 -------- 6
        # /|         /|
        # 3 -------- 2 .
        # | |        | |
        # . 4 -------- 5
        # |/         |/
        # 0 -------- 1

        # reorder corners to match the following order:
        #  6 -------- 7
        # /|         /|
        # 4 -------- 5 .
        # | |        | |
        # . 2 -------- 3
        # |/         |/
        # 0 -------- 1

        corners = corners[[0, 1, 4, 5, 3, 2, 7, 6]]
        box[dict_key] = corners.tolist()

    # note, the box list is also modified in place
    return boxes


def flip_towards_viewer(normal, points):
    # This code is taken from the SUNRGB-D toolbox:
    #   SUNRGBDtoolbox/SUNRGBDtoolbox/mBB/get_corners_of_bb3d.m
    dist = np.sqrt(np.sum(points**2, 1))
    points = points / matlib.repmat(dist, 1, 3).reshape(3, 3)
    proj = sum(points*normal, 1)
    flip = proj > 0
    normal[flip, :] = -normal[flip, :]
    return normal


def load_nyuv2_894_to_40_mapping():
    # load class mapping from nyuv2 864 to 40 classes (without void)
    class_mapping = scipy.io.loadmat(
        nyuv2_prepare_dataset.CLASSES_40_FILEPATH
    )
    class_mapping = class_mapping['mapClass'][0]

    # prepend void class
    class_mapping_with_void = np.concatenate([[0], class_mapping])

    # load class names
    class_names_40_with_void = NYUv2Meta.SEMANTIC_LABEL_LIST_40.class_names
    class_names_894_with_void = NYUv2Meta.SEMANTIC_LABEL_LIST_894.class_names

    # return mapping as dict: 894_classname: (40_idx, 40_classname)
    return {
        name_894: (class_mapping_with_void[i],
                   class_names_40_with_void[class_mapping_with_void[i]])
        for i, name_894 in enumerate(class_names_894_with_void)
    }


def load_additional_bounding_box_to_nyuv2_40_mapping():
    # There are bounding box classes which are not part of the original
    # 894 NYUv2 classes. We additionally map them to the 40 classes.
    # The mapping was done by hand.
    # For example the classes 'fridge' and 'frige' (typo) are mapped to
    # 'refrigerator'.
    # This additional mapping was only done for box classes which
    # appear more than 10 times in the dataset.
    additional_mapping = json.loads(
        resource_string(__name__, 'nyu_additional_class_mapping.json')
    )

    # load class names
    class_names_40_with_void = NYUv2Meta.SEMANTIC_LABEL_LIST_40.class_names

    # return mapping as dict: additional_classname: (40_idx, 40_classname)
    return {
        name: (idx_40, class_names_40_with_void[idx_40])
        for name, idx_40 in additional_mapping.items()
    }


@lru_cache
def load_bounding_box_class_to_nyuv2_40_mapping():
    # mapping for bounding box class names based on the nyuv2 894 classes
    nyuv2_894_to_40_mapping = load_nyuv2_894_to_40_mapping()

    # additional mapping for bounding box class names which are not part of
    # 894 classes
    additional_mapping = load_additional_bounding_box_to_nyuv2_40_mapping()

    assert all(k not in nyuv2_894_to_40_mapping for k in additional_mapping)

    return {**nyuv2_894_to_40_mapping, **additional_mapping}


def map_bounding_box_classes_to_nyu40_classes(boxes):
    # load class mapping from nyuv2 void+864 to void+40 classes
    class_mapping = load_bounding_box_class_to_nyuv2_40_mapping()

    for box in boxes:
        # try to map box class (use void class if no valid mapping is found)
        mapped_idx, mapped_name = class_mapping.get(
            box['raw']['classname'],
            class_mapping['void']    # default mapping
        )
        # add information to box dict
        box['semantic_40_class'] = mapped_name
        box['semantic_40_class_idx'] = int(mapped_idx)

    # note, the box list is also modified in place
    return boxes


def compute_point_cloud(depth, intrinsics, extrinsics):
    # depth is encoded in only 13 of the 16 bits, i.e., bits 3-15 store the
    # actual depth information
    # see: http://velastin.dynu.com/G3D/G3D.html:
    #    The depth information was also mapped to the colour coordinate
    #    space and stored in a 16-bit greyscale. The 16-bits of depth
    #    data contains 13 bits for depth data and 3 bits to identify
    #    the player.

    # see: https://social.msdn.microsoft.com/Forums/en-US/3fe21ce5-4b75-4b31-b73d-2ff48adfdf52/kinect-uses-12-bits-or-13-bits-for-depth-data?forum=kinectsdk

    # original code from toolbox:
    # -> SUNRGBDtoolbox/SUNRGBDtoolbox/readData/read3dPoints.m:
    #     function [rgb,points3d,depthInpaint,imsize]=read3dPoints(data)
    #             depthVis = imread(data.depthpath);
    #             imsize = size(depthVis);
    #             depthInpaint = bitor(bitshift(depthVis,-3), bitshift(depthVis,16-3));
    #             depthInpaint = single(depthInpaint)/1000;
    #             depthInpaint(depthInpaint >8)=8;
    #             [rgb,points3d]=read_3d_pts_general(depthInpaint,data.K,size(depthInpaint),data.rgbpath);
    #             points3d = (data.Rtilt*points3d')';

    # NOTE:
    # we only apply the shift to the right by 3 bits to get rid of the
    # lowest three bits and then clip to 8m; in the toolbox code above, the
    # lowest three bits are added again to the highest bits, and then the
    # depth values are clipped to 8000 (=8m); we do not know the exact
    # reason for the first step as subsequent clipping to 8000 again
    # excludes the highest 3 bits
    depth_ = np.right_shift(depth, 3)
    depth_[depth_ > 8000] = 8000
    depth_scale = 0.001
    depth_meters = depth_.astype('float64') * depth_scale
    height, width = depth.shape

    # compute point cloud based on depth image
    # depth value is valid as long as it is > 0 and <= maxdepth
    focal_length = (intrinsics['fx']*width, intrinsics['fy'] * height)
    principal_point = (intrinsics['cx']*width, intrinsics['cy']*height)
    pixel_h_map, pixel_w_map = np.meshgrid(np.arange(height),
                                           np.arange(width),
                                           indexing='ij')

    points = np.zeros((height, width, 3), dtype='float64')
    x_map = (pixel_w_map - principal_point[0])*depth_meters/focal_length[0]
    y_map = (pixel_h_map - principal_point[1])*depth_meters/focal_length[1]
    points[..., 0] = x_map
    points[..., 1] = y_map
    points[..., 2] = depth_meters

    # compute transformation (rotation) matrix from extrinsic parameters
    rotation_extrinsic = R.from_quat(
        [extrinsics['quat_x'], extrinsics['quat_y'], extrinsics['quat_z'],
         extrinsics['quat_w']]
    )
    # there is no translation in the extrinsic parameters
    assert extrinsics['x'] == extrinsics['y'] == extrinsics['z'] == 0

    # create rotation matrix for transformation from camera to world, i.e.,
    # rotation around x-axis by 90 degrees
    rotation_world_to_cam = R.from_rotvec(np.array([np.pi/2, 0, 0]))

    # combine rotation matrices
    rotation = rotation_world_to_cam.inv() * rotation_extrinsic

    # transform point cloud to world coordinates
    points = rotation.apply(points.reshape(-1, 3))

    # show point cloud with open3d
    # import open3d as o3d
    # pc = o3d.geometry.PointCloud()
    # pc.points = o3d.utility.Vector3dVector(points)    # must be float64
    # visualizer = o3d.visualization.VisualizerWithKeyCallback()
    # visualizer.create_window()
    # visualizer.add_geometry(pc)
    # visualizer.add_geometry(
    #     o3d.geometry.TriangleMesh.create_coordinate_frame()
    # )
    # visualizer.run()

    return points.reshape(height, width, 3)


@jit(nopython=True)
def check_direction(unit, point, dot_min, dot_max):
    dot_point = np.dot(point, unit)
    if (dot_max <= dot_point and dot_point <= dot_min) \
       or (dot_max >= dot_point and dot_point >= dot_min):
        return True
    return False


@jit(nopython=True, parallel=False)
def create_box_mask(pc, box_corners):
    # get mask for points inside the oriented box
    # see: https://math.stackexchange.com/q/1472049

    p1p2 = box_corners[0] - box_corners[4]
    p1p4 = box_corners[0] - box_corners[1]
    p1p5 = box_corners[0] - box_corners[2]

    u = np.cross(p1p4, p1p5)
    v = np.cross(p1p2, p1p5)
    w = np.cross(p1p2, p1p4)

    dot_min_u = np.dot(box_corners[0], u)
    dot_min_v = np.dot(box_corners[0], v)
    dot_min_w = np.dot(box_corners[0], w)

    dot_max_u = np.dot(box_corners[4], u)
    dot_max_v = np.dot(box_corners[1], v)
    dot_max_w = np.dot(box_corners[2], w)

    # check points
    n = pc.shape[0]
    mask = np.zeros(n, dtype=np.bool_)

    for i in prange(n):
        point = pc[i].astype(np.float64)

        if not check_direction(u, point, dot_min_u, dot_max_u):
            continue
        if not check_direction(v, point, dot_min_v, dot_max_v):
            continue
        if not check_direction(w, point, dot_min_w, dot_max_w):
            continue

        # point is in box
        mask[i] = True

    return mask


@lru_cache
def get_matching_class_ids(class_idx):
    # some NYUv2 40 classes are not labeled consistently, we try to handle
    # this with this function by defining a matching NYUv2 40 classes

    # we use the class mapping from the NYUv2 40 classes to the NYUv2 13
    # classes (see below) to derive some matching classes:
    # 0: void -> 0: void                20: floor mat -> 5: floor
    # 1: wall -> 12: wall               21: clothes -> 7: objects
    # 2: floor -> 5: floor              22: ceiling -> 3: ceiling
    # 3: cabinet -> 6: furniture        23: books -> 2: books
    # 4: bed -> 1: bed                  24: refrigerator -> 6: furniture
    # 5: chair -> 4: chair              25: television -> 11: tv
    # 6: sofa -> 9: sofa                26: paper -> 7: objects
    # 7: table -> 10: table             27: towel -> 7: objects
    # 8: door -> 12: wall               28: shower curtain -> 7: objects
    # 9: window -> 13: window           29: box -> 7: objects
    # 10: bookshelf -> 6: furniture     30: whiteboard -> 7: objects
    # 11: picture -> 8: picture         31: person -> 7: objects
    # 12: counter -> 6: furniture       32: night stand -> 6: furniture
    # 13: blinds -> 13: window          33: toilet -> 7: objects
    # 14: desk -> 10: table             34: sink -> 7: objects
    # 15: shelves -> 6: furniture       35: lamp -> 7: objects
    # 16: curtain -> 13: window         36: bathtub -> 7: objects
    # 17: dresser -> 6: furniture       37: bag -> 7: objects
    # 18: pillow -> 7: objects          38: otherstructure -> 7: objects
    # 19: mirror -> 7: objects          39: otherfurniture -> 6: furniture
    # 20: floor mat -> 5: floor         40: otherprop -> 7: objects
    classes_13 = scipy.io.loadmat(
        nyuv2_prepare_dataset.CLASSES_13_FILEPATH
    )['classMapping13'][0][0]
    mapping_40_to_13 = np.concatenate([[0], classes_13[0][0]])

    # furniture classes
    furniture_ids = np.where(mapping_40_to_13 == 6)[0]
    if class_idx in furniture_ids:
        return furniture_ids.tolist()

    # objects classes
    objects_ids = np.where(mapping_40_to_13 == 7)[0]
    if class_idx in objects_ids:
        return objects_ids.tolist()

    # objects classes
    table_desk_ids = np.where(mapping_40_to_13 == 10)[0]
    if class_idx in table_desk_ids:
        return table_desk_ids.tolist()

    # furthermore, we define some matches by hand

    def get_idx(name):
        return NYUv2Meta.SEMANTIC_LABEL_LIST_40.index(name)

    # handle chair <-> sofa
    if class_idx in [get_idx('chair'), get_idx('sofa')]:
        return [get_idx('chair'), get_idx('sofa')]

    # handle books -> bookshelf
    if class_idx in [get_idx('books')]:
        return furniture_ids.tolist()

    return [class_idx]    # matches at least itself


def create_instances(
    pc,
    boxes,
    semantic,
    classes_to_ignore=(0,),    # ignore void pixel during creation
    point_threshold=0.2,        # percentage of points in box that must match
):
    if 0 == len(boxes):
        # we do not have any boxes, so return an empty instance image
        instance = np.zeros_like(semantic, dtype=np.uint16)
        return instance, boxes

    # consider only relevant points / entries (speeds up processing)
    # furthermore, use flattened arrays to ease indices handling
    semantic_flat = semantic.flatten()
    pc_flat = pc.reshape(-1, 3)
    semantic_mask = np.isin(semantic_flat, classes_to_ignore,
                            invert=True)
    indices, = np.nonzero(semantic_mask)
    pc_masked = pc_flat[indices]

    # process each box
    instance = np.zeros_like(semantic_mask, dtype=np.uint16)
    for i, box in enumerate(boxes, start=1):    # 0 is no instance
        if box['semantic_40_class_idx'] == 0:
            # skip boxes with void label (void after mapping the box class)
            continue

        # get mask for points inside the oriented box
        # note, we use oversized boxes here (see box dict creation)
        box_mask = create_box_mask(pc_masked,
                                   np.array(box['corners_oversized']))

        # restrict relevant indices to points inside the box
        indices_box = indices[box_mask]

        # get number of (valid) points inside the box
        n_points = len(indices_box)

        if 0 == n_points:
            # no valid point at all inside the box, do not consider box
            continue

        # get semantic for each point inside the box
        semantic_box = semantic_flat[indices_box]

        # get semantic class distribution of box
        class_ids, counts = np.unique(semantic_box, return_counts=True)

        # order by count descending to emphasize the most frequent class
        order = np.argsort(counts)[::-1]
        class_ids = class_ids[order]
        counts = counts[order]

        # DEBUG: print output of "get_matching_class_ids()"
        # for i in range(41):
        #     matching_classes = [
        #         NYUv2Meta.SEMANTIC_LABEL_LIST_40.class_names[j]
        #         for j in get_matching_class_ids(i)
        #     ]
        #     print(f"{i}: {NYUv2Meta.SEMANTIC_LABEL_LIST_40.class_names[i]} "
        #           f"-> {matching_classes}")

        matching_mask = np.isin(
            class_ids,
            get_matching_class_ids(box['semantic_40_class_idx'])
        )
        class_ids = class_ids[matching_mask]
        counts = counts[matching_mask]
        for class_idx, count in zip(class_ids, counts):
            if count/n_points >= point_threshold:
                # assign instance id to points inside the box that have current
                # semantic class
                indices_instance = indices_box[semantic_box == class_idx]
                instance[indices_instance] = i

                # update instance id in box dict
                box['instance_id'] = i
                # also store the semantic class that was used produced the
                # matching, as it might be different from
                # semantic_40_class_idx due to get_matching_class_ids()
                box['instance_semantic_40_class_idx'] = int(class_idx)
                box['instance_semantic_40_class'] = \
                    NYUv2Meta.SEMANTIC_LABEL_LIST_40[int(class_idx)].class_name

                # we found a match, skip remaining candidates
                break

    # note, the box list is also modified in place
    return instance.reshape(semantic.shape), boxes


def main(args=None):
    # argument parser
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description='Prepare SUNRGB-D dataset.'
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path where to store dataset."
    )
    parser.add_argument(
        '--toolbox-filepath',
        default=None,
        help="Filepath to SUNRGB-D toolbox file."
    )
    parser.add_argument(
        '--data-filepath',
        default=None,
        help="Filepath to SUNRGB-D data zip file."
    )
    parser.add_argument(
        '--box-filepath',
        default=None,
        help="Filepath to SUNRGB-D 3d bounding boxes."
    )
    parser.add_argument(
        '--create-instances',
        default=False,
        action="store_true",
        help="Whether the instance masks should be created by matching "
             "3d boxes with point clouds."
    )
    parser.add_argument(
        '--instances-version',
        type=str,
        default='panopticndt',
        choices=SUNRGBDMeta.INSTANCE_VERSIONS,
        help="Version of instance annotations to extract. Over time, we have "
             "created two versions of SUNRGB-D with instance annotations "
             "extracted from annotated 3d boxes: 'emsanet': this initial "
             "version was created for training the original EMSANet - see "
             "IJCNN 2022 paper; 'panopticndt': referes to a revised version "
             "that was created along with the work for PanopticNDT - see "
             "IROS 2023 paper, it refines large parts of the instance "
             "extraction (see changelog for v0.6.0 of this package)."
    )
    parser.add_argument(
        '--copy-instances-from-nyuv2',
        default=False,
        action="store_true",
        help="Whether instances and orientations should copied from (already "
             "prepared!) NYUv2 dataset."
    )
    parser.add_argument(
        '--nyuv2-path',
        default="",
        type=str,
        help="Path to NYUv2 dataset for matching."
    )

    args_ = parser.parse_args(args)

    # version switch
    if 'emsanet' == args_.instances_version:
        # switch to (legacy) emsanet instances version

        from .legacy_emsanet_version.prepare_dataset import main as main_legacy

        return main_legacy(args)
    else:
        # continue with this file/version
        args = args_

    # output path
    output_path = os.path.expanduser(args.output_path)
    create_dir(output_path)

    # path where to store stuff during creation
    tmp_path = os.path.join(output_path, TMP_DIR)
    create_dir(tmp_path)

    # update or write metafile
    create_or_update_creation_metafile(
        output_path,
        additional_meta={'instances_version': args.instances_version, }
    )

    # download and extract data
    # toolbox
    if args.toolbox_filepath is None:
        # we need to download file
        zip_file_path = os.path.join(tmp_path, 'SUNRGBDtoolbox.zip')
        download_file(DATASET_TOOLBOX_URL, zip_file_path,
                      display_progressbar=True)
    else:
        zip_file_path = args.toolbox_filepath
    print(f"Extracting toolbox from: '{zip_file_path}' to '{tmp_path}'")
    extract_zip(zip_file_path, tmp_path)

    # data
    if args.data_filepath is None:
        zip_file_path = os.path.join(tmp_path, 'SUNRGBD.zip')
        download_file(DATASET_URL, zip_file_path, display_progressbar=True)
    else:
        zip_file_path = args.data_filepath
    print(f"Extracting images from '{zip_file_path}' to '{tmp_path}'")
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

    # semantic stuff classes (used for creating instance segmentation from
    # 3d boxes later)
    nyuv2_stuff_class_ids = np.where(
        [not label.is_thing for label in NYUv2Meta.SEMANTIC_LABEL_LIST_40]
    )

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

        # Depth images --------------------------------------------------------
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
        rgb_path = os.path.join(output_path, split_dir, SUNRGBDMeta.RGB_DIR,
                                cam_path, f'{i:05d}.jpg')
        create_dir(os.path.dirname(rgb_path))
        os.replace(rgb_path_tmp, rgb_path)

        # Semantic label ------------------------------------------------------
        # note, we store the labels in SUNRGBD and NYUv2 colormap
        semantic_path = os.path.join(output_path, split_dir,
                                     SUNRGBDMeta.SEMANTIC_DIR, cam_path,
                                     f'{i:05d}.png')
        create_dir(os.path.dirname(semantic_path))
        semantic = np.ascontiguousarray(
            SUNRGBD2Dseg[seglabel[i][0]][:].transpose(1, 0),
            dtype=np.uint8
        )   # force c-contiguous for later proccessing
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

        # Scene class ---------------------------------------------------------
        scene_class_path_tmp = os.path.join(data_path, sample_path,
                                            'scene.txt')
        scene_class_path = os.path.join(output_path,
                                        split_dir,
                                        SUNRGBDMeta.SCENE_CLASS_DIR,
                                        cam_path,
                                        f'{i:05d}.txt')

        create_dir(os.path.dirname(scene_class_path))
        os.replace(scene_class_path_tmp, scene_class_path)

        # Extrinsics ----------------------------------------------------------
        extrinsics_path = os.path.join(output_path, split_dir,
                                       SUNRGBDMeta.EXTRINSICS_DIR, cam_path,
                                       f'{i:05d}.json')
        create_dir(os.path.dirname(extrinsics_path))

        extrinsic = meta_3d.anno_extrinsics.tolist()
        extrinsic_quat = R.from_matrix(extrinsic).as_quat()
        quat_x, quat_y, quat_z, quat_w = extrinsic_quat
        extrinsics = {
            'x': 0,
            'y': 0,
            'z': 0,
            'quat_x': quat_x,
            'quat_y': quat_y,
            'quat_z': quat_z,
            'quat_w': quat_w
        }

        with open(extrinsics_path, 'w') as f:
            json.dump(extrinsics, f, indent=4)

        # Intrinsics ----------------------------------------------------------
        intrinsics_path = os.path.join(output_path, split_dir,
                                       SUNRGBDMeta.INTRINSICS_DIR, cam_path,
                                       f'{i:05d}.json')
        create_dir(os.path.dirname(intrinsics_path))

        # load intrinsics
        intrinsics_path_tmp = os.path.join(data_path, sample_path,
                                           'intrinsics.txt')

        with open(intrinsics_path_tmp, 'r') as f:
            intrinsics_raw = f.read()
        intrinsics_raw = intrinsics_raw.replace("\n", " ").split(" ")[:-1]
        intrinsics = {
            'fx': float(intrinsics_raw[0]),
            'fy': float(intrinsics_raw[4]),
            'cx': float(intrinsics_raw[2]),
            'cy': float(intrinsics_raw[5])
        }

        # normalize and write to file
        h, w = semantic.shape
        normalized_intrinsics = {
            'fx': intrinsics['fx'] / w,
            'fy': intrinsics['fy'] / h,
            'cx': intrinsics['cx'] / w,
            'cy': intrinsics['cy'] / h
        }
        with open(intrinsics_path, 'w') as f:
            json.dump(normalized_intrinsics, f, indent=4)

        # 3D boxes and orientations -------------------------------------------
        boxes_3d_path = os.path.join(output_path, split_dir,
                                     SUNRGBDMeta.BOXES_PANOPTICNDT_DIR,
                                     cam_path,
                                     f'{i:05d}.json')
        create_dir(os.path.dirname(boxes_3d_path))

        # save all box information
        boxes_3d = []
        if not isinstance(meta_3d.groundtruth3DBB, np.ndarray):
            meta_3d.groundtruth3DBB = [meta_3d.groundtruth3DBB]

        # extract all data for the 3d boxes from the mat file
        for box in meta_3d.groundtruth3DBB:
            box_dict = {
                'raw': {    # raw information from SUNRGBD
                    'basis': box.basis.tolist(),
                    'centroid': box.centroid.tolist(),
                    'classname': box.classname,
                    'coeffs': box.coeffs.tolist(),
                    'orientation': box.orientation.tolist()
                },
                'instance_id': 0,   # no instance (id not available)
                'instance_semantic_40_class_idx': 0,  # void
                'instance_semantic_40_class':
                    NYUv2Meta.SEMANTIC_LABEL_LIST_40[0].class_name
            }
            boxes_3d.append(box_dict)

        # convert boxes to corner notation (8 corners per box)
        # note, box dicts in boxes_3d are modified inplace
        convert_boxes_3d_to_corners(
            boxes=boxes_3d,
            oversize_factor=1.0,
            dict_key='corners'
        )

        # create another version in corner notation but with an oversize
        # factor of 1.15, i.e., enlarge boxes by 15%, we observed that this
        # leads to better instance masks for instance segmentation
        # note, box dicts in boxes_3d are modified inplace
        convert_boxes_3d_to_corners(
            boxes=boxes_3d,
            oversize_factor=1.15,
            dict_key='corners_oversized'
        )

        # map the original box class to the 40 semantic classes of NYUv2, which
        # are same as SUNRGB-D except for the last 3 classes that are omitted
        # in SUNRGBD
        # note, box dicts in boxes_3d are modified inplace
        map_bounding_box_classes_to_nyu40_classes(boxes_3d)

        # save 3d boxes
        with open(boxes_3d_path, 'w') as f:
            json.dump(boxes_3d, f, indent=4)

        # Instances -----------------------------------------------------------
        if args.create_instances:
            instance_path = os.path.join(output_path,
                                         split_dir,
                                         SUNRGBDMeta.INSTANCES_PANOPTICNDT_DIR,
                                         cam_path,
                                         f'{i:05d}.png')

            if len(boxes_3d) > 0:
                # load depth image
                depth = cv2.imread(depth_bfx_path, cv2.IMREAD_UNCHANGED)

                # compute point cloud (ij corresponding to depth image)
                pc = compute_point_cloud(depth,
                                         normalized_intrinsics,
                                         extrinsics)
                # note, this also adds instance ids to boxes_3d
                instance_img, _ = create_instances(
                    pc=pc,
                    boxes=boxes_3d,
                    semantic=semantic,
                    classes_to_ignore=nyuv2_stuff_class_ids     # 0, 1, 2, 22
                )
            else:
                # if no boxes are present, skip computing point cloud and
                # instance segmentation
                instance_img = np.zeros_like(semantic, dtype=np.uint16)

            # DEBUG: print boxes
            # print(i, f'{cam_path}/{i:05d}')
            # for box_i, box in enumerate(boxes_3d):
            #     print(
            #         f"{box_i}, {box['raw']['classname']} -> "
            #         f"{box['semantic_40_class']} -> {box['instance_id']}, "
            #         f"semantic: {box['instance_semantic_40_class']} "
            #     )

            # write instance image
            create_dir(os.path.dirname(instance_path))
            cv2.imwrite(instance_path, instance_img)

            # save 3d boxes again (instance id might have changed)
            with open(boxes_3d_path, 'w') as f:
                json.dump(boxes_3d, f, indent=4)

        # Orientations --------------------------------------------------------
        if args.create_instances or instance_img is not None:
            orientations_path = os.path.join(
                output_path, split_dir,
                SUNRGBDMeta.ORIENTATIONS_PANOPTICNDT_DIR,
                cam_path, f'{i:05d}.json'
            )
            create_dir(os.path.dirname(orientations_path))

            # compute orientations
            orientations_dict = {}
            for box in boxes_3d:
                if box['instance_id'] == 0:
                    # no instance was created for this box, skip it
                    continue

                # convert angle
                # from: https://rgbd.cs.princeton.edu/supp.pdf:
                #   If the object has a natural orientation (such as a chair),
                #   then we ask the oDesk worker to begin from the correct
                #   “front” side; if it does not (such as a round table),
                #   then they can begin from any side they choose.

                # the orientation of a box is given as normalized vector in
                # the plane below:
                #   y
                #   |
                #   . __ x
                # however, we want to have the angle pointing towards the
                # camera, e.g., for chairs (vector from seat back to seat
                # surface):
                # -> 0°: seat back parallel to camera plane, vector
                #        pointing towards camera
                # -> 90°: seat back orthogonal to camera plane, vector pointing
                #         towards the right
                # so we need to rotate the orientation vector by 90° counter-
                # clockwise, i.e., swap x/y and negate y:
                #       x
                #       |
                #  y __ .
                x, y, _ = box['raw']['orientation']
                y_new = x
                x_new = -y
                angle_rad = np.arctan2(y_new, x_new)
                angle_rad = (angle_rad+(2*np.pi)) % (2*np.pi)   # -> [0, 2pi]

                orientations_dict[box['instance_id']] = angle_rad

            # write orientation dict
            with open(orientations_path, 'w') as f:
                json.dump(orientations_dict, f, indent=4)

        # Other stuff ---------------------------------------------------------
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
