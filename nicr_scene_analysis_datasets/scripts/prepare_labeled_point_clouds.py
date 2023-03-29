# -*- coding: utf-8 -*-
"""
.. codeauthor:: Benedict Stephan <benedict.stephan@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
from collections import defaultdict
import os
import os.path as osp

import cv2
import numpy as np
import open3d as o3d
import plyfile
import tqdm
from scipy.spatial.transform import Rotation as R

from .. import get_dataset_class
from .. import KNOWN_DATASETS
from .. import ScanNet


def _parse_args():
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description=(
            "Create labeled point clouds with color, semantic, and instance "
            "information for a given dataset. The output is stored as ply "
            "file similar to ScanNet benchmark format. The point clouds are "
            "created by applying a voxel grid filter to the concatenated point "
            "cloud of each camera trajectory."
        )
    )
    parser.add_argument(
        'dataset',
        type=str,
        choices=KNOWN_DATASETS,
        help="Name of the dataset for which to compute the point clouds.",
    )
    parser.add_argument(
        'dataset_path',
        type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path to the output directory."
    )
    parser.add_argument(
        '--split',
        type=str,
        choices=('train', 'valid', 'test'),
        default='train',
        help="Split to use."
    )
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=0.01,
        help="Voxel size in meter."
    )
    parser.add_argument(
        '--max-depth',
        type=float,
        default=20,
        help="Maximum value (in meter) for valid depth measurements. Depth "
             "values larger than this value will be ignored."
    )
    parser.add_argument(
        '--shift',
        type=int,
        default=(1 << 16),
        help="Shift to be used for fusing semantic and instance ids as: "
             "semantic*shift + instance. Use 1000 for ScanNet benchmark. "
             "However, note that 1000 is too small for Hypersim as it has "
             "more than 4k instances."
    )
    parser.add_argument(
        '--scannet-map-to-benchmark',
        action="store_true",
        help="Whether to map semantic classes to benchmark format."
    )
    parser.add_argument(
        '--write-scannet-label',
        action="store_true",
        help="If set, ground-truth label files that can be used for ScanNet "
             "benchmarking scripts will be written."
    )

    return parser.parse_args()


def _preprocess_sample(sample, args):
    if 'scannet' == args.dataset:
        # for ScanNet, we might need to map the semantic labels to the correct
        # subset of classes used for benchmarks
        # in addition, we must resize depth to the same size as rgb

        if args.scannet_map_to_benchmark:
            # NOTE: we currently only support the ScanNet benchmark, ScanNet200
            # benchmark would mean to map from 200 to tsv ids, which exceeds the
            # current semantic class limit of 254 (uint8)

            mapping = ScanNet.SEMANTIC_CLASSES_20_MAPPING_TO_BENCHMARK  # w/ void
            mapping = np.array(list(mapping.values()), dtype=np.uint8)
            sample['semantic'] = mapping[sample['semantic']]

        # resize depth to rgb size
        h, w, _ = sample['rgb'].shape
        depth_img = cv2.resize(sample['depth'], (w, h),
                               interpolation=cv2.INTER_NEAREST)
        sample['depth'] = depth_img

    return sample


def _compute_labeled_point_cloud(sample, max_depth):
    # extract sample
    extrinsics = sample['extrinsics']
    depth = sample['depth'].astype('float32')
    color = sample['rgb']
    semantic = sample['semantic']
    instance = sample['instance']
    height, width = sample['depth'].shape

    # create empty point cloud
    pc = o3d.geometry.PointCloud()

    # compute point cloud based on depth image
    # depth value is valid as long as it is > 0 and <= maxdepth
    focal_length = (sample['depth_intrinsics']['fx'] * width,
                    sample['depth_intrinsics']['fy'] * height)
    principal_point = (sample['depth_intrinsics']['cx'] * width,
                       sample['depth_intrinsics']['cy'] * height)
    depth *= sample['depth_intrinsics']['a']
    pixel_h_map, pixel_w_map = np.meshgrid(np.arange(height),
                                           np.arange(width),
                                           indexing='ij')
    valid_mask = np.logical_and(depth > 0, depth <= max_depth)
    n_valid_points = np.sum(valid_mask)

    if 0 == n_valid_points:
        # no valid points at all, return empty point cloud and attributes
        return pc, {'instance': np.array([]), 'semantic': np.array([])}

    points = np.zeros((n_valid_points, 3), dtype='float64')
    x_map = (pixel_w_map - principal_point[0]) * depth / focal_length[0]
    y_map = (pixel_h_map - principal_point[1]) * depth / focal_length[1]
    points[:, 0] = x_map[valid_mask]
    points[:, 1] = y_map[valid_mask]
    points[:, 2] = depth[valid_mask]

    pc.points = o3d.utility.Vector3dVector(points)    # must be float64

    # compute transformation matrix from extrinsic parameters
    rotation = R.from_quat(
        [extrinsics['quat_x'], extrinsics['quat_y'], extrinsics['quat_z'],
         extrinsics['quat_w']]
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation.as_matrix()
    transform[:3, 3] = [extrinsics['x'], extrinsics['y'], extrinsics['z']]

    # transform point cloud to world coordinates
    pc.transform(transform)

    # assign attributes (color is stored in labeled point cloud)
    pc.colors = o3d.utility.Vector3dVector(
        color[valid_mask].astype('float64') / 255     # must be float64
    )
    instance_vector = instance[valid_mask]
    semantic_vector = semantic[valid_mask]

    return pc, {'instance': instance_vector, 'semantic': semantic_vector}


def _save_ply(filepath, pc, attributes, shift=(1 << 16)):
    label_uint32 = _combine_semantic_instance(attributes['semantic'],
                                              attributes['instance'],
                                              shift=shift)
    color_uint8 = np.asarray(np.asarray(pc.colors)*255, dtype='uint8')
    points_f32 = np.asarray(pc.points, dtype='float32')

    # create vertex data (similar to ScanNet)
    vertex_data = np.asarray(
        [(v[0][0], v[0][1], v[0][2], 0, 0, 0, v[1][0], v[1][1], v[1][2], v[2])
         for v in zip(points_f32, color_uint8, label_uint32)],
        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
               ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
               ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
               ('label', 'u4')]
    )
    vertex_el = plyfile.PlyElement.describe(vertex_data, 'vertex')

    # write ply file
    plyfile.PlyData([vertex_el]).write(filepath)


def _combine_semantic_instance(semantic, instance, shift=(1 << 16)):
    # compute (panoptic) label  as semantic*shift + instance
    semantic_label = semantic.astype(np.uint32)
    instance_label = instance.astype(np.uint32)
    return (semantic_label * shift) + instance_label


def main():
    # parse args
    args = _parse_args()

    # base output directories
    output_path = osp.join(args.output_path, args.dataset + '_ply', args.split)
    scannet_output_path = osp.join(args.output_path, args.dataset + '_gt',
                                   '3d', args.split)

    # helper function to load dataset
    def load_dataset(sample_keys):
        dataset_cls = get_dataset_class(args.dataset)
        if args.dataset == 'scannet':
            dataset = dataset_cls(
                dataset_path=args.dataset_path,
                split=args.split,
                sample_keys=sample_keys,
                semantic_n_classes=20 if args.scannet_map_to_benchmark else 40
            )
        else:
            dataset = dataset_cls(
                dataset_path=args.dataset_path,
                split=args.split,
                sample_keys=sample_keys
            )
        return dataset

    # load dummy dataset to build scene dictionary
    dummy_dataset = load_dataset(sample_keys=('identifier',))

    scenes = defaultdict(list)
    for i, sample in enumerate(dummy_dataset):
        current_scene_id = '_'.join(sample['identifier'][:-1])
        scenes[current_scene_id].append(i)

    # load full dataset
    sample_keys = (
        'identifier',    # helps to know afterwards which sample was loaded
        'rgb', 'depth',    # camera data
        'depth_intrinsics', 'extrinsics',    # camera parameters
        'semantic', 'instance'    # tasks
    )
    dataset = load_dataset(sample_keys=sample_keys)

    # process scenes
    pbar = tqdm.tqdm(total=len(dataset))
    for scene, indices in scenes.items():
        pbar.set_description(f"Scene: {scene}")

        # create empty point cloud
        pc = o3d.geometry.PointCloud()
        attributes = {'instance': [], 'semantic': []}

        # load and concatenate all point clouds of the scene
        for i in indices:
            sample = dataset[i]

            # preprocess sample
            sample = _preprocess_sample(sample, args)

            # compute labeled point cloud
            cur_pc, cur_attributes = _compute_labeled_point_cloud(
                sample, max_depth=args.max_depth
            )

            # add point cloud to global point cloud and update attributes
            pc += cur_pc
            for key in attributes.keys():
                attributes[key].append(cur_attributes[key])

            # update progress bar
            pbar.update(1)

        attributes = {key: np.concatenate(attributes[key])
                      for key in attributes.keys()}

        # filter point cloud
        filtered_pc, _, indices_per_voxel = pc.voxel_down_sample_and_trace(
            voxel_size=args.voxel_size,
            min_bound=pc.get_min_bound(),
            max_bound=pc.get_max_bound(),
            approximate_class=False
        )

        # get filtered attributes (we use the most common value in each voxel)
        filtered_attributes = {
            k: np.zeros(len(indices_per_voxel), dtype=v.dtype)
            for k, v in attributes.items()
        }

        # note: this is not the most efficient way to do this, but it works and
        # is done only once
        for new_point_index in tqdm.tqdm(range(len(indices_per_voxel)),
                                         desc="Computing attributes for "
                                              "filtered point cloud",
                                         leave=False):
            for attr in attributes:
                # get values of all points in the voxel
                values = attributes[attr][indices_per_voxel[new_point_index]]
                # get most common value (most frequent value)
                uniques, counts = np.unique(values, return_counts=True)
                new_value = uniques[np.argmax(counts)]

                filtered_attributes[attr][new_point_index] = new_value

        # save point cloud as ply file
        last_identifier = sample['identifier']
        os.makedirs(osp.join(output_path, *last_identifier[:-1]),
                    exist_ok=True)
        voxel_sizes_str = str(args.voxel_size).replace('.', '_')
        max_depth_str = str(args.max_depth).replace('.', '_')
        fp = osp.join(output_path, *last_identifier[:-1],
                      f'voxel_{voxel_sizes_str}_maxdepth_{max_depth_str}.ply')
        _save_ply(fp, filtered_pc, filtered_attributes, shift=args.shift)

        # write ScanNet ground-truth label files
        if args.write_scannet_label:
            filename = '_'.join(last_identifier[:-1]) + '.txt'

            # pure semantic labels
            semantic_path = osp.join(scannet_output_path, 'semantic')
            os.makedirs(semantic_path, exist_ok=True)
            np.savetxt(
                osp.join(semantic_path, filename),
                filtered_attributes['semantic'],
                fmt='%d'
            )

            # semantic instance (panoptic) labels
            semantic_instance_path = osp.join(scannet_output_path,
                                              'semantic_instance')
            os.makedirs(semantic_instance_path, exist_ok=True)
            np.savetxt(
                osp.join(semantic_instance_path, filename),
                _combine_semantic_instance(filtered_attributes['semantic'],
                                           filtered_attributes['instance'],
                                           shift=args.shift),
                fmt='%d'
            )

            # instance
            instance_path = osp.join(scannet_output_path, 'instance')
            os.makedirs(instance_path, exist_ok=True)
            np.savetxt(
                osp.join(instance_path, filename),
                filtered_attributes['instance'],
                fmt='%d'
            )


if __name__ == '__main__':
    main()
