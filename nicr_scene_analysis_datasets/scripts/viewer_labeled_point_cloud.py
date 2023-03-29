# -*- coding: utf-8 -*-
"""
.. codeauthor:: Benedict Stephan <benedict.stephan@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse as ap
import inspect
import warnings

import numpy as np
try:
    import open3d as o3d
    import plyfile
except ImportError:
    print("Please install 'open3d' and 'plyfile' to use this script: "
          "'pip install open3d plyfile' or re-install this package with "
          "'with3d' target.")
    raise

from .common import AVAILABLE_COLORMAPS
from .common import get_colormap
from .common import print_section


USAGE = inspect.cleandoc(
    """
    Use the keys 1-5 to switch between displaying:
    1: color,
    2: semantic ground-truth annotations,
    3: instance ground-truth annotations,
    4: predicted semantic labels (see --semantic-label-file`, and
    5: predicted instance labels (see --instance-label-file`).

    Press h to show further help and q to quit.
    """
)


def _parse_args():
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description=(
            "Open3D-based viewer for ply files containing semantic and "
            "instance labels."
        )
    )

    parser.add_argument(
        'filepath',
        type=str,
        help="Path to the ply file",
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=('color',
                 'semantic', 'instance',
                 'additional_semantic', 'additional_instance'),
        default='color',
        help='Render mode (use keys to switch)'
    )
    parser.add_argument(
        '--semantic-colormap',
        type=str,
        default='auto_n',
        choices=AVAILABLE_COLORMAPS,
        help="Colormap to use for visualizing semantic annotations."
    )
    parser.add_argument(
        '--instance-colormap',
        type=str,
        default='auto_n',
        choices=AVAILABLE_COLORMAPS,
        help="Colormap to use for visualizing instance annotations."
    )
    parser.add_argument(
        '--semantic-label-filepath',
        type=str,
        default=None,
        help="Path to txt file containing the semantic label for each vertex. "
             "Useful for visualizing predictions."
    )
    parser.add_argument(
        '--instance-label-filepath',
        type=str,
        default=None,
        help="Path to txt file containing the instance label for each vertex. "
             "Useful for visualizing predictions."
    )
    parser.add_argument(
        '--use-scannet-format',
        action='store_true',
        help="If specified, labels will be taken as sem*1000+inst instead of "
             "our sem*(1<<16)+ins encoding."
    )

    # experimental
    parser.add_argument(
        '--second-pc-filepath',
        type=str,
        default=None,
        help="Path to a second ply file, e.g., a mapped representation."
    )
    parser.add_argument(
        '--second-pc-correspondence-filepath',
        type=str,
        default=None,
        help="Path to a txt file containing the correspondence between the "
             "point clouds given by `filepath` and `--second-pc-filepath`."
    )

    return parser.parse_args()


def _get_instance_colors(instance_labels, instance_cmap):
    # convert to continuos labels
    # instance_labels = np.where(
    #     np.unique(instance_labels) == instance_labels[:, np.newaxis]
    # )[1]
    if instance_labels.max() >= len(instance_cmap):
        warnings.warn(
            f"Color map has only {len(instance_cmap)} colors, but the "
            f"largest instance id is {instance_labels.max()}. Using modulo."
        )

    return instance_cmap[instance_labels % len(instance_cmap)]


def _load_ply(filepath, semantic_cmap, instance_cmap, use_scannet_format=False):
    plydata = plyfile.PlyData.read(filepath)

    num_verts = plydata['vertex'].count
    vertices = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    vertices[:, 0] = plydata['vertex'].data['x']
    vertices[:, 1] = plydata['vertex'].data['y']
    vertices[:, 2] = plydata['vertex'].data['z']

    pc = o3d.geometry.PointCloud()

    pc.points = o3d.utility.Vector3dVector(vertices)

    if 'red' in plydata['vertex']:
        colors = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        colors[:, 0] = plydata['vertex'].data['red']
        colors[:, 1] = plydata['vertex'].data['green']
        colors[:, 2] = plydata['vertex'].data['blue']
    else:
        colors = np.ones(shape=[num_verts, 3], dtype=np.float32)

    if 'label' in plydata['vertex']:
        labels = plydata['vertex'].data['label']    # uint16 / uint32

        if use_scannet_format:
            # uint16
            semantic_labels = labels // 1000
            instance_labels = labels - semantic_labels * 1000
        else:
            # uint32
            semantic_labels = labels >> 16
            instance_labels = labels & 0xFFFF

        # semantic colors
        semantic_colors = semantic_cmap[semantic_labels]

        # instance colors
        instance_colors = _get_instance_colors(instance_labels, instance_cmap)
    else:
        # we do not have annotations
        instance_colors = np.zeros_like(colors)
        semantic_colors = np.zeros_like(colors)

    labels = {
        'color': colors.astype('float64') / 255,
        'instance': instance_colors.astype('float64') / 255,
        'semantic': semantic_colors.astype('float64') / 255
    }

    pc.colors = o3d.utility.Vector3dVector(colors)

    return pc, labels


def main():
    # parse args
    args = _parse_args()

    # get colormaps
    semantic_cmap = get_colormap(args.semantic_colormap)
    instance_cmap = get_colormap(args.instance_colormap)

    # load ply file
    pc, labels = _load_ply(
        filepath=args.filepath,
        semantic_cmap=semantic_cmap,
        instance_cmap=instance_cmap,
        use_scannet_format=args.use_scannet_format
    )
    pc.colors = o3d.utility.Vector3dVector(labels[args.mode])

    print_section(
        "PC Stats",
        "BBox:\n"
        f"\tmin: {pc.get_min_bound()}\n"
        f"\tmax: {pc.get_max_bound()}\n"
        f"Number of points: {len(pc.points)}"
    )

    # print usage
    print_section("Usage", USAGE)

    # load additional predicted labels
    if args.semantic_label_filepath is not None:
        semantic_labels = np.loadtxt(args.semantic_label_filepath,
                                     dtype=np.uint64)

        assert len(semantic_labels) == len(np.array(pc.points))

        labels['additional_semantic'] = semantic_cmap[semantic_labels] / 255

    if args.instance_label_filepath is not None:
        instance_labels = np.loadtxt(args.instance_label_filepath,
                                     dtype=np.uint64)

        assert len(instance_labels) == len(np.array(pc.points))

        labels['additional_instance'] = _get_instance_colors(
            instance_labels, instance_cmap
        ) / 255

    visualizer = o3d.visualization.VisualizerWithKeyCallback()
    visualizer.create_window()

    def _change_mode(mode):
        def _func(vis):
            print(f'Switching to type: {mode}')
            if mode not in labels:
                print(f'No labels for type: {mode}')
                return False
            pc.colors = o3d.utility.Vector3dVector(labels[mode])
            vis.update_geometry(pc)
            vis.update_renderer()
            vis.poll_events()

            return False
        return _func

    # draw initial pc
    visualizer.add_geometry(pc)

    # add callback for switching between modes
    visualizer.register_key_callback(ord('1'),
                                     _change_mode('color'))
    visualizer.register_key_callback(ord('2'),
                                     _change_mode('semantic'))
    visualizer.register_key_callback(ord('3'),
                                     _change_mode('instance'))
    visualizer.register_key_callback(ord('4'),
                                     _change_mode('additional_semantic'))
    visualizer.register_key_callback(ord('5'),
                                     _change_mode('additional_instance'))

    # experimental: visualize correspondences between the (grund-truth) point
    # cloud given by args.filepath and a second point cloud given by
    # args.second_pc_filepath, e.g., a mapped representation
    if all([args.second_pc_filepath is not None,
            args.second_pc_correspondence_filepath is not None]):
        # load second point cloud
        second_pc, _ = _load_ply(args.second_pc_filepath,
                                 semantic_cmap, instance_cmap)

        # load correspondences
        correspondences = np.loadtxt(args.second_pc_correspondence_filepath,
                                     dtype='int64')
        correspondences = [
            (i, correspondences[i])
            for i in range(len(correspondences))
            if correspondences[i] != -1     # -1 means no correspondence
        ]

        # create line set
        lines = o3d.geometry.LineSet.create_from_point_cloud_correspondences(
            pc, second_pc, correspondences
        )
        visualizer.add_geometry(lines)
        visualizer.add_geometry(second_pc)

    visualizer.run()


if __name__ == "__main__":
    main()
