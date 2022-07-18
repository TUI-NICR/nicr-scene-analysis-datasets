# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
.. codeauthor:: Söhnke Benedikt Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>

See: https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html
"""
import argparse as ap
import json
import os
import tarfile
from tempfile import gettempdir

import h5py
import cv2
import numpy as np
import pkg_resources
from scipy.io import loadmat
from tqdm import tqdm

from ...utils.img import dimshuffle
from ...utils.img import save_indexed_png
from ...utils.io import download_file
from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from .nyuv2 import NYUv2Meta


# https://github.com/VainF/nyuv2-python-toolkit/blob/master/splits.mat
SPLITS_FILEPATH = pkg_resources.resource_filename(__name__,
                                                  'splits.mat')
# https://github.com/VainF/nyuv2-python-toolkit/blob/master/class13Mapping.mat
CLASSES_13_FILEPATH = pkg_resources.resource_filename(__name__,
                                                      'class13Mapping.mat')
# https://github.com/VainF/nyuv2-python-toolkit/blob/master/classMapping40.mat
CLASSES_40_FILEPATH = pkg_resources.resource_filename(__name__,
                                                      'classMapping40.mat')
MANUAL_ORIENTATIONS_TRAIN_FILEPATH = pkg_resources.resource_filename(
    __name__,
    'manual_orientations_train.json'
)
MANUAL_ORIENTATIONS_TEST_FILEPATH = pkg_resources.resource_filename(
    __name__,
    'manual_orientations_test.json'
)

# see: http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/
DATASET_URL = 'http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat'

# see: https://cs.nyu.edu/~deigen/dnl/
# the file contains the precomputed surface normal
DATASET_NORMAL_URL = 'https://cs.nyu.edu/~deigen/dnl/normals_gt.tgz'


def tar_file_to_cv2(tar, tar_filename):
    # get member of filename in tar
    tar_member = tar.getmember(tar_filename)
    # get reference to file in tar
    file_extracted = tar.extractfile(tar_member)
    # read the file to a np array
    file_buffer = np.asarray(bytearray(file_extracted.read()),
                             dtype=np.uint8)
    # decode file array to cv2 image
    file_decoded = cv2.imdecode(file_buffer, cv2.IMREAD_UNCHANGED)
    return file_decoded


def main():
    # argument parser
    parser = ap.ArgumentParser(description='Prepare NYUv2 dataset.')
    parser.add_argument('output_path', type=str,
                        help='path where to store dataset')
    parser.add_argument('--mat-filepath', default=None,
                        help='filepath to NYUv2 mat file')
    parser.add_argument('--normal-filepath', default=None,
                        help='filepath to NYUv2 normal')
    parser.add_argument('--instance-min-relative-area',
                        type=float,
                        default=0.25/100,    # 0.25 %
                        help='minimum relative area for valid instances (the '
                             'actual number of pixels is calculated as '
                             '`instance-min-relative-area`*height*width')
    args = parser.parse_args()

    # preprocess args and expand user
    output_path = os.path.expanduser(args.output_path)
    if args.mat_filepath is None:
        mat_filepath = os.path.join(gettempdir(), 'nyu_depth_v2_labeled.mat')
    else:
        mat_filepath = os.path.expanduser(args.mat_filepath)

    # download mat file if mat_filepath does not exist
    if not os.path.exists(mat_filepath):
        print(f"Downloading mat file to: '{mat_filepath}'")
        download_file(DATASET_URL, mat_filepath, display_progressbar=True)

    if args.normal_filepath is None:
        normal_filepath = os.path.join(gettempdir(), 'normal_gt.tgz')
    else:
        normal_filepath = os.path.expanduser(args.normal_filepath)
    if not os.path.exists(normal_filepath):
        print(f"Downloading precomputed normals")
        download_file(DATASET_NORMAL_URL,
                      normal_filepath,
                      display_progressbar=True)

    # create output path if not exist
    create_dir(output_path)

    # create or update metafile
    create_or_update_creation_metafile(output_path)

    # load mat file and extract images
    print(f"Loading mat file: '{mat_filepath}'")

    with h5py.File(mat_filepath, 'r') as f:
        rgb_images = np.array(f['images'])
        labels = np.array(f['labels'])
        instances = np.array(f['instances'])
        depth_images = np.array(f['depths'])
        raw_depth_images = np.array(f['rawDepths'])

        # The sceneTypes key only holds a reference to the actual scene type
        scene_types_references = f['sceneTypes'][0]
        # Iterate over the references and store scene types in list
        scene_types = []
        for scene_type_reference in scene_types_references:
            # Resolve the reference to the actual scene type
            scene_type_obj = f[scene_type_reference][:, 0]
            # Convert scene type to string
            scene_type = ''.join(chr(c) for c in scene_type_obj)
            scene_types.append(scene_type)

    # dimshuffle images
    rgb_images = dimshuffle(rgb_images, 'bc10', 'b01c')
    labels = dimshuffle(labels, 'b10', 'b01')
    instances = dimshuffle(instances, 'b10', 'b01')
    depth_images = dimshuffle(depth_images, 'b10', 'b01')
    raw_depth_images = dimshuffle(raw_depth_images, 'b10', 'b01')

    # instances are connected to the semantic label.
    # The first instance of a class e.g. chair will have id 1 the second id 2
    # and so on. To be able to differentiate instances without the semantic
    # label, we need to store both information in the same array. The max
    # semantic label id in NYUv2 is 894, so it is enough to store it in 10 Bits
    # (2^10 = 1024). The maximum instance id is 37, so it is enough to store it
    # in 6 Bits (2^6 = 64). So we can use the first 10 Bit for the semantic
    # label (by shifting nums by 6) and use the other 6 Bits for encoding the
    # instance.
    instances = ((labels << 6) + instances)

    # determine minimum instance area, smaller instances are most likely not
    # valid (we observed that 768 pixel (2.5%*640*480) is a suitable threshold)
    min_instance_area = args.instance_min_relative_area * np.prod(instances.shape[-2:])

    # convert depth images (m to mm)
    depth_images = (depth_images * 1e3).astype('uint16')
    raw_depth_images = (raw_depth_images * 1e3).astype('uint16')

    # load split file (note that returned indexes start from 1)
    splits = loadmat(SPLITS_FILEPATH)
    train_idxs, test_idxs = splits['trainNdxs'][:, 0], splits['testNdxs'][:, 0]
    train_idxs -= 1
    test_idxs -= 1

    # load classes and class mappings (number of classes are without void)
    classes_40 = loadmat(CLASSES_40_FILEPATH)
    classes_13 = loadmat(CLASSES_13_FILEPATH)['classMapping13'][0][0]
    # class_names = {
    #     894: ['void'] + [c[0] for c in classes_40['allClassName'][0]],
    #     40: ['void'] + [c[0] for c in classes_40['className'][0]],
    #     13: ['void'] + [c[0] for c in classes_13[1][0]]
    # }
    mapping_894_to_40 = np.concatenate([[0], classes_40['mapClass'][0]])
    mapping_40_to_13 = np.concatenate([[0], classes_13[0][0]])

    # get color (1 (void) + n_colors)
    colors = {
        894: np.array(NYUv2Meta.SEMANTIC_CLASS_COLORS_894, dtype='uint8'),
        40: np.array(NYUv2Meta.SEMANTIC_CLASS_COLORS_40, dtype='uint8'),
        13: np.array(NYUv2Meta.SEMANTIC_CLASS_COLORS_13, dtype='uint8')
    }

    # normals
    normal_tar = tarfile.open(normal_filepath)
    normal_mask = np.zeros((480, 640), dtype=bool)
    normal_mask[44:471, 40:601] = True

    # orientations
    # note that the orientations were labeled manually in 3d space, None means
    # that the instance was skipped due to instance inconsistency or size
    orientations_for_split = {}
    with open(MANUAL_ORIENTATIONS_TRAIN_FILEPATH) as f:
        orientations_for_split['train'] = json.load(f)
    with open(MANUAL_ORIENTATIONS_TEST_FILEPATH) as f:
        orientations_for_split['test'] = json.load(f)

    # save images
    for idxs, split in zip((train_idxs, test_idxs), ('train', 'test')):
        print(f"Processing split: {split}")
        split_dir = NYUv2Meta.SPLIT_DIRS[split]
        rgb_base_path = os.path.join(output_path, split_dir, NYUv2Meta.RGB_DIR)
        instances_base_path = os.path.join(
            output_path, split_dir, NYUv2Meta.INSTANCES_DIR
        )
        depth_base_path = os.path.join(
            output_path, split_dir, NYUv2Meta.DEPTH_DIR
        )
        depth_raw_base_path = os.path.join(
            output_path, split_dir,
            NYUv2Meta.DEPTH_RAW_DIR
        )
        labels_894_base_path = os.path.join(
            output_path, split_dir,
            NYUv2Meta.SEMANTIC_DIR_FMT.format(894)
        )
        labels_40_base_path = os.path.join(
            output_path, split_dir, NYUv2Meta.SEMANTIC_DIR_FMT.format(40)
        )
        labels_13_base_path = os.path.join(
            output_path, split_dir, NYUv2Meta.SEMANTIC_DIR_FMT.format(13)
        )
        labels_894_colored_base_path = os.path.join(
            output_path, split_dir,
            NYUv2Meta.SEMANTIC_COLORED_DIR_FMT.format(894)
        )
        labels_40_colored_base_path = os.path.join(
            output_path, split_dir,
            NYUv2Meta.SEMANTIC_COLORED_DIR_FMT.format(40)
        )
        labels_13_colored_base_path = os.path.join(
            output_path, split_dir,
            NYUv2Meta.SEMANTIC_COLORED_DIR_FMT.format(13)
        )
        normal_base_path = os.path.join(output_path, split_dir,
                                        NYUv2Meta.NORMAL_DIR)
        orientations_base_path = os.path.join(output_path, split_dir,
                                              NYUv2Meta.ORIENTATIONS_DIR)
        scene_class_base_path = os.path.join(output_path, split_dir,
                                             NYUv2Meta.SCENE_CLASS_DIR)

        create_dir(rgb_base_path)
        create_dir(instances_base_path)
        create_dir(depth_base_path)
        create_dir(depth_raw_base_path)
        create_dir(labels_894_base_path)
        create_dir(labels_13_base_path)
        create_dir(labels_40_base_path)
        create_dir(labels_894_colored_base_path)
        create_dir(labels_13_colored_base_path)
        create_dir(labels_40_colored_base_path)
        create_dir(normal_base_path)
        create_dir(orientations_base_path)
        create_dir(scene_class_base_path)

        for idx in tqdm(idxs):
            # rgb image
            cv2.imwrite(os.path.join(rgb_base_path, f'{idx:04d}.png'),
                        cv2.cvtColor(rgb_images[idx], cv2.COLOR_RGB2BGR))

            # Due to the conversion above, the instance ids are not continuous
            # anymore and sparse.
            # This can be a disadvantage for tasks such as PQ calculation.
            # Thats why the following code changes the instance ids to be
            # smaller.
            # Furthermore, we filter instances with quite small area as they are
            # most likely not valid
            current_instance = instances[idx]
            instances_write = np.zeros_like(current_instance)
            for new_id, c_id in enumerate(np.unique(current_instance)):
                mask = current_instance == c_id

                if mask.sum() < min_instance_area:
                    # instance is too small, skip it
                    continue

                # assign new id
                # note that new_id==0 always corresponds to the void label due
                # to the instance encoding above ((labels << 6) + instances),
                # thus, we do not assign new_id==0 to any instance of a thing
                # class except its area is smaller than the area threshold
                instances_write[mask] = new_id

            cv2.imwrite(os.path.join(instances_base_path, f'{idx:04d}.png'),
                        instances_write)

            # depth image
            cv2.imwrite(os.path.join(depth_base_path, f'{idx:04d}.png'),
                        depth_images[idx])

            # raw depth image
            cv2.imwrite(os.path.join(depth_raw_base_path, f'{idx:04d}.png'),
                        raw_depth_images[idx])

            # label with 1+894 classes
            label_894 = labels[idx]
            cv2.imwrite(os.path.join(labels_894_base_path, f'{idx:04d}.png'),
                        label_894)

            # colored label image
            # (normal png16 as this type does not support indexed palettes)
            label_894_colored = colors[894][label_894]
            cv2.imwrite(os.path.join(labels_894_colored_base_path,
                                     f'{idx:04d}.png'),
                        cv2.cvtColor(label_894_colored, cv2.COLOR_RGB2BGR))

            # label with 1+40 classes
            label_40 = mapping_894_to_40[label_894].astype('uint8')
            cv2.imwrite(os.path.join(labels_40_base_path, f'{idx:04d}.png'),
                        label_40)
            # colored label image
            # (indexed png8 with color palette)
            save_indexed_png(os.path.join(labels_40_colored_base_path,
                                          f'{idx:04d}.png'),
                             label_40, colors[40])

            # label with 1+13 classes
            label_13 = mapping_40_to_13[label_40].astype('uint8')
            cv2.imwrite(os.path.join(labels_13_base_path, f'{idx:04d}.png'),
                        label_13)
            # colored label image
            # (indexed png8 with color palette)
            save_indexed_png(os.path.join(labels_13_colored_base_path,
                                          f'{idx:04d}.png'),
                             label_13, colors[13])

            # normals
            # get mask for normals as cv2 image
            mask_img = tar_file_to_cv2(normal_tar,
                                       f'normals_gt/masks/{idx:04d}.png')
            # get normal image as cv2 image
            # important: every valid normal vector in the dataset is allready
            #            normalized to unit length
            normal_img = tar_file_to_cv2(normal_tar,
                                         f'normals_gt/normals/{idx:04d}.png')
            # mask out all invalid normal vectors
            # the value is set to 127 because after the image gets loaded,
            # it is converted to a unit vector of length 0.
            normal_img[mask_img == 0] = (127, 127, 127)
            normal_path = os.path.join(normal_base_path, f'{idx:04d}.png')
            cv2.imwrite(normal_path, normal_img)

            # orientations
            orientations = orientations_for_split[split].get(f'{idx:04d}', {})
            fp = os.path.join(orientations_base_path, f'{idx:04d}.json')
            with open(fp, 'w') as f:
                # Get unique instances so we can remove orinatations which were
                # filtered.
                unique_instances = np.unique(instances_write)
                orientations_filtered = {}
                for i, o in orientations.items():
                    # we do have a valid orientation
                    if o is None:
                        continue
                    # verify that instance still exists after size filter
                    if int(i) not in unique_instances:
                        continue
                    orientations_filtered[i] = o
                json.dump(orientations_filtered, f, indent=4)

            # scene classes
            scene_class_path = os.path.join(scene_class_base_path,
                                            f'{idx:04d}.txt')
            with open(scene_class_path, 'w') as f:
                f.write(scene_types[idx])

    # save meta files
    print("Writing meta files")
    np.savetxt(os.path.join(output_path, 'class_names_1+13.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_13.class_names,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+13.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_13.colors_array,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+40.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_40.class_names,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+40.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_40.colors_array,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_names_1+894.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_894.class_names,
               delimiter=',', fmt='%s')
    np.savetxt(os.path.join(output_path, 'class_colors_1+894.txt'),
               NYUv2Meta.SEMANTIC_LABEL_LIST_894.colors_array,
               delimiter=',', fmt='%s')

    # splits
    np.savetxt(os.path.join(output_path,
                            NYUv2Meta.SPLIT_FILELIST_FILENAMES['train']),
               train_idxs,
               fmt='%04d')
    np.savetxt(os.path.join(output_path,
                            NYUv2Meta.SPLIT_FILELIST_FILENAMES['test']),
               test_idxs,
               fmt='%04d')

    # remove downloaded files
    if args.mat_filepath is None:
        print(f"Removing downloaded mat file: '{mat_filepath}'")
        os.remove(mat_filepath)
    if args.normal_filepath is None:
        print(f"Removing downloaded normal file: '{normal_filepath}'")
        os.remove(normal_filepath)


if __name__ == '__main__':
    main()
