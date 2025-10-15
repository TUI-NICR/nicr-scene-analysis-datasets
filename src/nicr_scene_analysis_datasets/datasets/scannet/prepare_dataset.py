# -*- coding: utf-8 -*-
"""
.. codeauthor:: Leonard Rabes <leonard.rabes@tu-ilmenau.de>
"""
from typing import Dict, List, Optional, Tuple, Union

import argparse as ap
import io
import json
import os
from multiprocessing.pool import Pool
import sys
import traceback
import zipfile

import cv2
import numpy as np
from PIL import Image
import pandas as pd
import pkg_resources
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from ...utils.io import create_dir
from ...utils.io import create_or_update_creation_metafile
from .scannet import ScanNetMeta
from .scannet import VALID_CLASS_IDS_549
from .SensorData import SensorData
from .scannet200_constants import VALID_CLASS_IDS_20
from .scannet200_constants import VALID_CLASS_IDS_200

DATASET_SCANS_DIR = ('scans', 'scans', 'scans_test')
DATASET_FNAME_COMBINED_LABELS = 'scannetv2-labels.combined.tsv'  # from dataset
DATASET_FNAME_EXTENSION_SENS = '{:s}.sens'
DATASET_FNAME_EXTENSION_SEMANTIC = '{:s}_2d-label{:s}.zip'
DATASET_FNAME_EXTENSION_INSTANCE = '{:s}_2d-instance{:s}.zip'

# https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_train.txt
# https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_val.txt
# https://github.com/ScanNet/ScanNet/blob/master/Tasks/Benchmark/scannetv2_test.txt
SPLITS_FILEPATHS = (
        pkg_resources.resource_filename(__name__, 'scannetv2_train.txt'),
        pkg_resources.resource_filename(__name__, 'scannetv2_val.txt'),
        pkg_resources.resource_filename(__name__, 'scannetv2_test.txt'),
    )

# colors scannet40 and nyuv2 operate on the same labels, only the RGB
# values are different
SEMANTIC_COLORS_PER_CLASS = {
    20: ('scannet20',),
    40: ('scannet40', 'nyuv2'),
    200: ('scannet200',),
    549: tuple(),  # dont export the 549 colors as images
}

EXCEPTION_JSON_DUMP_FILE = "exception_dict_dump.json"
FAILED_IMAGES_FILE = "failed_img.txt"
CV2_WRITE_FLAGS = (cv2.IMWRITE_PNG_COMPRESSION, 9)


class NICRSensorData(SensorData):
    def __init__(self, filename: str, blacklist: Optional[List[int]] = None):
        super().__init__(filename)
        if blacklist is not None:
            for index in blacklist:
                # stop frame at index from being exported
                self.frames[index] = None

    @property
    def color_shape(self):
        return (self.color_height, self.color_width)

    @property
    def depth_shape(self):
        return (self.depth_height, self.depth_width)

    def get_frame_strings(self) -> List[str]:
        # return all the possible frame names of all the frames currently loaded
        # complete with zfill ('124' -> '00124')
        return [str(i).zfill(self.ZFILL) for i, _ in enumerate(self.frames)]

    def export_depth_images(self, output_path: str, image_size=None, frame_skip=1):
        create_dir(output_path)
        if image_size is not None:
            # unsupported argument, in child class
            # therefore invoke super
            super().export_color_images(output_path, image_size, frame_skip)
            return

        for i in range(0, len(self.frames), frame_skip):
            frame = self.frames[i]

            if frame is None:
                continue

            depth_data = frame.decompress_depth(self.depth_compression_type)
            depth = np.frombuffer(depth_data, dtype='uint16').reshape(self.depth_height, self.depth_width)
            name = str(i).zfill(SensorData.ZFILL) + '.png'
            path = os.path.join(output_path, name)
            cv2.imwrite(path, depth, CV2_WRITE_FLAGS)

    def export_color_images(self, output_path: str, image_size=None, frame_skip=1, blacklist=None) -> None:
        create_dir(output_path)

        if image_size is not None:
            # unsupported argument, in child class
            # therefore invoke super
            super().export_color_images(output_path, image_size, frame_skip)
            return

        for i in range(0, len(self.frames), frame_skip):
            frame = self.frames[i]

            if frame is None:
                continue

            byts = frame.color_data  # directly load bytes from frame (is a jpg img: ff d8 ff e0)
            path = os.path.join(output_path, str(i).zfill(self.ZFILL) + '.jpg')
            with open(path, 'wb') as f:  # save bytes to file
                f.write(byts)

    def export_intrinsics(self, output_path, intrinsic_mat: np.ndarray, width, height) -> None:
        intr = NICRSensorData.__normalize_intrinsics(intrinsic_mat, width, height)
        parent_dir = os.path.dirname(output_path)
        create_dir(parent_dir)

        with open(output_path, 'w') as f:
            json.dump(intr, f)

    def export_intrinsics_depth(self, output_path: str) -> None:
        self.export_intrinsics(output_path, self.intrinsic_depth, self.depth_width, self.depth_height)

    def export_intrinsics_color(self, output_path: str) -> None:
        self.export_intrinsics(output_path, self.intrinsic_color, self.color_width, self.color_height)

    def export_extrinsics(self, output_dir: str, frame_skip=1) -> None:
        create_dir(output_dir)

        for i in range(0, len(self.frames), frame_skip):
            frame = self.frames[i]

            if frame is None:
                continue

            has_inf = np.isinf(frame.camera_to_world)
            has_nan = np.isnan(frame.camera_to_world)
            defective = True in np.logical_or(has_inf, has_nan).flatten()

            assert not defective, "Found inf or nan in extrinsics mat."

            ext = NICRSensorData.__convert_extrinsics(frame.camera_to_world)
            path = os.path.join(output_dir, str(i).zfill(self.ZFILL) + '.json')
            with open(path, 'w') as file:  # write 16-bit
                json.dump(ext, file)

    @staticmethod
    def __convert_extrinsics(mat: np.ndarray) -> Dict[str, float]:
        # convert extrinsic transform matrix (4x4)
        # [[r00, r01, r02, tx],
        #  [r10, r11, r12, ty],
        #  [r20, r21, r22, tz],
        #  [0,   0,   0,   1 ]]

        rot_mat = mat[0:3, 0:3]  # get 3x3 rotation matrix (rij | i,j in [0, 1, 2])
        transl = mat[0:3, 3]  # translation component of the matrix (tx, ty, tz)
        rotation_quat = Rotation.from_matrix(rot_mat).as_quat()  # get quaternion from rot matrix
        quat_x, quat_y, quat_z, quat_w = rotation_quat

        return {
            'x': float(transl[0]),
            'y': float(transl[1]),
            'z': float(transl[2]),
            'quat_x': float(quat_x),
            'quat_y': float(quat_y),
            'quat_z': float(quat_z),
            'quat_w': float(quat_w)
        }

    @staticmethod
    def __normalize_intrinsics(mat: np.ndarray, width: int, height: int) -> Dict[str, float]:
        # from intrinsic camera matrix:
        # [[fx, 0,  cx],
        #  [0,  fy, cy],
        #  [0,  0,  1 ]]
        fx = mat[0][0]
        fy = mat[1][1]
        cx = mat[0][2]
        cy = mat[1][2]

        return {
            'fx': fx/width,
            'fy': fy/height,
            'cx': cx/width,
            'cy': cy/height
        }


class NICRImageZip:
    ZFILL = 5

    def __init__(self, file_path: str, blacklist: Optional[List[int]] = None) -> None:
        self.file_path = file_path
        self.load(file_path, blacklist)

    def load(self, file_path: str, blacklist: Optional[List[str]]) -> None:
        with open(file_path, 'rb') as f:
            bts = f.read()  # load zip into memory to speed up random stream access
            self.__bytes = io.BytesIO(bts)
        self.zip = zipfile.ZipFile(self.__bytes, mode='r')
        self.info = self.zip.infolist()
        self.info = [info for info in self.info if info.file_size > 0]  # remove everything thats not a file
        self.info = sorted(self.info, key=self.__get_filename)  # sort alphabetically by names, which are formatted

        if blacklist is not None:
            for index in blacklist:
                self.info[index] = None

        # a task represents a way to export an image
        # a task is a tuple with (output_dir, label_map, color_map)
        # each task is applied to each image
        self.__export_tasks: List[Tuple[str, np.ndarray, Tuple[Tuple[int, int, int]]]] = []

    def export_all(self, frame_skip=1) -> Optional[List[str]]:
        # returns a list of images, that could not be exported, else returns None
        # contains a list of images, that have failed to be exported
        failed_images = []

        grouping_dict, arr_dict = self.__group_tasks()

        # actual image export
        for i in range(0, len(self.info), frame_skip):  # iterate all images
            inf = self.info[i]
            if inf is None:
                # enforce blacklist, only none if blacklisted
                continue

            with self.zip.open(inf) as f:  # load image from zip
                byts = np.frombuffer(f.read(), dtype='uint8')  # load bytes as ndarray
                pixels = cv2.imdecode(byts, cv2.IMREAD_UNCHANGED)  # get image pixels

                if pixels is None:  # handling of failed images
                    failed_images.append(inf.filename)
                    continue  # skip the image, because it could not be loaded

                for lmap_key, subtasks in grouping_dict.items():  # iterate the items in the group
                    lmap = arr_dict[lmap_key]  # get actual ndarray

                    if lmap is None:
                        # if no changes necessary use pixels directly
                        pix = pixels
                    else:  # lmap is not None
                        # apply the label mapping once
                        # label mapping determines the dtype of the resulting pix arr
                        pix = lmap[pixels]

                    for odir, cmap in subtasks:  # complete the rest of the task on the img
                        create_dir(odir)

                        name = NICRImageZip.__get_filename(inf)
                        path = os.path.join(odir, name + '.png')
                        NICRImageZip.__export_image(pix, path, cmap)

        self.__export_tasks = []

        return failed_images if len(failed_images) > 0 else None

    def add_export(
            self,
            out_dir: str,
            label_map: Union[np.ndarray, None] = None,
            color_map: Union[List[Tuple[int, int, int]], None] = None) -> None:
        self.__export_tasks.append((out_dir, label_map, color_map))  # add a way to export images to list

    def __group_tasks(self):
        # group all the tasks by the label_map ndarray
        # to remove the need to apply it multiple times
        # to a single image
        grouping_dict: Dict[bytes, List[Tuple]] = {}  # contains the remaining task
        arr_dict: Dict[bytes, np.ndarray] = {}  # maps bytes to the actual ndarray

        for odir, lmap, cmap in self.__export_tasks:
            # bytes as key, because its an immutable ndarray representation
            # needs to account for the possibility, that label map is none
            lmap_key = lmap.tobytes() if lmap is not None else None
            if lmap_key in grouping_dict:
                grouping_dict[lmap_key].append((odir, cmap))
            else:
                grouping_dict[lmap_key] = [(odir, cmap)]
                arr_dict[lmap_key] = lmap

        return (grouping_dict, arr_dict)

    @staticmethod
    def __export_image(
            pix: np.ndarray,
            out_path: str,
            color_map: Union[List[Tuple[int, int, int]], None] = None) -> None:
        if pix.dtype == 'uint8' and color_map is not None:  # save grayscale with palette
            new_img = Image.fromarray(pix)
            new_img.putpalette(list(np.asarray(color_map, dtype='uint8').flatten()))
            new_img.save(out_path)

        elif pix.dtype == 'uint16' and color_map is not None:  # convert from grayscale to rgb
            cols = np.array(color_map, dtype='uint8')
            res = cols[pix]
            cv2.imwrite(out_path, cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

        elif color_map is None:  # write back without replacing colors
            cv2.imwrite(out_path, pix, CV2_WRITE_FLAGS)

        else:
            raise NotImplementedError

    @staticmethod
    def __get_filename(info: zipfile.ZipInfo) -> str:
        return os.path.basename(info.orig_filename).split('.')[0].zfill(NICRImageZip.ZFILL)


def get_out_path_dict(
        output_path: str,
        split_name: str,
        is_benchmark_split=False,
        base_subsample: int = 1,
        add_subsamples: Optional[List[int]] = None) -> Dict[str, str]:
    sdir = os.path.join(output_path, split_name)

    # the basic amount of dirs each split has to have
    out_paths = {
        'parent_dir': sdir,
        'rgb_dir': os.path.join(sdir, ScanNetMeta.RGB_DIR),
        'depth_dir': os.path.join(sdir, ScanNetMeta.DEPTH_DIR),

        'intr_rgb_dir': os.path.join(sdir, ScanNetMeta.INTRINSICS_RGB_DIR),
        'intr_depth_dir': os.path.join(sdir, ScanNetMeta.INTRINSICS_DEPTH_DIR),
        'extr_dir': os.path.join(sdir, ScanNetMeta.EXTRINSICS_DIR),

        #set output file path, not a directory, name decided by base_subsample
        'split_file': os.path.join(
            output_path,
            ScanNetMeta.get_split_filelist_filenames(base_subsample)[split_name])
    }

    if add_subsamples is not None:
        for ss in add_subsamples:
            fname = ScanNetMeta.get_split_filelist_filenames(subsample=ss)[split_name]
            out_paths[f'split_{ss}_file'] = os.path.join(output_path, fname)

    # only dirs, that are needed for non benchmark scenes
    if not is_benchmark_split:
        # dirs of semantic and instance images
        for mode in ScanNetMeta.INSTANCE_SEMANTIC_MODES:
            #dirs of semantic img
            for n_class in ScanNetMeta.SEMANTIC_N_CLASSES:
                # add non colored dir
                out_paths[f'sem_{mode}_{n_class}_dir'] = os.path.join(
                    sdir,
                    ScanNetMeta.SEMANTIC_DIR_FMT.format(mode, n_class))

                #add colored directories
                for n_class_color in SEMANTIC_COLORS_PER_CLASS[n_class]:
                    col_name = n_class_color.replace(str(n_class), '')  # remove n_class from scannet color name (scannet40 -> scannet)
                    out_paths[f'sem_{mode}_{n_class_color}_dir'] = os.path.join(
                        sdir,
                        ScanNetMeta.SEMANTIC_COLORED_DIR_FMT.format(mode, n_class, col_name))

        #dirs of instance img
        for mode in ScanNetMeta.INSTANCE_SEMANTIC_MODES:
            out_paths[f'inst_{mode}_dir'] = os.path.join(
                    sdir,
                    ScanNetMeta.INSTANCES_DIR_FMT.format(mode))

        #dir for scene class
        out_paths['scene_class_dir'] = os.path.join(sdir, ScanNetMeta.SCENE_CLASS_DIR)

    return out_paths


def get_scene_split_dict() -> Dict[str, int]:
    split_dict = {}
    for i, _ in enumerate(ScanNetMeta.SPLITS):  # iterate all splits
        path = SPLITS_FILEPATHS[i]  # txt resource with scene names
        with open(path, 'r') as f:
            for line in f:  # load each line and write to dict
                li = line.replace('\n', '')
                split_dict[li] = i

    return split_dict


def get_combined_to_nyu40_dict(tsv_path: str) -> Dict[int, int]:
    combined = pd.read_csv(tsv_path, delimiter='\t')
    c_to_nyu40_dict = {0: 0}

    #create a dict mapping (combined -> nyu40)
    for _, row in combined[['id', 'nyu40id']].iterrows():
        raw_id, nyu40_id = row.values
        c_to_nyu40_dict[int(raw_id)] = int(nyu40_id)

    return c_to_nyu40_dict


def get_combined_to_nyu20_dict(tsv_path: str) -> Dict[int, int]:
    c_to_20 = get_combined_to_nyu40_dict(tsv_path)
    for key, val in c_to_20.items():
        if val in VALID_CLASS_IDS_20:
            # remove created gaps by the non valid labels
            # 0 (void) not in VALID_CLASS_IDS_20, therefore add 1 to index
            c_to_20[key] = VALID_CLASS_IDS_20.index(val) + 1
        else:
            c_to_20[key] = 0  # set all non valid ids to void
    return c_to_20


def get_combined_to_200_dict(tsv_path: str) -> Dict[int, int]:
    combined = pd.read_csv(tsv_path, delimiter='\t')

    c_to_200_dict = {0: 0}
    #collect all possible ids
    for _, row in combined[['id']].iterrows():
        # set them to 0 initially, populate it later
        # ensures that all non valid ids become void
        c_to_200_dict[int(row.values[0])] = 0

    for i, class_id in enumerate(VALID_CLASS_IDS_200):
        # enumerate the valid 200 classes and set their list index as new id
        c_to_200_dict[class_id] = i + 1  # +1 because void is not in VALID_CLASS_IDS_200

    return c_to_200_dict


def get_combined_to_549_dict(tsv_path: str) -> Dict[int, int]:
    c_to_nyu40 = get_combined_to_nyu40_dict(tsv_path)
    labels = list(c_to_nyu40.keys())
    labels = sorted(labels)

    c_to_549_dict = {0: 0}
    for i, l in enumerate(labels):  # forces the mapping to result in dense labels without gaps
        c_to_549_dict[l] = i

    return c_to_549_dict


def dict_map_to_array(map_dict: Dict, dtype='uint16') -> np.ndarray:
    #create an array for integer indexing from dict
    keys = map_dict.keys()
    max_val = max(keys)  # necessary, when the keys have gaps in between
    dtype_max_val = np.iinfo(dtype).max

    # fill with max value to be able to spot mistakes later
    # only the ones filled in later should be used for indexing
    li = np.ones(max_val+1, dtype=dtype) * dtype_max_val
    for k in keys:
        li[k] = map_dict[k]
        assert map_dict[k] <= dtype_max_val

    return li


def export_scene_class(txt_path: str, out_dir: str, scene_name: str) -> None:
    create_dir(out_dir)
    d = {}
    # parse uncommon text file format
    for k, v in (line.split(' = ') for line in open(txt_path, 'r')):
        k = k.replace('\n', '')
        v = v.replace('\n', '').lower()
        v = 'misc' if v == 'misc.' else v  # remove unnecessary '.' of the misc class

        d[k] = v

    fpath = os.path.join(out_dir, f'{scene_name}.txt')
    with open(fpath, 'w') as f:
        f.write(d['sceneType'])  # export only scene type


def export_split_filenames(
        scene_info_dict: Dict[str, Tuple[int, str]],
        scene_split_dict: Dict[str, int],
        out_paths_split: List[Dict[str, str]],
        base_subsample: int = 1,
        add_subsamples: List[int] = None
) -> None:

    def filename_gen(scenes: List[str], subsample: int = 1):
        # yields the correct amount of filenames for all given scenes
        # there is no stop between scenes, all given scenes are merged together
        for scene in scenes:
            bl = []
            if scene in ScanNetMeta.BLACKLIST:
                bl = ScanNetMeta.BLACKLIST[scene]

            count, camera = scene_info_dict[scene]  # get the amount of filenames and the camera
            for fidx in range(0, count, subsample):  # handle subsamples with step arg of range
                if fidx in bl:
                    # skip blacklisted index
                    continue
                name = str(fidx).zfill(NICRSensorData.ZFILL)
                yield os.path.join(camera, scene, name) + '\n'  # add linebreak, because f.writelines(...) doesn't

    # all scenes, that need to be exported presorted
    scenes = sorted(scene_info_dict.keys())

    # save each split data in .txt files
    # export the default split txt, with fixed subsample
    for i, _ in enumerate(ScanNetMeta.SPLITS):
        path = out_paths_split[i]['split_file']
        # list of scenes in this split
        split_scenes = [scene for scene in scenes if scene_split_dict[scene] == i]
        filenames = filename_gen(split_scenes, base_subsample)

        with open(path, 'w') as f:
            f.writelines(filenames)

    # export the remaining subsamples
    for i, _ in enumerate(ScanNetMeta.SPLITS):
        for sub in add_subsamples:  # iterate over all subsamples
            path = out_paths_split[i][f'split_{sub}_file']
            # list of scenes in this split
            split_scenes = [scene for scene in scenes if scene_split_dict[scene] == i]
            filenames = filename_gen(split_scenes, sub)

            with open(path, 'w') as f:
                f.writelines(filenames)


def schedule_scenes(
        processes: int,
        source_paths: Dict[str, str],
        out_paths_split: List[Dict],
        scene_split_dict: Dict[str, int],
        base_subsample: int = 1,
        add_subsamples: Optional[List[int]] = None
) -> None:
    scenes = list(scene_split_dict.keys())
    scenes = sorted(scenes)

    # get dictionaries for semantic label remapping
    c_to_549 = get_combined_to_549_dict(source_paths['tsv'])
    c_to_200 = get_combined_to_200_dict(source_paths['tsv'])
    c_to_nyu40 = get_combined_to_nyu40_dict(source_paths['tsv'])
    c_to_nyu20 = get_combined_to_nyu20_dict(source_paths['tsv'])
    # to arrays for integer indexing
    c_to_549 = dict_map_to_array(c_to_549, dtype='uint16')  # dtype here also decides which dtype the image has
    c_to_200 = dict_map_to_array(c_to_200, dtype='uint8')
    c_to_nyu40 = dict_map_to_array(c_to_nyu40, dtype='uint8')
    c_to_nyu20 = dict_map_to_array(c_to_nyu20, dtype='uint8')
    conversion_dict = {  # combine all to a dict for less needed params in calls
        20: c_to_nyu20,
        40: c_to_nyu40,
        200: c_to_200,
        549: c_to_549,
    }

    # check if conversion dict is working correctly
    for n_classes, conversion_arr in conversion_dict.items():
        cls_count = len(np.unique(conversion_arr))
        max_val = np.max(conversion_arr)
        # +2 because of void and index not in use (noted as integer max value, see dict_map_to_array())
        assert cls_count == n_classes + 2, f"{n_classes} conversion not correct!"
        assert np.count_nonzero(conversion_arr < max_val) == len(VALID_CLASS_IDS_549) + 1, (
            f"{n_classes} does not map the correct number of classes!"
        )
        # test mapping
        class_ids = np.array((0,) + VALID_CLASS_IDS_549)
        mapped = conversion_arr[class_ids]
        assert np.max(mapped) < max_val, f"{n_classes} conversion not correct!"

    def parse_scene_args_gen(scenes: List[str]):  # argument generator for each parse_scene call
        for scene in scenes:
            split = scene_split_dict[scene]  # find correct split for the scene
            sdir = os.path.join(source_paths['data'], DATASET_SCANS_DIR[split])
            out_paths = out_paths_split[split]

            yield (scene, sdir, out_paths, conversion_dict, base_subsample)

    pdir = os.path.dirname(out_paths_split[0]["parent_dir"])  # parent dictionary to
    dump_path = os.path.join(pdir, EXCEPTION_JSON_DUMP_FILE)  # filepath to dump the working dict to
    working_dict: Dict[str, Tuple[int, str]] = {}

    if os.path.exists(dump_path):  # check if the output dictionary contains a dump file
        with open(dump_path, 'r') as f:
            working_dict = json.load(f)
        print(f"Loaded {len(working_dict)} scenes from previously saved working dictionary from json dump file")
        # remove all scenes, which are already done
        scenes = [scene for scene in scenes if scene not in working_dict]

    try:
        with Pool(processes=processes) as pool:
            # imap unordered for returning results, when they are completed
            res_iter = pool.imap_unordered(star_parse_scene, parse_scene_args_gen(scenes))
            for scene, res in tqdm(res_iter, desc='Parsing Scenes', total=len(scenes)):
                working_dict[scene] = res  # save result in dict

            export_split_filenames(
                working_dict,
                scene_split_dict,
                out_paths_split,
                base_subsample,
                add_subsamples
            )

    except BaseException as ex:
        with open(dump_path, 'w') as f:
            # save entire dictionary to file
            json.dump(working_dict, f)
        print(f"Unhandled exception occured, working dictionary dumped to: '{dump_path}'")
        raise ex


def star_parse_scene(args):
    # necessary for unpacking the arguments for parse_scene in multiprocessing
    try:
        return (args[0], parse_scene(*args))
    except BaseException as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback, limit=10, file=sys.stdout)
        raise ex


def parse_scene(
        scene_name: str,
        scene_dir: str,
        out_paths: Dict,
        conversion_dict: Dict[int, np.ndarray],
        base_subsample: int = 1
) -> Tuple[int, str]:
    # check if this scene has faulty images
    blacklist = None
    if scene_name in ScanNetMeta.BLACKLIST:
        blacklist = ScanNetMeta.BLACKLIST[scene_name]

    # parse sensor data
    sens_path = os.path.join(scene_dir, scene_name, DATASET_FNAME_EXTENSION_SENS.format(scene_name))
    sens = NICRSensorData(sens_path, blacklist)

    fcount = len(sens.frames)

    camera = ScanNetMeta.CAMERA_FMT.format(*sens.color_shape)
    assert camera in ScanNetMeta.CAMERAS

    sens.export_color_images(
        os.path.join(out_paths['rgb_dir'], camera, scene_name),
        frame_skip=base_subsample)
    sens.export_depth_images(
        os.path.join(out_paths['depth_dir'], camera, scene_name),
        frame_skip=base_subsample)
    sens.export_intrinsics_color(
        os.path.join(out_paths['intr_rgb_dir'], camera, scene_name + '.json'))
    sens.export_intrinsics_depth(
        os.path.join(out_paths['intr_depth_dir'], camera, scene_name + '.json'))
    sens.export_extrinsics(
        os.path.join(out_paths['extr_dir'], camera, scene_name),
        frame_skip=base_subsample)

    del sens

    # abort if this is a benchmark scene
    if 'scene_class_dir' not in out_paths:
        return fcount, camera  # necessary to handle scan_test scenes without semantic/instance labels

    def export_failed_img(
            failed_img: Optional[List[str]],
            img_src: str) -> None:
        if failed_img is None:
            return
        # export failed image info
        path = os.path.join(out_paths['parent_dir'], FAILED_IMAGES_FILE)
        try:
            with open(path, 'a') as f:
                for fimg in failed_img:
                    str_out = f"{img_src},{fimg}\n"
                    f.write(str_out)
        except IOError:
            # hoping, that 2 failed images are not found at the same time
            print(f"File could not be opened to store failed img: {str(failed_img)} in {img_src}")

    #parse semantic data
    for mode in ScanNetMeta.INSTANCE_SEMANTIC_MODES:
        mode_ext = f'-filt' if mode == ScanNetMeta.INSTANCE_SEMANTIC_MODES[1] else ''
        sem_path = os.path.join(scene_dir, scene_name, DATASET_FNAME_EXTENSION_SEMANTIC.format(scene_name, mode_ext))
        sem_zip = NICRImageZip(sem_path, blacklist)

        for n_class in ScanNetMeta.SEMANTIC_N_CLASSES:
            # add non colored export
            sem_zip.add_export(
                os.path.join(out_paths[f'sem_{mode}_{n_class}_dir'], camera, scene_name),
                conversion_dict[n_class])

            #add colored exports
            for n_class_color in SEMANTIC_COLORS_PER_CLASS[n_class]:
                sem_zip.add_export(
                    os.path.join(out_paths[f'sem_{mode}_{n_class_color}_dir'], camera, scene_name),
                    conversion_dict[n_class],
                    ScanNetMeta.SEMANTIC_CLASS_COLORS[n_class_color])

        failed = sem_zip.export_all(frame_skip=base_subsample)  # execute all prev scheduled ways to export images
        export_failed_img(failed, sem_path)

    #parse instance data
    for mode in ScanNetMeta.INSTANCE_SEMANTIC_MODES:
        mode_ext = f'-filt' if mode == ScanNetMeta.INSTANCE_SEMANTIC_MODES[1] else ''
        inst_path = os.path.join(scene_dir, scene_name, DATASET_FNAME_EXTENSION_INSTANCE.format(scene_name, mode_ext))
        inst_zip = NICRImageZip(inst_path, blacklist)

        inst_zip.add_export(os.path.join(out_paths[f'inst_{mode}_dir'], camera, scene_name))  # export instance img as is
        failed = inst_zip.export_all(frame_skip=base_subsample)
        export_failed_img(failed, inst_path)

    #parse scene label
    export_scene_class(
        os.path.join(scene_dir, scene_name, scene_name + '.txt'),
        os.path.join(out_paths['scene_class_dir'], camera),
        scene_name,
    )

    return fcount, camera


def main(args=None) -> None:
    # argument parser
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Prepare ScanNet dataset."
    )
    parser.add_argument(
        'source_path',
        type=str,
        help="Path where dataset is stored"
    )
    parser.add_argument(
        'output_path',
        type=str,
        help="Path where to store parsed dataset"
    )
    parser.add_argument(
        '--label-map-file',
        default=None,
        type=str,
        help="Path to scannet-labels.combined.tsv, "
             "if not specified, it is assumed to be in the `source_path`"
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=8,
        help='Number of worker processes spawned'
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=1,
        help="Create subsampled versions with every N samples of the the "
             "ScanNet dataset. This sample will change directly how many "
             "views are output to the output path."
    )
    parser.add_argument(
        '--additional-subsamples',
        type=int,
        nargs='*',
        default=[50, 100, 200, 500],
        help="Create subsampled versions with every N samples of the the "
             "ScanNet dataset. These additional subsamples do not change "
             "the amount of sample to the output path. They only add files "
             "with subsampled filepaths."
    )
    args = parser.parse_args(args)

    # check, that subsamples are correct
    subsample = args.subsample
    add_subsamples = args.additional_subsamples
    assert isinstance(subsample, int)
    assert subsample > 0, "Undefined behaviour for subsample < 1"
    if add_subsamples is not None:
        for asub in add_subsamples:
            assert isinstance(asub, int)
            assert asub > 0, "Undefined behaviour for subsample < 1"
            assert asub % subsample == 0, ("Additional subsamples "
                                           "need to be multiples of "
                                           "the output subsample")

    # handle source paths
    source_path: str = os.path.expanduser(args.source_path)
    tsv_path: str = args.label_map_file
    if tsv_path is None:  # get the default tsv path
        tsv_path = os.path.join(source_path, DATASET_FNAME_COMBINED_LABELS)
    else:
        tsv_path = os.path.expanduser(tsv_path)

    if not os.path.isdir(source_path) or not os.path.exists(tsv_path):
        raise IOError()  # check if sources exist

    source_paths = {  # reduce arguments for calls
        "data": source_path,
        "tsv": tsv_path
    }

    #handle output paths
    output_path: str = os.path.expanduser(args.output_path)

    create_dir(output_path)

    # directories for each split
    out_paths_split: list[dict] = []

    for split in ScanNetMeta.SPLITS:
        sdir = os.path.join(output_path, ScanNetMeta.SPLIT_DIRS[split])
        create_dir(sdir)

        out_paths = get_out_path_dict(
            output_path,
            split, split == 'test',
            base_subsample=subsample,
            add_subsamples=add_subsamples)

        # create all of the directories for export
        for key, dir in out_paths.items():
            if 'dir' in key:  # only create dirs
                create_dir(dir)

        out_paths_split.append(out_paths)

    scene_split_dict = get_scene_split_dict()

    # start dataset export
    schedule_scenes(
        args.n_processes,
        source_paths,
        out_paths_split,
        scene_split_dict,
        base_subsample=subsample,
        add_subsamples=add_subsamples)

    create_or_update_creation_metafile(output_path)


if __name__ == '__main__':
    main()
