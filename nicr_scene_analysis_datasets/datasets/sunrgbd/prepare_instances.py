# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
import json

from numba import jit
from numpy import matlib
import numpy as np
from pkg_resources import resource_string
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R


from ..nyuv2 import nyuv2
from ..nyuv2 import prepare_dataset


class SetEncoder(json.JSONEncoder):
    """
    Class for easy serialization of sets.
    See: https://stackoverflow.com/questions/8230315/how-to-json-serialize-sets
    """
    def default(self, o):
        if isinstance(o, set):
            return list(o)
        return json.JSONEncoder.default(self, o)


class SUNRGBDInstances:
    def __init__(self):
        self.nyuv2_40_classes = loadmat(prepare_dataset.CLASSES_40_FILEPATH)
        self.nyuv2_894_to_40_mapping = \
            np.concatenate([[0], self.nyuv2_40_classes['mapClass'][0]])

        self.nyuv_894_classes_strings = nyuv2.NYUv2Meta.SEMANTIC_LABEL_LIST_894.class_names

        self.additional_mapping = json.loads(
            resource_string(__name__, "nyu_additional_class_mapping.json"))

        weak_mapping = json.loads(
            resource_string(__name__, "nyu_weak_box_3d_mapping.json"))

        mat_len = len(self.nyuv2_894_to_40_mapping)
        self.weak_mapping_mat = np.zeros((mat_len, mat_len))  # + 1
        for i in range(len(self.nyuv2_894_to_40_mapping)):
            self.weak_mapping_mat[i, i] = 1

        for key, value in weak_mapping.items():
            key_idx = nyuv2.NYUv2Meta.SEMANTIC_LABEL_LIST_40.index(key)
            for v in value:
                v_idx = nyuv2.NYUv2Meta.SEMANTIC_LABEL_LIST_40.index(v)
                self.weak_mapping_mat[key_idx, v_idx] = 1
        self.unmapped_classes = {}

        self.box_rot = R.from_rotvec([np.pi/2, 0, 0])

    def get_instance(self, boxes_3d, intrinsics, extrinscis,
                     depth_image, semantic_label):

        boxes_3d_class = boxes_3d["class"]
        boxes_3d_class = self.map_box_classes_to_nyuv_int(boxes_3d_class)

        boxes_3d_coords = np.asarray(boxes_3d["coordinates"])
        height, width = semantic_label.shape[:2]
        # If not box classes are found, there are no instance labels
        if len(boxes_3d_coords) > 0:
            point_image = self.depth_img_to_point_image(depth_image,
                                                        intrinsics,
                                                        extrinscis)

            instance_img = \
                self.do_instances_segmentation(*semantic_label.shape[:2],
                                               boxes_3d_coords,
                                               boxes_3d_class,
                                               semantic_label,
                                               point_image,
                                               self.weak_mapping_mat)
        else:
            instance_img = np.zeros((height, width), dtype=np.uint16)

        box_mapping = np.zeros(len(boxes_3d_class), dtype=np.uint16)
        for idx, c in enumerate(boxes_3d_class):
            if c != 0:
                box_mapping[idx] = idx + 1

        return instance_img, box_mapping

    def map_box_classes_to_nyuv_int(self, box_classes):
        boxes_classes_int = []
        for c in box_classes:
            if c in self.nyuv_894_classes_strings:
                c_num = self.nyuv_894_classes_strings.index(c)
                c_num_40 = self.nyuv2_894_to_40_mapping[c_num]
                boxes_classes_int.append(int(c_num_40))
            elif c in self.additional_mapping:
                boxes_classes_int.append(int(self.additional_mapping[c]))
            else:
                # 0 = void for unmatched boxes
                boxes_classes_int.append(0)
                if c not in self.unmapped_classes:
                    self.unmapped_classes[c] = 0
                self.unmapped_classes[c] += 1
        return np.asarray(boxes_classes_int, dtype=np.uint8)

    def depth_img_to_point_image(self, depth_image, intrinsics,
                                 extrinscis, scale=1000.0):
        """
        This method converts a image with depth values to an image
        which contains global coordinates by using intrinsics and extrinscis.
        This is usefull cause then its possible to check if a point in
        the image is withhin a 3d bounding box
        """
        fx = intrinsics["Fx"]
        fy = intrinsics["Fy"]
        cx = intrinsics["Cx"]
        cy = intrinsics["Cy"]
        extrinscis_rot = R.from_matrix(extrinscis)
        h, w = depth_image.shape[:2]

        point_img = self.calc_point_coords(depth_image, fx, fy,
                                           cx, cy, scale, h, w)
        point_img[:, :] = extrinscis_rot\
            .apply(point_img.reshape(h*w, 3))\
            .reshape(h, w, 3)

        return point_img

    @staticmethod
    @jit(nopython=True)
    def calc_point_coords(depth_image, fx, fy, cx, cy, scale, h, w):
        '''
        Method needs to be static so we don't have to jit whole class
        '''
        point_img = np.zeros((h, w, 3), np.float32)
        for idy in range(h):
            for idx in range(w):
                depth_val = depth_image[idy, idx]
                # This conversion is taken from the SUN RGBD matlab
                # source code.
                depth_val = (depth_val >> 3) | (depth_val >> (16-3))
                depth_val_scale = depth_val/scale
                point_x = (idx-cx)*depth_val_scale/fx
                point_y = (idy-cy)*depth_val_scale/fy
                point_img[idy, idx] = np.asarray([point_x,
                                                  point_y,
                                                  depth_val_scale])
        return point_img

    @staticmethod
    @jit(nopython=True)
    def do_instances_segmentation(height, width, boxes, boxes_classes,
                                  semantic_mat, point_mat, weak_mapping):
        '''
        This method check for every point in a point image if it is
        within a 3d bounding box.
        After that the box class gets compared to its semantic label.
        If both match we declare it as a instance.
        '''
        instances_img = np.zeros((height, width), dtype=np.uint16)
        for instance_id, (box_3d, box_c) in enumerate(zip(boxes,
                                                          boxes_classes)):
            if box_c == 0:
                continue
            current_box_classes, = np.where(weak_mapping[box_c] != 0)
            # See https://math.stackexchange.com/q/1472049
            p1p2 = box_3d[0] - box_3d[4]
            p1p4 = box_3d[0] - box_3d[1]
            p1p5 = box_3d[0] - box_3d[2]

            u = np.cross(p1p4, p1p5)
            v = np.cross(p1p2, p1p5)
            w = np.cross(p1p2, p1p4)

            dot_min_u = np.dot(np.ascontiguousarray(box_3d[0]), u)
            dot_min_v = np.dot(np.ascontiguousarray(box_3d[0]), v)
            dot_min_w = np.dot(np.ascontiguousarray(box_3d[0]), w)

            dot_max_u = np.dot(np.ascontiguousarray(box_3d[4]), u)
            dot_max_v = np.dot(np.ascontiguousarray(box_3d[1]), v)
            dot_max_w = np.dot(np.ascontiguousarray(box_3d[2]), w)

            num_points_in_box = 0

            class_nums_in_box = np.zeros(41)
            for box_class in current_box_classes:
                for idy in range(height):
                    for idx in range(width):

                        if semantic_mat[idy, idx] != box_class:
                            continue

                        num_points_in_box += 1

                        point = point_mat[idy, idx].astype(np.float64)
                        if not check_direction(u, point, dot_min_u, dot_max_u):
                            continue
                        if not check_direction(v, point, dot_min_v, dot_max_v):
                            continue
                        if not check_direction(w, point, dot_min_w, dot_max_w):
                            continue
                        class_nums_in_box[box_class] += 1

            max_class_idx = class_nums_in_box.argmax()
            max_class = class_nums_in_box[max_class_idx]
            # ignore void
            if max_class_idx == 0:
                continue

            if not max_class/num_points_in_box > 0.1:
                continue

            for idy in range(height):
                for idx in range(width):

                    if semantic_mat[idy, idx] != max_class_idx:
                        continue

                    point = point_mat[idy, idx].astype(np.float64)
                    if not check_direction(u, point, dot_min_u, dot_max_u):
                        continue
                    if not check_direction(v, point, dot_min_v, dot_max_v):
                        continue
                    if not check_direction(w, point, dot_min_w, dot_max_w):
                        continue

                    instances_img[idy, idx] = instance_id + 1

        return instances_img

    def convert_boxes_3d(self, boxes_json):
        """
        Sun RGBD stores boxes not in corner notation.
        For easyier use we convert it.
        """
        new_boxes_dict = {}
        new_boxes_dict["orientations"] = np.array(boxes_json["orientations"])
        new_boxes_dict["class"] = boxes_json["class"]
        new_boxes_dict["extrinsics"] = boxes_json["extrinsics"]

        boxes_arr = []
        for basis, coeffs, centroid in zip(boxes_json["basis"],
                                           boxes_json["coeffs"],
                                           boxes_json["centroid"]):
            basis = np.array(basis)
            coeffs = np.array(coeffs)
            # Enlarge boxes a little.
            # This is easier to do it here then in corner notation.
            coeffs += coeffs*0.15
            centroid = np.array(centroid)

            corners = np.zeros((8, 3))
            idx = np.argsort(np.abs(basis[:, 0]))[::-1]

            basis = basis[idx, :]
            coeffs = coeffs[idx]

            idx = np.argsort(np.abs(basis[1:2, 1]))[::-1]
            if idx[0] == 2:
                basis[1:2, :] = np.flip(basis[1:2, :], 0)
                coeffs[1:2] = np.flip(coeffs[1:2], 1)

            basis = self.flip_towards_viewer(basis,
                                             matlib.repmat(centroid, 3, 1))

            coeffs = np.abs(coeffs)

            # front_lower_left
            corners[0, :] = -basis[0, :] * coeffs[0] \
                + basis[1, :] * coeffs[1] \
                + basis[2, :] * coeffs[2]

            # front_lower_right
            corners[1, :] = basis[0, :] * coeffs[0] \
                + basis[1, :] * coeffs[1] \
                + basis[2, :] * coeffs[2]

            # back_lower_right
            corners[2, :] = basis[0, :] * coeffs[0]  \
                + -basis[1, :] * coeffs[1] \
                + basis[2, :] * coeffs[2]

            # back_lower_left
            corners[3, :] = -basis[0, :] * coeffs[0] \
                + -basis[1, :] * coeffs[1] \
                + basis[2, :] * coeffs[2]

            # front_upper_left
            corners[4, :] = -basis[0, :] * coeffs[0] \
                + basis[1, :] * coeffs[1] \
                + -basis[2, :] * coeffs[2]

            # front_upper_right
            corners[5, :] = basis[0, :] * coeffs[0] \
                + basis[1, :] * coeffs[1] \
                + -basis[2, :] * coeffs[2]

            # back_upper_right
            corners[6, :] = basis[0, :] * coeffs[0] \
                + -basis[1, :] * coeffs[1] \
                + -basis[2, :] * coeffs[2]

            # back_upper_left
            corners[7, :] = -basis[0, :] * coeffs[0] \
                + -basis[1, :] * coeffs[1] \
                + -basis[2, :] * coeffs[2]

            corners += matlib.repmat(centroid, 8, 1)
            corners = self.box_rot.apply(corners)

            # Change order to the expected one for point cloud matching
            corners = corners[[0, 1, 4, 5, 3, 2, 7, 6]]
            boxes_arr.append(corners)

        boxes_coords = np.ascontiguousarray(boxes_arr)
        new_boxes_dict["coordinates"] = boxes_coords
        return new_boxes_dict

    def flip_towards_viewer(self, normal, points):
        dist = np.sqrt(np.sum(points**2, 1))
        points = points / matlib.repmat(dist, 1, 3).reshape(3, 3)
        proj = sum(points*normal, 1)
        flip = proj > 0
        normal[flip, :] = -normal[flip, :]
        return normal

    def load_intrinsics(self, intrinsics_path):
        with open(intrinsics_path, "r") as f:
            intrinsics = f.read()
        intrinsics = intrinsics.replace("\n", " ").split(" ")[:-1]
        intrinsics_dict = {}
        intrinsics_dict["Fx"] = float(intrinsics[0])
        intrinsics_dict["Fy"] = float(intrinsics[4])
        intrinsics_dict["Cx"] = float(intrinsics[2])
        intrinsics_dict["Cy"] = float(intrinsics[5])
        return intrinsics_dict


@jit(nopython=True)
def check_direction(unit, point, dot_min, dot_max):
    dot_point = np.dot(point, unit)
    if (dot_max <= dot_point and dot_point <= dot_min) \
       or (dot_max >= dot_point and dot_point >= dot_min):
        return True
    return False
