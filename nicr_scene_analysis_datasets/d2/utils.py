# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Callable, Iterable, Dict

import copy
import os

import numpy as np
from panopticapi.utils import IdGenerator
from scipy import stats
import torch

from ..dataset_base import DatasetConfig


class NICRChainedDatasetMapper:
    def __init__(
        self,
        mapper_list: Iterable[Callable]
    ) -> None:
        """Simple class which can hold multiple dataset mappers
           and calls them after another.
        """
        self.mapper_list = mapper_list

    def __call__(self, data: Dict) -> Dict:
        for mapper in self.mapper_list:
            data = mapper(data)
        return data


class NICRSceneAnalysisDatasetMapper:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        *args, **kwargs
    ) -> None:
        self.dataset_config = dataset_config

        # Store categories for panoptic api
        categories = []
        for idx, label in enumerate(dataset_config.semantic_label_list_without_void):
            label_dict = {}
            label_dict['supercategory'] = label.class_name
            label_dict['name'] = label.class_name
            # + 1 cause of void
            label_dict['id'] = idx + 1
            label_dict['isthing'] = int(label.is_thing)
            label_dict['color'] = [int(a) for a in label.color]
            categories.append(label_dict)
        self.categories_list = categories
        self.categories = {cat['id']: cat for cat in categories}

        self.ignore_label = 255

    def __call__(self, dataset_dict: Dict) -> Dict:
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # Dict where data in d2 layout is stored
        mapped_dataset = {}

        # Get the rgb image
        rgb_image = dataset_dict['rgb']
        h, w, _ = rgb_image.shape

        # Get semantic segmentation label
        semantic = dataset_dict['semantic']
        # Semantic to tensor and int64
        semantic = torch.from_numpy(semantic).long()
        # Set void label to ignore_label
        void_mask = semantic == 0
        semantic[void_mask] = self.ignore_label

        # Prepare panoptic segmentation for d2 implementations
        # like Panoptic Deeplab
        instance = dataset_dict['instance']
        # There are different memory layouts for panoptic segmentation.
        # 1. RGB where each pixel corresponds to a semantic class and
        #    a instance id. This is the usual layout if the panoptic
        #    segmentation gets stored in a file.
        # 2. A one channel image where each pixel corresponds to a semantic
        #    class and instance id which are combined e.g. by following formula:
        #    pan_id = 256 * semantic_id + instance_id
        # As this mapper mimics that the images are loaded from a file, we
        # create it as in the first methode.
        # Create a empty rgb image with the same size as the semantic image
        # and three channels.
        panoptic_segmentation = np.zeros((h, w, 3), dtype=np.uint8)
        # The IdGenerator of the panopticapi can be used to generate the correct color.
        id_generator = IdGenerator(self.categories)
        ids = np.unique(instance)
        segm_info = []

        segment_id_dict = {}
        for id in ids:
            instance_mask = instance == id
            semantic_id = stats.mode(semantic[instance_mask]).mode[0]

            if semantic_id == self.ignore_label:
                continue

            segment_id, color = id_generator.get_id_and_color(semantic_id)
            # segment_id -= 1
            panoptic_segmentation[instance_mask] = color

            # For stuff classes it's possible that there are duplicates
            # of the same segment_id.
            # This can break panopticapi in evaluation.
            # To prevent this, we store the segment_id in a dict and reuse
            # the segment if it allready exists.
            if segment_id in segment_id_dict:
                segment_id_dict[segment_id]['area'] += int(instance_mask.sum())
            else:
                segment_id_dict[segment_id] = {
                    'id': segment_id,
                    # 0 was initally reserved for void.
                    # Hoewever, void can't be predicted.
                    # For a contigous mapping to range [0, num_classes), we
                    # have to decrement the id by 1.
                    'category_id': int(semantic_id) - 1,
                    'iscrowd': 0,
                    'area': int(instance_mask.sum())
                }

        segm_info = list(segment_id_dict.values())

        # Store all data in d2 layout
        mapped_dataset['image_id'] = dataset_dict['identifier'][0]
        mapped_dataset['file_name'] = dataset_dict['identifier'][0] + ".png"
        mapped_dataset['image'] = np.ascontiguousarray(rgb_image)
        mapped_dataset['width'] = w
        mapped_dataset['height'] = h
        mapped_dataset['sem_seg'] = np.ascontiguousarray(semantic)
        mapped_dataset['pan_seg'] = np.ascontiguousarray(panoptic_segmentation)
        mapped_dataset['segments_info'] = segm_info
        # segments_info hold the information about the segments as required by
        # detectron2. In this case the category_id is in the range [0, num_classes).
        # However the originale coco json layout, which is also used by the
        # panopticapi for evaluation needs to work on the original ids.
        # This is why we calculate the originale id again and stored it in
        # segments_info_json, which then can be used by the panopticapi.
        # The current solution isn't optimal, as it only works for datasets with
        # contiguous ids and void with id = 0.
        # A better way would be to store the original id in the segments_info_json
        # which should be then mapped to the range [0, num_classes) similar to
        # https://github.com/facebookresearch/detectron2/blob/cbbc1ce26473cb2a5cc8f58e8ada9ae14cb41052/detectron2/data/datasets/coco_panoptic.py#L26
        # However this is fine for all implemented datasets, because the
        # requirement holds true for all of them.

        segm_info_json = []
        for segm in segm_info:
            segm_cp = copy.deepcopy(segm)
            # +1 for PQ calculation
            segm_cp['category_id'] += 1
            segm_info_json.append(segm_cp)

        mapped_dataset['segments_info_json'] = segm_info_json
        return mapped_dataset


def set_dataset_path(dataset_path: str):
    # In detectron2 the standart dataset path can be changed using the following
    # env var.
    os.environ['DETECTRON2_DATASETS'] = dataset_path
