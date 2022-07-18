# -*- coding: utf-8 -*-
"""
.. codeauthor:: Soehnke Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
"""
from typing import Tuple

import os

from detectron2.data import DatasetCatalog
from detectron2.data import MetadataCatalog

from ..pytorch import Cityscapes
from ..pytorch import COCO
from ..pytorch import SUNRGBD
from ..pytorch import Hypersim
from ..pytorch import NYUv2
from ..pytorch import SceneNetRGBD
from ..dataset_base._base_dataset import DatasetBase


def register_dataset_to_d2(
    name_prefix: str,
    dataset_class: DatasetBase,
    sample_keys: Tuple = ('identifier', 'rgb', 'semantic', 'instance'),
    **kwargs
):
    # For registering the dataset to D2 a dataset path is required.
    # However, at this point of execution, the path is not known yet.
    # In D2 the path is usually controlled by the `DETECTRON2_DATASETS`
    # environment variable which defaults to `./datasets`.
    # See https://github.com/facebookresearch/detectron2/tree/main/datasets
    # for further information.
    # It can be required to change this path at execution time, e.g. when
    # the dataset is loaded through a batch system (e.g. SLURM).
    # The path itself is required, as the registered dataset must return a
    # list of the length of the dataset.
    # However the length is only known, when the dataset is instantiated
    # with the correct split, subsample and path.
    # This is why we need to register the dataset with a "lambda" which
    # dynamically resolves the path.
    def get_dataset(split: str) -> DatasetBase:
        dataset_path = os.environ.get('DETECTRON2_DATASETS', f'./datasets/{name_prefix}')
        dataset_inst = dataset_class(
            dataset_path=dataset_path,
            split=split,
            sample_keys=sample_keys,
            **kwargs
        )
        return dataset_inst

    splits = dataset_class.SPLITS
    for split in splits:
        name = name_prefix + '_' + split
        # The lambda is very important. Else the split parameter will be
        # overwritten and the get_dataset will allways return the same dataset.
        # By using the lambda with a defailt parameter, the split parameter
        # is specific to this registration to d2.
        DatasetCatalog.register(name, lambda split=split: get_dataset(split))

    dataset = dataset_class(disable_prints=True, **kwargs)
    # Metadata is same for every split so we only have to create it once.
    is_thing = dataset.config.semantic_label_list_without_void.classes_is_thing
    is_stuff = [not x for x in is_thing]
    class_ids = list(range(len(is_thing)))

    semantic_label = dataset.config.semantic_label_list_without_void.class_names
    semantic_color = dataset.config.semantic_label_list_without_void.colors_array
    semantic_color = semantic_color.tolist()

    # thing_id_global = list(itertools.compress(class_ids, is_thing))
    # # +1 as 0 is reserved for void
    thing_dataset_id_to_contiguous_id = {}
    for idx, id in enumerate(class_ids):
        if is_thing[idx]:
            # + 1 cause of void.
            # Void can't be predicted which is why our first segment has
            # 1 and in contiguous mapping id 0.
            thing_dataset_id_to_contiguous_id[id+1] = idx

    # stuff_id_global = list(itertools.compress(class_ids, is_stuff))
    # +1 as 0 is reserved for void
    stuff_dataset_id_to_contiguous_id = {}
    for idx, id in enumerate(class_ids):
        if is_stuff[idx]:
            stuff_dataset_id_to_contiguous_id[id+1] = idx

    ignore_label = 255

    # Store metadata in MetadataCatalog for every split
    for split in splits:
        name = name_prefix + '_' + split
        catalog = MetadataCatalog.get(name)
        catalog.set(name=name)
        catalog.set(thing_classes=semantic_label)
        catalog.set(thing_colors=semantic_color)
        catalog.set(thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id)
        catalog.set(stuff_classes=semantic_label)
        catalog.set(stuff_colors=semantic_color)
        catalog.set(stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id)
        catalog.set(ignore_label=ignore_label)
        # Required for some implementations (e.g. Panoptic DeepLab)
        # to use correct evaluator.
        catalog.set(evaluator_type='coco_panoptic_seg')
        # Store the whole dataset configuration in the metadata.
        catalog.set(dataset_config=dataset.config)
        catalog.set(label_divisor=256)


# Automatically register all datasets so that they are available through
# detectron2's DatasetCataloge.
# Note that they are just registered so that the stats can be access.
# For using the dataset, the 'set_dataset_path' function should be called first.
register_dataset_to_d2(name_prefix='cityscapes',
                       dataset_class=Cityscapes,
                       sample_keys=('identifier', 'rgb', 'semantic'))
register_dataset_to_d2(name_prefix='coco', dataset_class=COCO)
register_dataset_to_d2(name_prefix='hypersim', dataset_class=Hypersim)
register_dataset_to_d2(name_prefix='nyuv2', dataset_class=NYUv2)
register_dataset_to_d2(name_prefix='scenenetrgbd',
                       dataset_class=SceneNetRGBD,
                       sample_keys=('identifier', 'rgb', 'semantic'))
register_dataset_to_d2(name_prefix='sunrgbd', dataset_class=SUNRGBD)
