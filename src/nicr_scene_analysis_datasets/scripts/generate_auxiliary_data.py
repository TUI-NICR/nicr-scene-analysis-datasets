# -*- coding: utf-8 -*-
"""
.. codeauthor:: Söhnke Benedikt Fischedick <soehnke-benedikt.fischedick@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, List, Optional, Sequence, Tuple

import argparse as ap
import os
import shutil
from collections import Counter

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .. import get_dataset_class
from ..dataset_base._base_dataset import DatasetBase
from ..dataset_base._depth_dataset import depth_compute_stats
from ..utils.imports import is_depth_estimation_available
from ..utils.imports import is_embedding_estimation_available
from ..utils.io import create_dir
from ..utils.io import create_or_update_creation_metafile
from ..utils.misc import partial_class

# Constants
MAX_INSTANCES_PER_CATEGORY = 1 << 16  # Used for panoptic segmentation
VOID_LABEL = 0  # Common void label used across functions

# Datasets that support unified domestic scene labels
SCENE_INDOOR_DOMESTIC_DATASETS = ['hypersim', 'nyuv2', 'scannet', 'sunrgbd']

# Default template for text embedding generation
DEFAULT_TEXT_EMBEDDING_TEMPLATE = "a detailed photo of a {:s}"

# Default depth estimators with their VRAM requirements
DEFAULT_DEPTH_ESTIMATORS = (
    'depthanything_v2__indoor_large',  # ~5GB VRAM
    # Models were not used for DVEFormer
    # 'zoedepth__indoor',                # ~8GB VRAM
    # 'dino_v2_dpt__indoor_giant',
)

# Default embedding estimators
DEFAULT_EMBEDDING_ESTIMATORS = (
    'alpha_clip__l14-336-grit-20m',
    # Models were not used for DVEFormer
    # 'alpha_clip__l14-336-grit-1m',
    # 'alpha_clip__l14-combined',
)

# Maximum number of pixels for depth estimation (Full HD)
DEFAULT_MAX_DEPTH_PIXELS = 1920 * 1080


def _parse_args(args: Optional[Sequence[str]] = None) -> ap.Namespace:
    # Determine available auxiliary data --------------------------------------
    available_auxiliary_data = []
    if is_depth_estimation_available(raise_error=False):
        available_auxiliary_data.append('depth')
        from ..auxiliary_data.depth_estimation import KNOWN_DEPTH_ESTIMATORS
    else:
        KNOWN_DEPTH_ESTIMATORS = ['PLEASE INSTALL WITH DEPTH ESTIMATION TARGET']

    if is_embedding_estimation_available(raise_error=False):
        available_auxiliary_data.append('panoptic-embedding')
        available_auxiliary_data.append('image-embedding')
        from ..auxiliary_data.embedding_estimation import \
            KNOWN_EMBEDDING_ESTIMATORS
    else:
        KNOWN_EMBEDDING_ESTIMATORS = ['PLEASE INSTALL WITH EMBEDDING TARGET']

    # Parse the actual args  --------------------------------------------------
    parser = ap.ArgumentParser(
        formatter_class=ap.ArgumentDefaultsHelpFormatter,
        description="Generate auxiliary data for a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="The dataset to generate auxiliary data for."
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        help="Path to the dataset. This is the root directory of the dataset."
             "Note that the generated auxiliary data will be stored in the "
             "same directory."
    )
    parser.add_argument(
        '--subsample',
        type=int,
        default=None,
        help="Optional subsample factor forwarded to the dataset (currently "
             "used by ScanNet to keep every n-th frame)."
    )
    parser.add_argument(
        '--split',
        nargs='+',
        type=str,
        default=('all',),
        help="The split(s) to generate auxiliary data for."
             "If 'all' is given, all splits are processed."
    )
    parser.add_argument(
        '--auxiliary-data',
        nargs='+',
        type=str,
        choices=available_auxiliary_data,
        help="The auxiliary data to generate."
    )
    parser.add_argument(
        '--cache-models',
        action='store_true',
        default=False,
        help="Cache the models for the estimators.",
    )
    parser.add_argument(
        '--n-processes',
        type=int,
        default=1,
        help="Number of processes to use for depth stats computation."
    )

    # Handle depth estimation auxiliary data -----------------------------------
    depth_parser = parser.add_argument_group(
        'Depth Estimation',
        description="Options for generating depth auxiliary data."
    )
    depth_parser.add_argument(
        '--depth-estimators',
        nargs='+',
        type=str,
        default=DEFAULT_DEPTH_ESTIMATORS,
        choices=KNOWN_DEPTH_ESTIMATORS,
        help="Depth estimators to use for predicting missing depth. "
    )
    depth_parser.add_argument(
        '--depth-estimator-device',
        type=str,
        default='cpu',
        help="Device to use for depth estimation."
    )
    depth_parser.add_argument(
        '--depth-estimator-max-pixels',
        type=int,
        default=DEFAULT_MAX_DEPTH_PIXELS,
        help="Maximum number of input pixels (h*w) passed to a depth "
             "estimator. If an input exceeds this number of pixels, it is "
             "resized while keeping the aspect ratio, thus, this parameter "
             "can be used to reduce the memory consumption. "
             f"Default is {DEFAULT_MAX_DEPTH_PIXELS}."
    )
    # Handle embedding auxiliary data ------------------------------------------
    embedding_parser = parser.add_argument_group(
        'Embedding Estimation',
        description="Options for generating embedding auxiliary data."
    )
    embedding_parser.add_argument(
        '--embedding-estimators',
        nargs='+',
        type=str,
        default=DEFAULT_EMBEDDING_ESTIMATORS,
        choices=KNOWN_EMBEDDING_ESTIMATORS,
        help="Embedding estimators to use for predicting missing embeddings."
    )
    embedding_parser.add_argument(
        '--embedding-estimator-device',
        type=str,
        default='cpu',
        help="Device to use for embedding estimation."
    )
    embedding_parser.add_argument(
        '--embedding-semantic-n-classes',
        nargs='+',
        type=int,
        default=(-1,),
        help="Number of semantic classes which are used to obtain the "
             "embedding. If -1 is given, all classes are used."
    )

    # Parse the actual arguments
    args = parser.parse_args(args)

    # Determine the number of semantic classes to use for the embedding
    if args.embedding_semantic_n_classes == (-1,):
        dataset_class = get_dataset_class(args.dataset)
        args.embedding_semantic_n_classes = \
            dataset_class.SEMANTIC_N_CLASSES
    # Ensure that its a tuple
    if isinstance(args.embedding_semantic_n_classes, int):
        args.embedding_semantic_n_classes = \
            (args.embedding_semantic_n_classes,)
    elif isinstance(args.embedding_semantic_n_classes, list):
        args.embedding_semantic_n_classes = \
            tuple(args.embedding_semantic_n_classes)
    elif isinstance(args.embedding_semantic_n_classes, tuple):
        pass
    else:
        raise ValueError(
            "Invalid type for 'embedding_semantic_n_classes'. "
            "Expected int, list or tuple, got "
            f"{type(args.embedding_semantic_n_classes)}"
        )
    return args


def get_dataset(
    args,
    split: str,
    sample_keys: List[str],
    semantic_n_classes: Optional[int] = None,
    depth_estimator: Optional[str] = None
) -> DatasetBase:
    dataset_class = get_dataset_class(args.dataset, with_auxiliary_data=True)
    available_sample_keys = dataset_class.get_available_sample_keys(split)
    # Only take the sample keys which are available in the dataset
    reduced_sample_keys = \
        [sk for sk in sample_keys if sk in available_sample_keys]
    # Print a warning if some sample keys are not available
    if len(reduced_sample_keys) != len(sample_keys):
        print(
            f"Warning: Some sample keys are not available in the dataset. "
            f"Available sample keys for split '{split}': "
            f"{available_sample_keys}. Using {reduced_sample_keys} instead."
        )
    kwargs = {
        'dataset_path': args.dataset_path,
        'split': split,
        'sample_keys': reduced_sample_keys,
        'depth_estimator': depth_estimator
    }

    # E.g. for ScanNet as smaller subsample might be useful for mapping
    subsample = args.subsample
    if subsample is not None:
        kwargs['subsample'] = subsample
    if semantic_n_classes is not None:
        kwargs['semantic_n_classes'] = semantic_n_classes

    return dataset_class(**kwargs)


def generate_depth(args, split: str, tmp_path: str) -> None:
    # prepare depth estimation ------------------------------------------------
    # Set the required sample keys for the dataset.
    # Identifer is required for setting the output file name.
    # RGB is required for the depth estimation.
    required_dataset_keys = ['identifier', 'rgb']
    # Get the dataset object
    dataset = get_dataset(args, split, required_dataset_keys)

    # prepare factory for depth estimators
    depth_estimator_factory = {}

    # import are done here to avoid general dependency on torch and
    # transformers
    from ..auxiliary_data.depth_estimation import get_depth_estimator_class

    cache_basepath = None
    if not args.cache_models:
        cache_basepath = os.path.join(tmp_path, 'depth_estimators')

    for estimator_name in args.depth_estimators:
        depth_estimator_factory[estimator_name] = partial_class(
            get_depth_estimator_class(estimator_name),
            device=args.depth_estimator_device,
            max_pixels=args.depth_estimator_max_pixels,
            auto_set_up=True,
            cache_basepath=cache_basepath
        )

    n_estimators = len(depth_estimator_factory)
    for i, (name, Estimator) in enumerate(depth_estimator_factory.items()):
        estimator = Estimator()
        desc = f'Predicting depth ({i+1}/{n_estimators} - {name})'
        for idx, sample in enumerate(tqdm(dataset, desc=desc)):
            depth_img = estimator.predict(sample['rgb'])
            dst_depth_filename = dataset._get_filename(idx) + '.png'
            dst_depth_filepath = os.path.join(
                args.dataset_path,
                dataset.split,
                dataset.AUXILIARY_DEPTH_DIR_FMT.format(name),
                dst_depth_filename
            )
            create_dir(os.path.dirname(dst_depth_filepath))

            # Write the depth image with error handling
            if not cv2.imwrite(dst_depth_filepath, depth_img):
                print(
                    "Warning: Failed to write depth image to "
                    f"{dst_depth_filepath}"
                )

        # Free up memory
        del estimator


def compute_depth_stats(
    args, split: str, tmp_path: str
) -> Dict[str, Dict[str, Dict[str, float]]]:
    required_dataset_keys = ['depth']
    depth_stats = {
        estimator_name: {} for estimator_name in args.depth_estimators
    }

    for depth_estimator in args.depth_estimators:
        dataset = get_dataset(
            args=args,
            split=split,
            sample_keys=required_dataset_keys,
            depth_estimator=depth_estimator
        )
        # Mandatory to compute the correct stats
        dataset.add_new_depth_estimator(depth_estimator)
        depth_stats[depth_estimator][split] = \
            depth_compute_stats(
                dataset=dataset,
                n_threads=args.n_processes,
                debug=False
            )
    return depth_stats


def generate_image_and_text_scene_embedding(
    args, split: str, tmp_path: str
) -> Dict[str, Dict[str, np.ndarray]]:
    # Set the required sample keys for the dataset.
    required_dataset_keys = ['identifier', 'rgb']
    # Get the dataset object
    dataset = get_dataset(args, split, required_dataset_keys)

    # prepare factory for embedding estimators
    embedding_estimator_factory = {}

    # import are done here to avoid general dependency on torch and alpha_clip
    from ..auxiliary_data.embedding_estimation \
        import get_embedding_estimator_class

    cache_basepath = \
        os.path.join(tmp_path, 'embedding_estimators') \
        if not args.cache_models else None

    for estimator_name in args.embedding_estimators:
        embedding_estimator_factory[estimator_name] = partial_class(
            get_embedding_estimator_class(estimator_name),
            device=args.embedding_estimator_device,
            cache_basepath=cache_basepath
        )

    n_estimators = len(embedding_estimator_factory)
    estimator_scene_class_embedding_dicts = {}
    for i, (name, Estimator) in enumerate(embedding_estimator_factory.items()):
        estimator = Estimator()
        desc = f'Predicting image embedding ({i+1}/{n_estimators} - {name})'
        for idx, sample in enumerate(tqdm(dataset, desc=desc)):
            rgb_input = sample['rgb']
            h, w, _ = rgb_input.shape

            # Alpha-CLIP expects and rgb image and a binary mask as an input.
            # Usually the mask would be positive for the object of interest.
            # In this case we want to get the embedding for the whole image,
            # so we create a mask with all ones (whole image of interest).
            mask_input = np.ones((h, w, 1), dtype=np.uint8)
            image_embedding = estimator.predict(rgb_input, mask_input)

            # Store the embedding for the whole image
            dst_filename = dataset._get_filename(idx) + '.npz'
            dst_filepath = os.path.join(
                args.dataset_path,
                dataset.split,
                dataset.AUXILIARY_IMAGE_EMBEDDING_DIR_FMT.format(name),
                dst_filename
            )
            create_dir(os.path.dirname(dst_filepath))
            np.savez_compressed(dst_filepath, image_embedding)

        # We also want to generate the scene class name embeddings
        # for the scene labels of the dataset.
        # As the dataset might support different scene labels (original and
        # adjusted for indoor domestic from EMSANet Paper), this is done
        # twice.

        # Check if the dataset supports indoor domestic labels
        dataset_name = args.dataset
        supports_indoor_domestic = dataset_name in SCENE_INDOOR_DOMESTIC_DATASETS

        # Create label configurations for both standard and indoor domestic
        # (if supported)
        label_configurations = [
            {
                'scene_use_indoor_domestic_labels': False,
                'suffix': '',
                'label_type': 'default'
            }
        ]

        if supports_indoor_domestic:
            label_configurations.append({
                'scene_use_indoor_domestic_labels': True,
                'suffix': '_indoor_domestic',
                'label_type': 'indoor domestic'
            })

        # Process each label configuration
        template = DEFAULT_TEXT_EMBEDDING_TEMPLATE
        for config in label_configurations:
            print(
                f"Predicting scene class name embedding "
                f"({i+1}/{n_estimators} - {name}) - "
                f"{config['label_type']} labels"
            )

            # Create appropriate dataset object based on configuration
            if config['scene_use_indoor_domestic_labels']:
                current_dataset = get_dataset_class(args.dataset)(
                    dataset_path=args.dataset_path,
                    split=split,
                    sample_keys=required_dataset_keys,
                    scene_use_indoor_domestic_labels=True
                )
            else:
                current_dataset = dataset

            # Extract scene class names and generate embeddings
            scene_class_names = \
                current_dataset.config.scene_label_list.class_names
            formatted_class_names = \
                [template.format(name) for name in scene_class_names]

            text_embeddings = \
                estimator._get_text_embedding(formatted_class_names)
            text_embeddings = text_embeddings.cpu().numpy()

            # Create dictionary mapping class names to embeddings
            scene_class_names_embedding_dict = {
                scene_class_name: text_embedding
                for scene_class_name, text_embedding in zip(
                    formatted_class_names, text_embeddings
                )
            }

            # Store embeddings with appropriate suffix
            estimator_scene_class_embedding_dicts[name + config['suffix']] = \
                scene_class_names_embedding_dict

        # Free up memory
        del estimator
    return estimator_scene_class_embedding_dicts


# This function is a copy from here:
# https://github.com/TUI-NICR/nicr-multitask-scene-analysis/blob/ffd00b3f62a87aef220463f804fe669bac72c84d/nicr_mt_scene_analysis/utils/panoptic_merge.py#L43
# It's currently copied here to avoid a dependency on the multitask repository.
# TODO: Maybe there is a better solution to avoid this dependency.
def naive_merge_semantic_and_instance_np(
    sem_seg: np.ndarray,
    ins_seg: np.ndarray,
    max_instances_per_category: int,
    thing_ids: Sequence[int],
    void_label: int
) -> Tuple[torch.Tensor, Dict[int, int]]:
    # check some input assumptions to ensure that uint32 for output is fine
    assert sem_seg.dtype in (np.uint8, np.uint16)
    assert ins_seg.dtype == np.uint16
    assert void_label >= 0

    # In case thing mask does not align with semantic prediction.
    pan_seg = np.zeros_like(sem_seg, dtype=np.uint32) + void_label

    # Keep track of instance id for each class.
    class_id_tracker = Counter()

    # This dict keeps track of which panoptic id corresponds to which
    # instance id.
    id_dict = {}

    # Paste thing by majority voting.
    instance_ids = np.unique(ins_seg)

    for ins_id in instance_ids:
        if ins_id == 0:
            continue

        thing_mask = (ins_seg == ins_id)
        if len(thing_mask.nonzero()[0]) == 0:
            continue

        semantic_labels = np.unique(sem_seg[thing_mask])
        # Naive approach is to always take the full semantic mask.
        # If a instance label includes more than one semantic label, the
        # instance is divided in multiple parts.
        for class_id in semantic_labels:
            # ignore void
            if class_id == 0:
                continue
            class_id = class_id.astype(np.uint32)
            class_id_tracker[class_id.item()] += 1   # -> first id is 1
            new_ins_id = class_id_tracker[class_id.item()]
            panoptic_id = (class_id * max_instances_per_category + new_ins_id)
            id_dict[int(panoptic_id)] = int(ins_id)

            label_mask = (sem_seg == class_id)
            mask = label_mask & thing_mask
            pan_seg[mask] = panoptic_id

    # Paste stuff to unoccupied area.
    class_ids = np.unique(sem_seg)
    for class_id in class_ids:
        # ignore void
        if class_id == 0:
            continue
        if class_id.item() in thing_ids:
            # thing class
            continue
        class_id = class_id.astype(np.uint32)
        stuff_mask = (sem_seg == class_id) & (ins_seg == 0)
        pan_seg[stuff_mask] = (class_id * max_instances_per_category)

    return pan_seg, id_dict


def generate_panoptic_and_text_class_name_embedding(
    args,
    split: str,
    tmp_path: str,
    semantic_n_classes: Optional[int] = None,
    void_label: int = None,
    max_instances_per_category: int = None
) -> Dict[str, Dict[str, np.ndarray]]:
    # Set the required sample keys for the dataset.
    required_dataset_keys = ['identifier', 'rgb', 'instance', 'semantic']
    # Get the dataset object
    dataset = get_dataset(
        args, split, required_dataset_keys, semantic_n_classes
    )
    # The get_dataset function might remove some sample keys which are not
    # available in the dataset (e.g. instance or semantic).
    # Therefore, we need to check if the required keys are actually available.
    # If not we just skip the current split.
    actual_keys = dataset.get_available_sample_keys(split)
    if not all([key in actual_keys for key in required_dataset_keys]):
        print(
            f"Skipping split '{split}' as not all required keys are available."
            f"Required keys: {required_dataset_keys}, "
            f"Available keys: {actual_keys}"
        )
        return None

    # prepare factory for embedding estimators
    embedding_estimator_factory = {}

    # import are done here to avoid general dependency on torch and alpha_clip
    from ..auxiliary_data.embedding_estimation \
        import get_embedding_estimator_class

    cache_basepath = \
        os.path.join(tmp_path, 'embedding_estimators') \
        if not args.cache_models else None

    for estimator_name in args.embedding_estimators:
        embedding_estimator_factory[estimator_name] = partial_class(
            get_embedding_estimator_class(estimator_name),
            device=args.embedding_estimator_device,
            cache_basepath=cache_basepath
        )

    estimator_semantic_class_embedding_dicts = {}
    n_estimators = len(embedding_estimator_factory)
    for i, (name, Estimator) in enumerate(embedding_estimator_factory.items()):
        estimator = Estimator()
        desc = f'Predicting panoptic embedding ({i+1}/{n_estimators} - {name})'

        # Use global constants instead of redefining
        thing_class_ids = np.where(
            dataset.config.semantic_label_list.classes_is_thing
        )[0]

        for idx, sample in enumerate(tqdm(dataset, desc=desc)):
            rgb_input = sample['rgb']
            h, w, _ = rgb_input.shape

            instance_input = sample['instance']
            semantic_input = sample['semantic']

            # Create panoptic segmentation by merging semantic and instance
            panoptic_input, id_dict = naive_merge_semantic_and_instance_np(
                semantic_input,
                instance_input,
                max_instances_per_category or MAX_INSTANCES_PER_CATEGORY,
                thing_class_ids,
                void_label or VOID_LABEL
            )

            # Process each unique panoptic segment
            panoptic_embedding_dict = {}
            for panoptic_id in np.unique(panoptic_input):
                panoptic_id = panoptic_id.item()
                # skip void label
                if panoptic_id == void_label:
                    continue

                # Create mask for the current panoptic segment and add channel
                # dim
                panoptic_mask = (panoptic_input == panoptic_id)[:, :, None]

                # Generate embedding for the segment
                image_embedding = estimator.predict(rgb_input, panoptic_mask)

                assert panoptic_id not in panoptic_embedding_dict
                panoptic_embedding_dict[panoptic_id] = image_embedding

            # Save embeddings for this sample
            dst_filename = dataset._get_filename(idx) + '.npz'
            dst_filepath = os.path.join(
                args.dataset_path,
                dataset.split,
                dataset.AUXILIARY_PANOPTIC_EMBEDDING_DIR_FMT.format(
                    str(semantic_n_classes), name
                ),
                dst_filename
            )

            create_dir(os.path.dirname(dst_filepath))

            # Ensure that the keys are strings as required for np.savez
            panoptic_embedding_dict = {
                str(key): value
                for key, value in panoptic_embedding_dict.items()
            }
            np.savez_compressed(dst_filepath, **panoptic_embedding_dict)

        # Generate text embeddings for semantic class names
        print(
            f"Predicting semantic class name embedding "
            f"({i+1}/{n_estimators} - {name})"
        )

        semantic_class_names = dataset.config.semantic_label_list.class_names
        # Use template for prompt engineering (CLIP works better with templates)
        template = DEFAULT_TEXT_EMBEDDING_TEMPLATE
        formatted_class_names = [
            template.format(name) for name in semantic_class_names
        ]

        text_embeddings = estimator._get_text_embedding(formatted_class_names)
        text_embeddings = text_embeddings.cpu().numpy()

        # Create mapping from class names to embeddings
        semantic_class_names_embedding_dict = {
            semantic_class_name: text_embedding
            for semantic_class_name, text_embedding in zip(
                formatted_class_names, text_embeddings
            )
        }

        # Store in the result dictionary using the dataset's format key
        name_key = \
            dataset.AUXILIARY_SEGMENTATION_CLASS_NAMES_EMITTER_FMT.format(
                str(semantic_n_classes), name
            )
        estimator_semantic_class_embedding_dicts[name_key] = \
            semantic_class_names_embedding_dict

        # Free up memory
        del estimator

    return estimator_semantic_class_embedding_dicts


def main(args=None) -> None:
    args = _parse_args(args)
    tmp_path = os.path.join(args.dataset_path, 'tmp')
    create_dir(tmp_path)

    auxiliary_meta = {}

    # TODO: We should check if the splits actually exist in the dataset
    splits_to_process = None
    dataset_class = get_dataset_class(args.dataset)
    if 'all' in args.split:
        splits_to_process = dataset_class.SPLITS
    else:
        for split in args.split:
            assert split in dataset_class.SPLITS, \
                f"Split '{split}' not available for dataset '{args.dataset}'."
        splits_to_process = args.split

    # Handle depth estimation auxiliary data -----------------------------------
    if (
        'depth' in args.auxiliary_data and
        is_depth_estimation_available(raise_error=True)
    ):
        depth_stats = {}
        print("Generating depth auxiliary data...")
        for split in splits_to_process:
            print(f"Processing split '{split}'")
            # Create the dataset object
            generate_depth(args=args, split=split, tmp_path=tmp_path)
            print("Computing depth stats...")
            depth_stats_split = \
                compute_depth_stats(args=args, split=split, tmp_path=tmp_path)
            # Update the dictionary with the new stats
            depth_stats = {
                key: {
                    **depth_stats.get(key, {}),
                    **depth_stats_split.get(key, {})
                }
                for key in depth_stats.keys() | depth_stats_split.keys()
            }

        auxiliary_meta['auxiliary_depth_estimators'] = args.depth_estimators
        auxiliary_meta['auxiliary_depth_stats'] = depth_stats

    # Handle image embedding auxiliary data ------------------------------------
    if (
        'image-embedding' in args.auxiliary_data and
        is_embedding_estimation_available(raise_error=True)
    ):
        print("Generating image embedding auxiliary data...")
        for split in splits_to_process:
            print(f"Processing split '{split}'")
            class_name_embeddings = generate_image_and_text_scene_embedding(
                args=args, split=split, tmp_path=tmp_path
            )

        # Store the class name embeddings in a file
        npy_file_path = os.path.join(
            args.dataset_path,
            'auxiliary_meta',
        )
        create_dir(npy_file_path)
        npy_file_path = os.path.join(
            npy_file_path,
            'scene_class_name_embeddings.npy'
        )
        relative_npy_file_path = os.path.join(
            'auxiliary_meta',
            'scene_class_name_embeddings.npy'
        )
        np.save(npy_file_path, class_name_embeddings)

        auxiliary_meta['auxiliary_image_embedding_estimators'] = \
            args.embedding_estimators
        auxiliary_meta['auxiliary_scene_class_name_embeddings'] = \
            relative_npy_file_path

    if 'panoptic-embedding' in args.auxiliary_data and \
            is_embedding_estimation_available(raise_error=True):
        semantic_class_name_embeddings = {}
        for semantic_n_classes in args.embedding_semantic_n_classes:
            print(
                f"Generating panoptic embedding auxiliary data for "
                f"{semantic_n_classes} semantic classes..."
            )
            void_label = 0
            max_instances_per_category = (1 << 16)
            for split in splits_to_process:
                print(f"Processing split '{split}'")
                # Some datasets only have one valid class spectrum for semantic
                # and don't expect the user to provide the number of classes
                # as an argument.
                semantic_n_classses_arg = None
                if len(args.embedding_semantic_n_classes) > 1:
                    semantic_n_classses_arg = semantic_n_classes

                current_semantic_class_name_embeddings = \
                    generate_panoptic_and_text_class_name_embedding(
                        args=args, split=split, tmp_path=tmp_path,
                        semantic_n_classes=semantic_n_classses_arg,
                        void_label=void_label,
                        max_instances_per_category=max_instances_per_category
                    )
                # Might happen if some sample keys were missing for the current
                # combination of dataset and split.
                if current_semantic_class_name_embeddings is None:
                    continue
                semantic_class_name_embeddings.update(
                    current_semantic_class_name_embeddings
                )

        # Store the semantic class name embeddings in a file
        npy_file_path = os.path.join(
            args.dataset_path,
            'auxiliary_meta',
        )
        create_dir(npy_file_path)
        npy_file_path = os.path.join(
            npy_file_path,
            'semantic_class_name_embeddings.npy'
        )
        relative_npy_file_path = os.path.join(
            'auxiliary_meta',
            'semantic_class_name_embeddings.npy'
        )
        np.save(npy_file_path, semantic_class_name_embeddings)

        auxiliary_meta['auxiliary_panoptic_embedding_estimators'] = \
            args.embedding_estimators
        auxiliary_meta['auxiliary_panoptic_embedding_void_label'] = void_label
        auxiliary_meta['auxiliary_panoptic_embedding_semantic_n_classes'] = \
            args.embedding_semantic_n_classes
        auxiliary_meta[
            'auxiliary_panoptic_embedding_max_instances_per_category'
        ] = \
            max_instances_per_category
        auxiliary_meta['auxiliary_semantic_class_name_embeddings'] = \
            relative_npy_file_path

    create_or_update_creation_metafile(args.dataset_path, **auxiliary_meta)

    # Clean up
    shutil.rmtree(tmp_path)


if __name__ == "__main__":
    main()
