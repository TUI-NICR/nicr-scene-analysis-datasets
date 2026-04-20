#!/bin/bash

# usage:
#
# OUTPUT_PATH="/local/test_location_for_nicr_scene_analysis_datasets"
# FORCE_RECREATE=true
# RUN_TESTS=true
# CLEAN_UP=true
#
# source test_datasets.sh
#
# test_*

# defaults
# OUTPUT_PATH="${OUTPUT_PATH:-"/local/test_location_for_nicr_scene_analysis_datasets"}"
OUTPUT_PATH="${OUTPUT_PATH:-"/datasets_nas/test_location_for_nicr_scene_analysis_datasets"}"
FORCE_RECREATE="${FORCE_RECREATE:-true}"
RUN_TESTS="${RUN_TESTS:-true}"
RUN_COMPARISONS="${RUN_COMPARISONS:-true}"
CLEAN_UP="${CLEAN_UP:-true}"


# Helpers --------------------------------------------------------------------------------------------------------------
prepare_dataset() {
    local dataset_dirname="$1"
    local additional_args=("${@:2}")

    local dataset_name="${1%%_*}"  # everything up to first _
    local dataset_path="${OUTPUT_PATH}/${dataset_dirname}"

    # DO NOT accidentally overwrite datasets in /datasets_nas/nicr_scene_analysis_datasets
    if [[ "${dataset_path}" == *"/datasets_nas/nicr_scene_analysis_datasets"* ]]; then
        echo "ERROR: Refusing to create dataset at '${dataset_path}'"
        return 1
    fi

    # remove existing dataset if FORCE_RECREATE is true
    if [ -d "${dataset_path}" ] && [ "${FORCE_RECREATE}" = true ]; then
        echo "Dataset already exists at: '${dataset_path}', removing due to FORCE_RECREATE=true"
        rm -rf "${dataset_path}"
    fi

    # prepare dataset if it does not exist
    if [ ! -d "${dataset_path}" ]; then
        echo "Preparing ${dataset_name} dataset at: '${dataset_path}'"
        nicr_sa_prepare_dataset \
            "${dataset_name}" \
            "${dataset_path}" \
            "${additional_args[@]}"
    else
        echo "Dataset already exists at: '${dataset_path}', skipping preparation"
    fi
}

generate_auxiliary_data() {
    local dataset_dirname="$1"
    local additional_args=("${@:2}")

    local dataset_name="${1%%_*}"  # everything up to first _
    local dataset_path="${OUTPUT_PATH}/${dataset_dirname}"

    # DO NOT accidentally overwrite auxiliary data in production datasets
    if [[ "${dataset_path}" == *"/datasets_nas/nicr_scene_analysis_datasets"* ]]; then
        echo "ERROR: Refusing to generate auxiliary data at '${dataset_path}'"
        return 1
    fi

    # check if dataset exists
    if [ ! -d "${dataset_path}" ]; then
        echo "ERROR: Dataset does not exist at '${dataset_path}', cannot generate auxiliary data"
        return 1
    fi

    echo "Generating auxiliary data for ${dataset_name} at: '${dataset_path}'"
    nicr_sa_generate_auxiliary_data \
        --dataset "${dataset_name}" \
        --dataset-path "${dataset_path}" \
        --auxiliary-data depth image-embedding panoptic-embedding \
        --embedding-estimator-device cuda \
        --embedding-estimators alpha_clip__l14-336-grit-20m \
        --depth-estimator-device cuda \
        --depth-estimators depthanything_v2__indoor_large \
        --cache-models \
        "${additional_args[@]}"
}

run_tests() {
    local dataset_dirname="$1"
    local dataset_name="${1%%_*}"  # everything up to first _

    if [ "${RUN_TESTS}" = true ]; then
        echo "Running tests for dataset: '${dataset_dirname}'"

        # create symlink if dataset_name != dataset_dirname
        if [ "${dataset_name}" != "${dataset_dirname}" ]; then
            echo "Creating symlink from '${dataset_name}' to '${dataset_dirname}'"
            rm -f "${OUTPUT_PATH}/${dataset_name}"  # remove any existing symlink, do NOTHING if it is folder
            ln -s "./${dataset_dirname}" "${OUTPUT_PATH}/${dataset_name}"
        fi

        # test files are relative to this file, so get path to this file
        local script_path="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

        # run tests
        NICR_SA_DATASET_BASEPATH="${OUTPUT_PATH}" py.test "${script_path}/test_${dataset_name}.py" -rx -s -vv \
            2>&1 | tee ${OUTPUT_PATH}/test_${dataset_dirname}.log

        # remove symlink
        rm -f "${OUTPUT_PATH}/${dataset_name}"  # remove any existing symlink, do NOTHING if it is folder
    else
        echo "Skipping tests for dataset: '${dataset_dirname}'"
    fi
}

compare_datasets() {
    local dataset_dirname="$1"
    local reference_dataset_path="$2"
    # drop dataset_dirname and reference path, keep remaining args for passthrough
    shift 2

    local dataset_path="${OUTPUT_PATH}/${dataset_dirname}"

    if [ "${RUN_COMPARISONS}" = true ]; then
        echo "Comparing dataset: '${dataset_dirname}' ('${dataset_path}') against reference at: '${reference_dataset_path}'"
        local script_path="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"
        python "${script_path}/compare_dataset_paths.py" \
            "${dataset_path}" \
            "${reference_dataset_path}" \
            --checksum \
            --disable-progress-bar \
            "$@" \
            | tee ${OUTPUT_PATH}/compare_${dataset_dirname}.log
    else
        echo "Skipping comparison for dataset: '${dataset_dirname}' ('${dataset_path}')"
    fi
}

remove_dataset() {
    local dataset_dirname="$1"

    local dataset_path="${OUTPUT_PATH}/${dataset_dirname}"

    if [ "${CLEAN_UP}" = true ]; then
        # DO NOT accidentally delete datasets in /datasets_nas/nicr_scene_analysis_datasets
        if [[ "${dataset_path}" == *"/datasets_nas/nicr_scene_analysis_datasets"* ]]; then
            echo "ERROR: Refusing to delete dataset at: '${dataset_path}'"
            return 1
        fi

        echo "Cleaning up: '${dataset_path}'"
        rm -rf "${dataset_path}"
    fi
}


# ADE20K ---------------------------------------------------------------------------------------------------------------
test_ade20k() {
    # ADE20K dataset (~31 GB)

    prepare_dataset ade20k \
        --challenge-2016-filepath /datasets_nas/segmentation/ade20k/ADEChallengeData2016.zip \
        --challenge-2017-instances-filepath /datasets_nas/segmentation/ade20k/annotations_instance.tar \
        --n-processes 16

    compare_datasets ade20k \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/ade20k

    run_tests ade20k

    remove_dataset ade20k
}


# Cityscapes -----------------------------------------------------------------------------------------------------------
test_cityscapes() {
    # Cityscapes dataset (~35 GB)

    prepare_dataset cityscapes \
        /datasets_nas2b/segmentation/cityscapes/

    compare_datasets cityscapes \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/cityscapes

    run_tests cityscapes

    remove_dataset cityscapes
}


# COCO -----------------------------------------------------------------------------------------------------------------
test_coco() {
    # coco dataset (~22 GB)

    prepare_dataset coco \
        --do-not-delete-zip \
        --download-path /datasets_nas/segmentation/coco/

    compare_datasets coco \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/coco

    run_tests coco

    remove_dataset coco
}


# Hypersim -------------------------------------------------------------------------------------------------------------
test_hypersim_tilt_shift() {
    # Hypersim dataset with tilt-shift conversion as used in newer works (~142 GB)

    prepare_dataset hypersim_tilt_shift \
        /datasets_nas/segmentation/hypersim/apple-hypersim \
        --additional-subsamples 2 5 10 20 \
        --n-processes 16

    compare_datasets hypersim_tilt_shift \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/hypersim \
        --hypersim-relax-rgb-check

    run_tests hypersim_tilt_shift

    remove_dataset hypersim_tilt_shift
}

test_hypersim_no_tilt_shift() {
    # Hypersim dataset without tilt-shift conversion as used in EMSANet and EMSAFormer (~142 GB)

    prepare_dataset hypersim_no_tilt_shift \
        /datasets_nas/segmentation/hypersim/apple-hypersim \
        --additional-subsamples 2 5 10 20 \
        --n-processes 16 \
        --no-tilt-shift-conversion

    # there were some breaking changes from v050 to v051 which is why
    # --hypersim-legacy-equivalence is passed to compare beyond file level
    compare_datasets hypersim_no_tilt_shift \
        /datasets_nas/nicr_scene_analysis_datasets/version_050/hypersim \
        --hypersim-legacy-equivalence \
        --hypersim-relax-rgb-check

    run_tests hypersim_no_tilt_shift

    remove_dataset hypersim_no_tilt_shift
}

test_hypersim() {
    test_hypersim_tilt_shift
    test_hypersim_no_tilt_shift
}

test_mapping_hypersim() {
    # Hypersim dataset with tilt-shift conversion (~142 GB)

    prepare_dataset hypersim_mapping \
        /datasets_nas/segmentation/hypersim/apple-hypersim \
        --additional-subsamples 2 5 10 20 \
        --n-processes 16

    # extract 3D ground truth (plys and ScanNet benchmark format)
    local dataset_path="${OUTPUT_PATH}/hypersim_mapping"
    # local dataset_path="/datasets_nas/nicr_scene_analysis_datasets/version_084/hypersim"
    local dataset_3d_path="${OUTPUT_PATH}/hypersim_mapping_3d"

    # valid: 15GB, test: 15GB
    echo "Preparing hypersim_mapping_3d dataset at: '${dataset_3d_path}'"
    for SPLIT in "valid" "test"
    do
        nicr_sa_prepare_labeled_point_clouds \
            hypersim \
            "${dataset_path}" \
            "${dataset_3d_path}" \
            --split ${SPLIT} \
            --voxel-size 0.01 \
            --max-depth 20 \
            --write-scannet-benchmark-ground-truth
    done

    compare_datasets hypersim_mapping_3d \
        /datasets_nas/nicr_scene_analysis_datasets/version_053/hypersim_3d \
        --hypersim-relax-rgb-check

    remove_dataset hypersim_mapping
    remove_dataset hypersim_mapping_3d
}


# NYUv2 ----------------------------------------------------------------------------------------------------------------
test_nyuv2() {
    # NYUv2 dataset (~2 GB)

    prepare_dataset nyuv2_plain \
        --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat \
        --enable-normal-extraction \
        --normal-filepath /datasets_nas/segmentation/nyuv2/normals_gt.tgz

    compare_datasets nyuv2_plain \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/nyuv2

    run_tests nyuv2_plain

    # TODO: we require NYUv2 for SUNRGB-D and it is small, so do not clean up here?
    remove_dataset nyuv2_plain
}


test_nyuv2_with_auxiliary_data() {
    # NYUv2 dataset (~2 GB)

    prepare_dataset nyuv2_with_auxiliary_data \
        --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat \
        --enable-normal-extraction \
        --normal-filepath /datasets_nas/segmentation/nyuv2/normals_gt.tgz

    # generate auxiliary data
    # note, only done for nyuv2 as its the smallest dataset and thus fastest to process
    generate_auxiliary_data nyuv2_with_auxiliary_data \
        --embedding-semantic-n-classes 40

    compare_datasets nyuv2_with_auxiliary_data \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/nyuv2

    run_tests nyuv2_with_auxiliary_data

    # TODO: we require NYUv2 for SUNRGB-D and it is small, so do not clean up here?
    remove_dataset nyuv2_with_auxiliary_data
}


# ScanNet --------------------------------------------------------------------------------------------------------------
test_scannet() {
    # ScanNet dataset as used for image segmentation experiments (~33 GB)

    prepare_dataset scannet_plain \
        /datasets_nas/segmentation/ScanNet/ \
        --n-processes 16 \
        --subsample 50 \
        --additional-subsamples 100 200 500

    compare_datasets scannet_plain \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/scannet

    run_tests scannet_plain

    remove_dataset scannet_plain
}

test_mapping_scannet() {
    # ScanNet dataset as used for mapping experiments in PanopticNDT (~300 GB)

    # prepare dataset
    prepare_dataset scannet_mapping \
        /datasets_nas/segmentation/ScanNet/ \
        --n-processes 16 \
        --subsample 5 \
        --additional-subsamples 10 50 100 200 500

    # extract 3D ground truth (plys and ScanNet benchmark format)
    local dataset_path="${OUTPUT_PATH}/scannet_mapping"
    # local dataset_path="/datasets_nas/nicr_scene_analysis_datasets/version_051_scannet_subsample_5_mapping"
    local dataset_3d_path="${OUTPUT_PATH}/scannet_mapping_3d"

    local scannet_split_files_path=$(python -c 'import os; from nicr_scene_analysis_datasets.datasets import scannet; print(os.path.dirname(scannet.__file__))')
    local scannet_download_path="/datasets_nas/segmentation/ScanNet"

    local current_dir=$(pwd)
    local script_path="$(dirname "$(realpath "${BASH_SOURCE[0]}")")"

    cd "${script_path}/../external/ScanNet/BenchmarkScripts/3d_helpers"
    pip install imageio

    # valid: 2.3GB, test: 0.7GB (no ground truth)
    echo "Preparing scannet_mapping_3d dataset at: '${dataset_3d_path}'"
    for SPLIT in "valid" "test"
    do
        python extract_scannet_ground_truth.py \
            "${scannet_download_path}" \
            "${scannet_split_files_path}" \
            "${scannet_download_path}/scannetv2-labels.combined.tsv" \
            "${dataset_3d_path}" \
            --split ${SPLIT} \
            --shift $((2**16))
    done

    cd $current_dir

    compare_datasets scannet_mapping \
        /datasets_nas/nicr_scene_analysis_datasets/version_051_scannet_subsample_5_mapping

    compare_datasets scannet_mapping_3d \
        /datasets_nas/nicr_scene_analysis_datasets/version_051_scannet_subsample_5_mapping_3d

    remove_dataset scannet_mapping
    remove_dataset scannet_mapping_3d
}


# SceneNet RGB-D -------------------------------------------------------------------------------------------------------
test_scenenet_rgbd() {
    # SceneNet RGB-D dataset (~5 GB)

    # (re)compile the protobuf file
    echo "(Re)compiling SceneNet RGB-D protobuf file"
    local proto_path=$(python -c "import os; from nicr_scene_analysis_datasets.datasets import scenenetrgbd; print(os.path.dirname(scenenetrgbd.__file__))")
    cd $proto_path
    protoc --python_out=./ scenenet.proto
    cd -

    prepare_dataset scenenetrgbd \
        /datasets_nas/segmentation/SceneNetRGBD \
        --n-random-views-to-include-train 3 \
        --n-random-views-to-include-valid 6  \
        --force-at-least-n-classes-in-view 4

    compare_datasets scenenetrgbd \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/scenenetrgbd

    run_tests scenenetrgbd

    remove_dataset scenenetrgbd
}


# SUNRGB-D -------------------------------------------------------------------------------------------------------------
test_sunrgbd_emsanet() {
    # SUNRGB-D in EMSANet version (~4 GB)

    # we require NYUv2 to be already prepared
    local nyuv2_path="${OUTPUT_PATH}/nyuv2_plain"
    if [ ! -d "${nyuv2_path}" ]; then
        prepare_dataset nyuv2_plain --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat
    fi
    echo "Using prepared NYUv2 dataset at: '${nyuv2_path}'"

    prepare_dataset sunrgbd_emsanet \
        --create-instances \
        --instances-version emsanet \
        --copy-instances-from-nyuv2 \
        --nyuv2-path "${nyuv2_path}" \
        --toolbox-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDtoolbox.zip \
        --data-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBD.zip \
        --box-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDMeta3DBB_v2.mat

    compare_datasets sunrgbd_emsanet \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/sunrgbd_emsanet

    run_tests sunrgbd_emsanet

    remove_dataset sunrgbd_emsanet
}

test_sunrgbd_panopticndt() {
    # SUNRGB-D in PanopticNDT version (~4 GB)

    # we require NYUv2 to be already prepared
    local nyuv2_path="${OUTPUT_PATH}/nyuv2_plain"
    if [ ! -d "${nyuv2_path}" ]; then
        prepare_dataset nyuv2_plain --mat-filepath /datasets_nas/segmentation/nyuv2/nyu_depth_v2_labeled.mat
    fi
    echo "Using prepared NYUv2 dataset at: '${nyuv2_path}'"

    prepare_dataset sunrgbd_panopticndt \
        --create-instances \
        --instances-version panopticndt \
        --copy-instances-from-nyuv2 \
        --nyuv2-path "${nyuv2_path}" \
        --toolbox-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDtoolbox.zip \
        --data-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBD.zip \
        --box-filepath /datasets_nas/segmentation/SunRGBD/SUNRGBDMeta3DBB_v2.mat

    compare_datasets sunrgbd_panopticndt \
        /datasets_nas/nicr_scene_analysis_datasets/version_083/sunrgbd_panopticndt

    run_tests sunrgbd_panopticndt

    remove_dataset sunrgbd_panopticndt
}

test_sunrgbd() {
    test_sunrgbd_emsanet
    test_sunrgbd_panopticndt
}

test_all_local() {
    DATE="$(date +%Y-%m-%d_%H-%M-%S)"
    OUTPUT_PATH="/datasets_nas2/test_location_for_nicr_scene_analysis_datasets_local_${DATE}"
    FORCE_RECREATE=true
    RUN_TESTS=true
    RUN_COMPARISONS=true
    CLEAN_UP=false

    test_ade20k
    test_cityscapes
    test_coco
    test_hypersim
    test_mapping_hypersim
    test_nyuv2
    test_nyuv2_with_auxiliary_data
    test_scannet
    test_mapping_scannet
    test_scenenet_rgbd
    test_sunrgbd
}
