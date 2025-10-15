# ADE20K dataset
This dataset provides access to the ADE20K-based semantic segmentation benchmarks, primarily focusing on the MIT Scene Parse Benchmark Challenge 2016 and Places Challenge 2017. 
The 2016 Challenge data contains over 20K scene-centric images with pixel-level semantic annotations for 150 object categories, including both stuff classes (sky, road, grass) and thing classes (person, car, bed). 
The 2017 Places Challenge adds instance segmentation annotations for the same images.

For more details about the challenges, see: [MIT Scene Parse Benchmark Challenge 2016](http://sceneparsing.csail.mit.edu/)

## Dataset versions
There are two main versions of the dataset which are currently supported in this package:

1. [MIT Scene Parse Benchmark Challenge 2016 - MIT SceneParse150](http://sceneparsing.csail.mit.edu/): 
- see [GitHub](https://github.com/CSAILVision/sceneparsing)
- benchmark that contains a subset of images and labels of the ADE20K dataset
- is often misleadingly referred to as 'ADE20K' in literature as the data come from the ADE20K dataset
- contains 20,210 training images and 2,000 images for validation, for which semantic annotations and scene classes are available; in addition to that, there is a test set containing 3,352 images
- 150 semantic classes - result from selecting the 150 most frequent object classes from ADE20K (which contains 3600+ classes) - annotations that are not part of this 150 class subset are ignored (treated as background/void)
- usually used for semantic segmentation tasks
- images taken from ADE20K are rescaled so that their longer side is at most 512px

2. [Places Challenge 2017](http://placeschallenge.csail.mit.edu/)
- see [GitHub](https://github.com/CSAILVision/placeschallenge)
- adds instance segmentation annotations to the challenge data from 2016 (above)
- can be combined to panoptic segmentation data (100 thing classes and 50 stuff classes)

### Additional dataset versions (currently not supported)
3. [ADE20K Dataset 2021](https://groups.csail.mit.edu/vision/datasets/ADE20K/):
- version from January 17, 2021 - most recent version of the dataset
- 27,574 images (25,574 for training and 2,000 for testing/validation) - therefore, containing about 5K additional training images:
  - images come in various sizes, some of them larger than the 512px mentioned above
  - semantic, instance, scene and part annotations
  - primarily used for open vocabulary segmentation
 
## Prepare dataset

1. Download and convert the dataset to the desired format:
  The Challenge data can be downloaded without having to register.
  Therefore, the prepare script can handle the download. 
  If you prefer to download the files yourself, you may provide the path to the archives of the challenge data with `--challenge-2016-filepath` AND `--challenge-2017-instances-filepath` (instance annotations are given as a separate download).

  ```bash
  # general usage
  nicr_sa_prepare_dataset ade20k \
      /path/where/to/store/ade20k/ \
      [--challenge-2016-filepath] \
      [--challenge-2017-instances-filepath] \
      [--n-processes N]
  ```

  With arguments:
  - `--challenge-2016-filepath`:
    Path to the '2016 Scene Parse Benchmark Challenge' zip file (ADEChallengeData2016.zip).
  - `--challenge-2017-instances-filepath`:
    Path to the tar file containing the instance annotations of the '2017 Places Challenge' tar file (annotations_instance.tar).
  - `--n-processes`:
      The number of worker processes to spawn.

2. (Optional) Generate auxiliary data
    > **Note**: To use auxiliary data generation, the package must be installed with the `withauxiliarydata` option:
    > ```bash
    > pip install -e .[withauxiliarydata]
    > ```

    ```bash
    # for auxiliary data such as synthetic depth and rgb/panoptic embeddings
    nicr_sa_generate_auxiliary_data \
        --dataset ade20k \
        --dataset-path /path/to/already/prepared/ade20k/dataset \
        --auxiliary-data depth image-embedding panoptic-embedding \
        --embedding-estimator-device cuda \
        --embedding-estimators alpha_clip__l14-336-grit-20m \
        --depth-estimator-device cuda \
        --depth-estimators depthanything_v2__indoor_large \
        --cache-models
    ```
    With arguments:
    - `--dataset-path`:
      Path to the prepared ADE20k dataset.
    - `--auxiliary-data`:
      Types of auxiliary data to generate:
        - `depth`: Generates synthetic depth images from RGB.
        - `image-embedding`: Uses Alpha-CLIP to generate an embedding for the entire image.
        - `panoptic-embedding`: Uses Alpha-CLIP to generate an embedding for each panoptic mask.
    - `--depth-estimator-device`:
      Device to use for depth estimation (`cpu` or `cuda`).
    - `--depth-estimators`:
      Depth estimator(s) to use. Use `depthanything_v2__indoor_large` to match DVEFormer.
    - `--embedding-estimator-device`:
      Device to use for embedding estimation (`cpu` or `cuda`).
    - `--embedding-estimators`:
      Embedding estimator(s) to use. Use `alpha_clip__l14-336-grit-20m` to match DVEFormer.
    - `--cache-models`:
      Cache models locally to avoid reloading them in future runs.

