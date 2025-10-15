
# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
from typing import Dict, Callable, Optional

from importlib import metadata
from importlib import util as importlib_util
from importlib.abc import MetaPathFinder
from packaging.version import Version
import sys
import warnings


class DependencyImportHook(MetaPathFinder):
    def __init__(self, module_handlers: Dict[str, Callable]) -> None:
        self._module_handlers = module_handlers

    def find_spec(self, fullname, path, target=None):
        if fullname in self._module_handlers:
            # call the handler to check the dependency
            self._module_handlers[fullname]()

        # continue with normal import
        return None


def is_package_available(
    package_name: str,
    raise_error: bool = True,
    min_version: Optional[str] = None,
    additional_error_msg: Optional[str] = None
) -> bool:
    if min_version is not None:
        required_package_str = f"{package_name}>={min_version}"
    else:
        required_package_str = f"{package_name}"

    # remove potentially existing instances of our DependencyImportHook
    # to avoid infinite recursion
    dependency_check_hooks = [
        hook
        for hook in sys.meta_path
        if isinstance(hook, DependencyImportHook)
    ]
    for hook in dependency_check_hooks:
        sys.meta_path.remove(hook)

    installed_str = ''
    try:
        # check if a spec for the package exists without actually importing it

        # note, we purposely avoid importlib.import_module here as this function
        # (is_package_available) might be called from inside a meta-path
        # finder (DependencyImportHook); that can lead to the same package being
        # initialized reentrantly (effectively imported twice or partially
        # twice), which can produce duplicate global registrations or other
        # side effects; for packages that register global state at import time
        # (PyTorch does this with TORCH_LIBRARY), that double initialization
        # raises errors

        spec = importlib_util.find_spec(package_name)
        if spec is None:
            # not installed / not importable
            raise ImportError(f"Cannot find spec for '{package_name}'")

        # try to determine version without importing the package
        version = None
        try:
            # map top-level package name to distribution name(s) - this helps
            # for cv2/opencv-python
            dists = metadata.packages_distributions().get(package_name)
            if dists:
                # take first distribution providing the package
                # TODO: this might not be the correct one if multiple
                #       distributions exist?
                version = metadata.version(dists[0])
            else:
                # fallback: try to get a distribution/version with the same name
                version = metadata.version(package_name)
        except Exception:
            # we could not determine a distribution/version without importing
            warnings.warn(
                f"Cannot determine version for '{package_name}' without "
                "importing. Version checks will be skipped."
            )
            version = None

        if min_version is not None and version is not None:
            if Version(version) < Version(min_version):
                installed_str = f" (found version: {version})"

    except ImportError:
        if raise_error:
            raise ImportError(
                "nicr-scene-analysis-datasets requires "
                f"'{required_package_str}'{installed_str}. "
                f"{additional_error_msg or ''}"
            )
        return False

    finally:
        # re-insert the hooks
        sys.meta_path[:0] = dependency_check_hooks

    return True


def is_opencv_available(
    raise_error: bool,
    min_version: Optional[str] = None
) -> bool:
    # might be NVIDIA's opencv, apt's python3-opencv, or opencv-python from PyPI
    return is_package_available(
        package_name='cv2',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred OpenCV version yourself or "
            "re-install the nicr-scene-analysis-datasets package with the "
            "additional 'withopencv' target to install a default version, "
            "i.e., `pip install nicr-scene-analysis-datasets[withopencv]`."
        )
    )


def is_torch_available(
    raise_error: bool,
    min_version: Optional[str] = None
) -> bool:
    return is_package_available(
        package_name='torch',
        raise_error=raise_error,
        min_version=min_version,
        additional_error_msg=(
            "Please install your preferred PyTorch version yourself or "
            "re-install the nicr-scene-analysis-datasets package with the "
            "additional 'withtorch' target to install a default version, "
            "i.e., `pip install nicr-scene-analysis-datasets[withtorch]`."
        )
    )


def is_depth_estimation_available(raise_error: bool) -> bool:
    if not is_torch_available(
        raise_error=raise_error,
        min_version='2.3.1'  # DepthAnythingV2 requires torch.nn.RMSNorm
    ):
        return False

    # we further need transformers from Hugging Face
    return is_package_available(
        package_name='transformers',
        raise_error=raise_error,
        min_version=None,  # TODO: might need to check for specific versions
        additional_error_msg=(
            "Please re-install the nicr-scene-analysis-datasets package with "
            "the additional 'withdepthestimation' target, i.e., "
            "`pip install nicr-scene-analysis-datasets[withdepthestimation]`."
        )
    )


def is_embedding_estimation_available(raise_error: bool) -> bool:
    if not is_torch_available(
        raise_error=raise_error,
        min_version=None,  # TODO: might need to check for specific versions
    ):
        return False

    # we further need alpha_clip
    return is_package_available(
        package_name='alpha_clip',
        raise_error=raise_error,
        min_version=None,  # TODO: might need to check for specific versions
        additional_error_msg=(
            "Please re-install the nicr-scene-analysis-datasets package with "
            "the additional 'withembeddingestimation' target, i.e., "
            "`pip install nicr-scene-analysis-datasets[withembeddingestimation]`."
        )
    )


def install_nicr_scene_analysis_datasets_dependency_import_hooks():
    sys.meta_path.insert(
        0,
        DependencyImportHook(
            module_handlers={
                "cv2": lambda: is_opencv_available(raise_error=True),
                "torch": lambda: is_torch_available(raise_error=True),
            }
        )
    )
