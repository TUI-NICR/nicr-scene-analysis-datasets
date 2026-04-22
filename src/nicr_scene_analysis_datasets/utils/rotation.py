from importlib import metadata
from packaging.version import Version

import numpy as np
from scipy.spatial.transform import Rotation as _Rotation


SCIPY_VERSION = Version(metadata.version("scipy"))


class PatchedSciPyRotation(_Rotation):
    """
    Subclass of scipy.spatial.transform.Rotation to enforce the old behavior
    for from_matrix.

    Notes
    -----
    Starting with SciPy 1.15.2, the behavior of
    scipy.spatial.Rotation.from_matrix has changed. If the given input matrix
    is not perfectly (they are testing with atol=1e-12 !!) orthogonal
    (mat^T*mat = 1), an approximation using orthogonal Procrustes projection
    (basically SVD) is created. Afterward, the matrix is converted to
    quaternions. These quaternions are normalized, and serve as an internal
    representation in the Rotation object.
    In versions earlier than 1.15.2, no such orthogonalization step was
    performed; only the quaternion normalization compensated slightly for
    non-orthogonal input matrices.

    While the new handling is probably mathematically more correct and
    stable (this change makes sense in general), it also has a huge impact on
    our existing pipelines in terms of both results and runtime. Rotation
    matrices (especially from estimated extrinsics) are often not perfectly
    orthogonal and, thus, will be silently modified by this new behavior.

    As of 2026-01-08, there is no way to enforce the old behavior in
    SciPy >= 1.15.2. The upcoming SciPy 1.17.0 (currently RC2) will introduce
    a keyword argument 'assume_valid' with default value False, mainly to skip
    the expensive orthogonalization step, but at the same time enabling the
    old behavior. However, as the argument is unknown in previous versions,
    simply passing assume_valid=True will break older environments.

    Therefore, we implement the old `from_matrix` behavior for
    1.15.2 <= SciPy < 1.17.0 here, and use the official implementation for
    >= 1.17.0. The resulting class
    `nicr_scene_analysis_datasets.utils.rotation.PatchedSciPyRotation` can be
    used as drop-in replacement for `scipy.spatial.transform.Rotation`.

    References
    ----------
    - https://github.com/scipy/scipy/issues/22417
    - https://github.com/scipy/scipy/pull/22418
    - https://github.com/scipy/scipy/pull/24092

    """
    if SCIPY_VERSION < Version("1.15.2"):

        @classmethod
        def from_matrix(cls, matrix, assume_valid=True):
            if not assume_valid:
                raise ValueError(
                    "Changing `assume_valid` requires SciPy >= 1.17.0. "
                    f"You are using SciPy {SCIPY_VERSION}."
                )
            return super().from_matrix(matrix)

    elif Version("1.15.2") <= SCIPY_VERSION < Version("1.17.0"):

        @classmethod
        def from_matrix(cls, matrix, assume_valid=True):
            if not assume_valid:
                raise ValueError(
                    "Changing `assume_valid` requires SciPy >= 1.17.0. "
                    f"You are using SciPy {SCIPY_VERSION}."
                )

            # copied and adapted from:
            # https://github.com/scipy/scipy/blob/v1.15.1/scipy/spatial/transform/_rotation.pyx

            is_single = False
            matrix = np.asarray(matrix, dtype=float)

            if (matrix.ndim not in [2, 3] or
                matrix.shape[len(matrix.shape)-2:] != (3, 3)):
                raise ValueError("Expected `matrix` to have shape (3, 3) or "
                                 "(N, 3, 3), got {}".format(matrix.shape))

            if matrix.shape == (3, 3):
                matrix = matrix[None, :, :]
                is_single = True

            num_rotations = matrix.shape[0]

            decision = np.empty((4,), dtype=float)
            quat = np.empty((num_rotations, 4), dtype=float)
            for ind in range(num_rotations):
                decision[0] = matrix[ind, 0, 0]
                decision[1] = matrix[ind, 1, 1]
                decision[2] = matrix[ind, 2, 2]
                decision[3] = matrix[ind, 0, 0] + matrix[ind, 1, 1] \
                            + matrix[ind, 2, 2]
                choice = np.argmax(decision)  # _argmax4(decision)

                if choice != 3:
                    i = choice
                    j = (i + 1) % 3
                    k = (j + 1) % 3

                    quat[ind, i] = 1 - decision[3] + 2 * matrix[ind, i, i]
                    quat[ind, j] = matrix[ind, j, i] + matrix[ind, i, j]
                    quat[ind, k] = matrix[ind, k, i] + matrix[ind, i, k]
                    quat[ind, 3] = matrix[ind, k, j] - matrix[ind, j, k]
                else:
                    quat[ind, 0] = matrix[ind, 2, 1] - matrix[ind, 1, 2]
                    quat[ind, 1] = matrix[ind, 0, 2] - matrix[ind, 2, 0]
                    quat[ind, 2] = matrix[ind, 1, 0] - matrix[ind, 0, 1]
                    quat[ind, 3] = 1 + decision[3]

                # normalize
                quat[ind] /= np.linalg.norm(quat[ind])  # _normalize4(...])

                if is_single:
                    return cls(quat[0], normalize=False, copy=False)
                else:
                    return cls(quat, normalize=False, copy=False)
