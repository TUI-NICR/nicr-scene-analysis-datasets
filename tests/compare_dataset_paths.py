# -*- coding: utf-8 -*-
"""
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import argparse
from collections import Counter
import hashlib
import os
import subprocess
import glob
import json
import re

import cv2
import numpy as np
import open3d as o3d
import plyfile
from tqdm import tqdm

from nicr_scene_analysis_datasets.utils.rotation import PatchedSciPyRotation
try:
    import termcolor

    cprint = termcolor.cprint
    can_colorize = termcolor.can_colorize

except ImportError:

    def cprint(*args, **kwargs):
        kwargs.pop('color', None)
        kwargs.pop('on_color', None)
        kwargs.pop('attrs', None)

        print(*args, **kwargs)

    def can_colorize():
        return False


def _parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Compare contents of two dataset paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('dataset_path1', help="First dataset path")
    parser.add_argument('dataset_path2', help="Second dataset path")

    parser.add_argument(
        '--include-regex',
        action="append",
        default=[],
        help="Only include files whose relative path matches this regex "
             "(can be given multiple times; default: include all).",
    )
    parser.add_argument(
        '--ignore-regex',
        action="append",
        default=[r'creation_meta\.json'],
        help="Ignore files whose relative path matches this regex "
             "(can be given multiple times).",
    )

    parser.add_argument(
        '--checksum',
        action="store_true",
        default=False,
        help="Use checksums instead of file size for comparison.",
    )

    parser.add_argument(
        '--text-diff',
        action="store_true",
        default=False,
        help="Perform a textual diff for text files.",
    )

    parser.add_argument(
        '--ply-eps',
        type=float,
        default=1e-3,
        help="Distance threshold for PLY point matching.",
    )

    parser.add_argument(
        '--force-deep-compare',
        action="store_true",
        default=False,
        help="Force deep comparison of all common files, even if size matches.",
    )

    parser.add_argument(
        '--disable-progress-bar',
        action="store_true",
        default=False,
        help="Disable progress bars.",
    )
    parser.add_argument(
        '--hypersim-legacy-equivalence',
        action="store_true",
        default=False,
        help="Relax Hypersim comparisons for documented breaking changes "
             "(instance encoding/extrinsics rotation) while preserving "
             "equivalent in-memory content.",
    )
    parser.add_argument(
        '--hypersim-relax-rgb-check',
        action="store_true",
        default=False,
        help="Allow per-channel RGB differences of +-1 for Hypersim.",
    )

    return parser.parse_args(args)


def _collect_files(
    dataset_path,
    include_regexes,
    ignore_regexes,
    disable_progress_bar=False
):
    files = {}

    include_patterns = [re.compile(r) for r in include_regexes]
    ignore_patterns = [re.compile(r) for r in ignore_regexes]

    if not disable_progress_bar:
        tbar = tqdm(
            unit=" files",
            desc=f"Collecting files in '{dataset_path}'"
        )
    else:
        print(f"Collecting files in '{dataset_path}' ...")

    for abs_path in glob.iglob(
        os.path.join(dataset_path, '**', '*'),
        recursive=True
    ):
        if not disable_progress_bar:
            tbar.update(1)

        if not os.path.isfile(abs_path):
            continue

        rel_path = os.path.relpath(abs_path, dataset_path)

        # include check (default: include all)
        if include_patterns:
            matched = False
            for rx in include_patterns:
                if rx.search(rel_path):
                    matched = True
                    break
            if not matched:
                continue

        # ignore check
        ignored = False
        for rx in ignore_patterns:
            if rx.search(rel_path):
                ignored = True
                break
        if ignored:
            continue

        files[rel_path] = abs_path

    return files


def _get_file_size(path):
    return os.path.getsize(path)


def _get_file_checksum(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _same_text_files(fp1, fp2, text_diff=False):
    if text_diff:
        # line-based comparison using diff
        print(f"\t[TEXT DIFF] {fp1} vs {fp2}")
        res = subprocess.run(['diff', '-u', fp1, fp2], check=False)

        if res.returncode == 0:  # 0: identical, 1: differs, >1: error
            print("\t[TEXT DIFF] Files are identical")
            return True

        return False

    # compute basic stats like line count and char count
    with open(fp1, 'r') as f:
        lines1 = f.readlines()
    with open(fp2, 'r') as f:
        lines2 = f.readlines()

    n_lines1 = len(lines1)
    n_lines2 = len(lines2)
    n_chars1 = sum(len(line) for line in lines1)
    n_chars2 = sum(len(line) for line in lines2)
    print(
        f"\t[TEXT] Lines: {n_lines1} vs {n_lines2}, "
        f"Chars: {n_chars1} vs {n_chars2}"
    )

    if n_lines1 == n_lines2 and n_chars1 == n_chars2:
        print("\t[TEXT] Files are identical")
        return True

    return False


def _compare_json_values(v1, v2, float_eps, path, diffs):
    if isinstance(v1, dict):
        keys1 = set(v1.keys())
        keys2 = set(v2.keys())

        for k in keys1 - keys2:
            diffs.append(f"{path}.{k}: key only in json1")
        for k in keys2 - keys1:
            diffs.append(f"{path}.{k}: key only in json2")

        for k in keys1 & keys2:
            _compare_json_values(v1[k], v2[k], float_eps, f"{path}.{k}", diffs)

    elif isinstance(v1, list):
        if len(v1) != len(v2):
            diffs.append(f"{path}: list length mismatch {len(v1)} vs {len(v2)}")

        for i, (a, b) in enumerate(zip(v1, v2)):
            _compare_json_values(a, b, float_eps, f"{path}[{i}]", diffs)

    elif isinstance(v1, (int, float)):
        if abs(v1 - v2) > float_eps:
            diffs.append(f"{path}: {v1} vs {v2} (Δ={abs(v1 - v2):.6g})")
    else:
        if v1 != v2:
            diffs.append(f"{path}: {v1} vs {v2}")


def _same_json_files(fp1, fp2, json_eps=1e-6):
    with open(fp1, 'r') as f:
        j1 = json.load(f)
    with open(fp2, 'r') as f:
        j2 = json.load(f)

    diffs = []
    _compare_json_values(j1, j2, json_eps, '$', diffs)

    if not diffs:
        print("\t[JSON] Files are identical")
        return True

    print(f"\t[JSON] {len(diffs)} differences:")
    for d in diffs:
        print("\t[JSON]\t" + d)

    return False


def _same_image_files(fp1, fp2, max_abs_diff=None):
    img1 = cv2.imread(fp1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(fp2, cv2.IMREAD_UNCHANGED)

    if img1.shape != img2.shape:
        print(f"\t[IMAGE] Shape mismatch: {img1.shape} vs {img2.shape}")
        return False

    diff_pixels = np.count_nonzero(img1 != img2)
    p = diff_pixels / img1.size * 100.0
    print(f"\t[IMAGE] Deviating pixels: {diff_pixels} ({p:0.2f}%)")

    if diff_pixels == 0:
        print("\t[IMAGE] Files are identical")
        return True

    if max_abs_diff is not None:
        img1_adj = img1.copy()
        img2_adj = img2.copy()
        # Hypersim may zero pixel (0,0) after tilt-shift reprojection, so
        # ignore it when applying the max-diff tolerance.
        img1_adj[0, 0] = img2_adj[0, 0]
        diff = np.abs(
            img1_adj.astype(np.int32) - img2_adj.astype(np.int32)
        )
        if diff.max() <= max_abs_diff:
            print(
                "\t[IMAGE] Max abs diff <= "
                f"{max_abs_diff}; treating as match"
            )
            return True

    if img1.ndim >= 2 and img2.ndim >= 2:
        img1_masked = img1.copy()
        img2_masked = img2.copy()
        img1_masked[0, 0] = img2_masked[0, 0]
        diff_pixels_masked = np.count_nonzero(img1_masked != img2_masked)
        if diff_pixels_masked == 0:
            print(
                "\t[IMAGE] Only pixel (0,0) differs; "
                "expected for Hypersim UV mapping fix in v084"
            )
            return True

    return False


def _same_hypersim_instance_files(fp1, fp2):
    img1 = cv2.imread(fp1, cv2.IMREAD_UNCHANGED)
    img2 = cv2.imread(fp2, cv2.IMREAD_UNCHANGED)

    if img1 is None or img2 is None:
        print("\t[HYPERSIM INSTANCE] Unable to load images")
        return False

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # legacy Hypersim instances stored instance-id bytes as [low, high];
    # newer datasets use [high, low], so swap to normalize.
    img1 = img1[:, :, [0, 2, 1]]
    if np.array_equal(img1, img2):
        print("\t[HYPERSIM INSTANCE] Match after swapping instance bytes")
        return True

    print("\t[HYPERSIM INSTANCE] Instance mismatch after byte swap")
    return False


def _same_hypersim_extrinsics(fp1, fp2, pos_eps=1e-6, quat_eps=1e-6):
    if os.path.basename(fp1) == "0000.json":
        print(
            "\t[HYPERSIM EXTRINSICS] Skipping frame 0000; "
            "first frames may have invalid rotations (see Hypersim README)."
        )
        return False

    with open(fp1, 'r') as f:
        e1 = json.load(f)
    with open(fp2, 'r') as f:
        e2 = json.load(f)

    p1 = np.array([e1['x'], e1['y'], e1['z']], dtype=np.float64)
    p2 = np.array([e2['x'], e2['y'], e2['z']], dtype=np.float64)
    if not np.allclose(p1, p2, atol=pos_eps):
        print(f"\t[HYPERSIM EXTRINSICS] Position mismatch: {p1} vs {p2}")
        return False

    r1 = PatchedSciPyRotation.from_quat(
        [e1['quat_x'], e1['quat_y'], e1['quat_z'], e1['quat_w']]
    ).as_matrix()
    r2 = PatchedSciPyRotation.from_quat(
        [e2['quat_x'], e2['quat_y'], e2['quat_z'], e2['quat_w']]
    ).as_matrix()

    # changed in v051:
    # apply 180 degree rotation around x-axis, i.e., flipping y-axis and z-axis
    constant_rot = PatchedSciPyRotation.from_euler(
        'zyx', [0, 0, 180], degrees=True
    ).as_matrix()
    if np.allclose(
        (r1 @ constant_rot), r2, atol=quat_eps
    ):
        print("\t[HYPERSIM EXTRINSICS] Match after +180deg X rotation")
        return True

    print("\t[HYPERSIM EXTRINSICS] Rotation mismatch")
    return False


def _same_hypersim_orientations(fp1, fp2):
    with open(fp1, 'r') as f:
        ref = json.load(f)
    with open(fp2, 'r') as f:
        new = json.load(f)

    deltas = []
    for key, value in ref.items():
        if key in new:
            deltas.append(new[key] - value)

    if not deltas:
        print("\t[HYPERSIM ORIENTATIONS] No overlapping entries")
        return True

    deltas = np.array(deltas, dtype=np.float64)
    # normalize to [0, 2pi) to handle wraparound
    deltas = deltas % (2 * np.pi)
    if not np.all(np.isclose(deltas, deltas[0], atol=1e-9)):
        return False

    print(
        "\t[HYPERSIM ORIENTATIONS] "
        "Constant delta; expected after v051 +180deg extrinsics"
    )
    return True


def _load_point_cloud_ply_file(fp):
    pc_data = plyfile.PlyData.read(fp)
    points = np.empty(shape=[pc_data['vertex'].count, 3], dtype=np.float32)
    points[:, 0] = pc_data['vertex'].data['x']
    points[:, 1] = pc_data['vertex'].data['y']
    points[:, 2] = pc_data['vertex'].data['z']
    return points


def _same_ply_files(fp1, fp2, ply_eps, use_plyfile=True):
    if use_plyfile:
        points1 = _load_point_cloud_ply_file(fp1)
        points2 = _load_point_cloud_ply_file(fp2)

        pc1 = o3d.t.geometry.PointCloud()
        pc1.point['positions'] = o3d.core.Tensor(points1,
                                                 dtype=o3d.core.Dtype.Float32)
        pc2 = o3d.t.geometry.PointCloud()
        pc2.point['positions'] = o3d.core.Tensor(points2,
                                                 dtype=o3d.core.Dtype.Float32)
    else:
        # note open3d may produce a lot of warnings when reading PLY files
        pc1 = o3d.t.io.read_point_cloud(fp1)
        pc2 = o3d.t.io.read_point_cloud(fp2)

        if 'positions' not in pc1.point or 'positions' not in pc2.point:
            print("\t[PLY] Missing positions tensor")
            return False

        points1 = pc1.point.positions.numpy()
        points2 = pc2.point.positions.numpy()

    n1 = points1.shape[0]
    n2 = points2.shape[0]

    # same size -> direct comparison
    if points1.shape == points2.shape:
        dists = np.linalg.norm(points1 - points2, axis=1)
        deviating = np.count_nonzero(dists > ply_eps)
        print(f"\t[PLY] Deviating points (> eps): {deviating}")
        print(f"\t[PLY] Max deviation: {dists.max():.6g}")

        if deviating == 0:
            print("\t[PLY] Files are identical")
            return True

        return False

    # different size -> symmetric NN matching
    print(f"\t[PLY] Point count mismatch: {n1} vs {n2}")
    print(f"\t[PLY] Nearest-neighbor matching (eps={ply_eps})")

    kdtree_1 = o3d.geometry.KDTreeFlann(pc1.to_legacy())
    kdtree_2 = o3d.geometry.KDTreeFlann(pc2.to_legacy())

    matched_1 = np.zeros(n1, dtype=bool)
    matched_2 = np.zeros(n2, dtype=bool)

    # points1 -> points2
    for i in range(n1):
        _, idx, dist2_squared = kdtree_2.search_knn_vector_3d(points1[i], 1)
        if dist2_squared[0] <= ply_eps * ply_eps:
            matched_1[i] = True
            matched_2[idx[0]] = True

    # points2 -> points1
    for i in range(n2):
        _, idx, dist2_squared = kdtree_1.search_knn_vector_3d(points2[i], 1)
        if dist2_squared[0] <= ply_eps * ply_eps:
            matched_2[i] = True
            matched_1[idx[0]] = True

    matches = np.count_nonzero(matched_1)
    misses_1 = n1 - np.count_nonzero(matched_1)
    misses_2 = n2 - np.count_nonzero(matched_2)

    def as_p(x, total):
        return f"{x / total * 100:.2f}%"

    print(
        f"\t[PLY] -> Matches (<= eps): {matches} "
        f"({as_p(matches, n1)} vs {as_p(matches, n2)})\n"
        f"\t[PLY] -> Misses in pc1:    {misses_1} ({as_p(misses_1, n1)})\n"
        f"\t[PLY] -> Misses in pc2:    {misses_2} ({as_p(misses_2, n2)})"
    )

    if misses_1 == 0 and misses_2 == 0:
        print("\t[PLY] Files are identical")
        return True

    # TODO: compare labels?

    return False


def _load_embedding_file(fp):
    ext = os.path.splitext(fp)[1].lower()

    if ext == '.npz':
        # instance-level or image-level embeddings
        data = np.load(fp)

        # unify to text embedding format, i.e. a two-level dict (see below)
        return {
            '$': {
                f: data[f] for f in data.files
            }
        }

    elif ext == '.npy':
        #  semantic/scene class text embeddings given as dicts with format:
        # {'configuration': {text: embedding}, ...}
        data = np.load(fp, allow_pickle=True)
        if data.dtype == object:
            return data.item()  # extract dict
        else:
            print("\t[EMBEDDING] Unknown .npy format")

    print("\t[EMBEDDING] Unable to load embedding(s) file")
    return None


def _same_embedding(emb1, emb2, hint=None):
    hint = f"{hint or ''}: "

    # compare shapes
    if emb1.shape != emb2.shape:
        print(
            f"\t[EMBEDDING] {hint}shape mismatch: "
            f"{emb1.shape} vs {emb2.shape}"
        )
        return False

    # compare norms
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if not np.isclose(norm1, norm2, atol=1e-3):
        print(f"\t[EMBEDDING] {hint}norm mismatch: {norm1} vs {norm2}")
        return False

    # compare cosine similarity
    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    if not np.isclose(cos_sim, 1.0, atol=1e-4):
        print(f"\t[EMBEDDING] {hint}cosine similarity: 1.0 != {cos_sim}")
        return False

    return True


def _same_aux_embedding_files(fp1, fp2):
    # load embeddings
    embs1 = _load_embedding_file(fp1)
    embs2 = _load_embedding_file(fp2)
    if embs1 is None or embs2 is None:
        print("\t[EMBEDDING] Unable to load embeddings")
        return False

    # compare embeddings
    all_match = True

    # compare top-level keys
    keys1 = set(embs1.keys())
    keys2 = set(embs2.keys())
    top_level_key_mismatch = keys1 != keys2
    if top_level_key_mismatch:
        print(f"\t[EMBEDDING] Top-level key mismatch: {keys1} vs {keys2}")
        print(f"\t[EMBEDDING] Analyzing only common keys: {keys1 & keys2}")
        all_match = False

    # compare second-level keys and embeddings for common top-level keys
    common_top_keys = keys1 & keys2
    for k in common_top_keys:
        top_level_match = True
        sub_keys1 = set(embs1[k].keys())
        sub_keys2 = set(embs2[k].keys())
        if sub_keys1 != sub_keys2:
            print(
                f"\t[EMBEDDING] Second-level key mismatch for {k}: "
                f"{sub_keys1} vs {sub_keys2}"
            )
            print(f"\t[EMBEDDING] Analyzing only common sub-keys: "
                  f"{sub_keys1 & sub_keys2}")
            all_match = False
            top_level_match = False

        # compare embeddings for common sub-keys
        for sk in (sub_keys1 & sub_keys2):
            if not _same_embedding(
                embs1[k][sk].ravel(), embs2[k][sk].ravel(),
                hint=f"Embedding {k}.{sk}"
            ):
                all_match = False
                top_level_match = False

        if top_level_match and len(common_top_keys) > 1:
            print(f"\t[EMBEDDING] All embeddings match for top-level key {k}")

    if all_match:
        print("\t[EMBEDDING] All embeddings match")
        return True

    return False


def _same_aux_depth_files(fp1, fp2, max_mean_diff=1.2):
    d1 = cv2.imread(fp1, cv2.IMREAD_UNCHANGED)
    d2 = cv2.imread(fp2, cv2.IMREAD_UNCHANGED)

    if d1.shape != d2.shape:
        print(f"\t[PREDICTED DEPTH] Shape mismatch: {d1.shape} vs {d2.shape}")
        return False

    diff_pixels = np.count_nonzero(d1 != d2)
    p = diff_pixels / d1.size * 100.0
    print(f"\t[PREDICTED DEPTH] Deviating pixels: {diff_pixels} ({p:0.2f}%)")

    if diff_pixels == 0:
        # best case
        print("\t[PREDICTED DEPTH] Files are identical")
        return True

    # as the depth images are predicted, check again with relaxed tolerance
    diff = np.abs(d1.astype(np.int32) - d2.astype(np.int32))

    counts = np.unique(diff, return_counts=True)
    print("\t[PREDICTED DEPTH] Deviating value counts:")
    for value, count in zip(*counts):
        print(f"\t[PREDICTED DEPTH]\t{value:5d}: {count}")

    mean = np.mean(diff)
    max_ = np.max(diff)
    min_ = np.min(diff)
    median = np.median(diff)
    print(
        "\t[PREDICTED DEPTH] "
        f"Mean: {mean}, max: {max_}, min: {min_}, med: {median}"
    )

    if mean < max_mean_diff:
        # still ok
        print(
            "\t[PREDICTED DEPTH] Files are identical "
            f"(as mean is below max mean diff ({max_mean_diff}))"
        )
        return True

    return False


def _same_unknown(fp1, _):
    print(f"\t[NO HANDLER] No comparison handler for {fp1}")
    return False


def _deep_compare(fp1, fp2, args):
    ext = os.path.splitext(fp1)[1].lower()

    if re.search(
        r".*image_embedding_.*|.*panoptic_.*embedding_.*|.*_embeddings.npy",
        fp1
    ):
        # auxiliary data: predicted embeddings
        return _same_aux_embedding_files(fp1, fp2)
    elif re.search(r".*depth_.*__.*", fp1):
        # auxiliary data: predicted depth maps
        return _same_aux_depth_files(fp1, fp2)

    if ext in {".txt", ".xml", ".csv", ".yaml"}:
        return _same_text_files(fp1, fp2, text_diff=args.text_diff)
    if ext in {".json"}:
        is_same = _same_json_files(fp1, fp2)
        if all((
            not is_same,
            args.hypersim_legacy_equivalence,
            f"{os.sep}extrinsics{os.sep}" in fp1
        )):
            return _same_hypersim_extrinsics(fp1, fp2)
        if all((
            not is_same,
            args.hypersim_legacy_equivalence,
            f"{os.sep}orientations{os.sep}" in fp1
        )):
            return _same_hypersim_orientations(fp1, fp2)
        return is_same
    if ext in {".jpg", ".jpeg", ".png"}:
        max_abs_diff = None
        # Hypersim tonemapping is sensitive to NumPy SIMD paths, which is why
        # we allow for a relaxed comparison that treats images as identical
        # they only differ by a small per-channel RGB difference
        if args.hypersim_relax_rgb_check and f"{os.sep}rgb{os.sep}" in fp1:
            max_abs_diff = 1
        is_same = _same_image_files(fp1, fp2, max_abs_diff=max_abs_diff)
        if all((
            not is_same,
            args.hypersim_legacy_equivalence,
            f"{os.sep}instance{os.sep}" in fp1 and ext
        )):
            return _same_hypersim_instance_files(fp1, fp2)
        return is_same
    if ext in {".ply"}:
        return _same_ply_files(fp1, fp2, ply_eps=args.ply_eps)

    return _same_unknown(fp1, fp2)


def _cprint_step(*args, **kwargs):
    if can_colorize():
        cprint(*args, color='blue', attrs=('bold',))
    else:
        # this might also happen if we redirect output to a file or use tee
        print("-"*80)
        print(*args, **kwargs)
        print("-"*80)


def _summarize_missing_by_directory(rel_paths, min_fraction=0.01):
    counter = Counter()

    for rel_path in rel_paths:
        # split path into its parts and count all parent directories
        parts = rel_path.split(os.sep)
        for i in range(1, len(parts)):
            counter[os.sep.join(parts[:i])] += 1

    max_count = max(counter.values())

    # sort: bigger first, shorter paths first
    items = sorted(counter.items(), key=lambda x: (-x[1], x[0].count(os.sep)))

    # select directories to report
    selected = []
    covered = set()

    def is_topk_level(path, k=1):
        return path.count(os.sep) < k

    for path, count in items:
        # rule 1: always keep topk levels
        if is_topk_level(path, k=2):
            selected.append((path, count))
            covered.add(path)
            continue

        # rule 2: keep dominant directories
        if (count / max_count) < min_fraction:
            continue

        # rule 3: skip if already covered by a parent directory
        parent = path.rsplit(os.sep, 1)[0]
        if parent in covered:
            continue

        selected.append((path, count))
        covered.add(path)

    # order final selection by path name
    selected = sorted(selected, key=lambda x: x[0])

    return selected


def main(args=None):
    # parse args
    args = _parse_args(args=args)

    _cprint_step("Comparing dataset paths")
    print(
        f"Dataset path 1: {args.dataset_path1}\n"
        f"Dataset path 2: {args.dataset_path2}\n"
        "Options:"
    )
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # get files
    _cprint_step("Collecting files")
    print("Collecting files in dataset_path1 ...")
    filepaths1 = _collect_files(
        args.dataset_path1,
        include_regexes=args.include_regex,
        ignore_regexes=args.ignore_regex,
        disable_progress_bar=args.disable_progress_bar
    )
    print("Collecting files in dataset_path2 ...")
    filepaths2 = _collect_files(
        args.dataset_path2,
        include_regexes=args.include_regex,
        ignore_regexes=args.ignore_regex,
        disable_progress_bar=args.disable_progress_bar
    )

    set1 = set(filepaths1.keys())
    set2 = set(filepaths2.keys())

    only1 = sorted(set1 - set2)
    only2 = sorted(set2 - set1)
    common = sorted(set1 & set2)

    # report files only in one dataset
    _cprint_step("Files only in dataset_path1")
    for f in only1:
        print(f)

    _cprint_step("Files only in dataset_path2")
    for f in only2:
        print(f)

    # compare common files
    _cprint_step("Comparing common files")

    deeply_compared = 0
    deeply_mismatched = []
    filesize_mismatches = 0 if not args.checksum else -1
    checksum_mismatches = 0 if args.checksum else -1

    for fp_rel in (bar := tqdm(common, disable=args.disable_progress_bar)):
        bar.set_description(f"Comparing {fp_rel}")
        fp1 = filepaths1[fp_rel]
        fp2 = filepaths2[fp_rel]

        if not args.checksum:
            size1 = _get_file_size(fp1)
            size2 = _get_file_size(fp2)
            if size1 != size2:
                print(f"[FILESIZE MISMATCH] {fp_rel}: {size1} vs {size2}")
                filesize_mismatches += 1
            else:
                if not args.force_deep_compare:
                    continue
        else:
            c1 = _get_file_checksum(fp1)
            c2 = _get_file_checksum(fp2)
            if c1 != c2:
                print(f"[CHECKSUM MISMATCH] {fp_rel}")
                checksum_mismatches += 1
            else:
                continue

        # file sizes (or checksums) do not match -> do deep comparison
        deeply_compared += 1
        are_same = _deep_compare(fp1, fp2, args)
        if not are_same:
            deeply_mismatched.append(fp_rel)

    # summary
    _cprint_step("Summary")
    print(
        f"Total files in dataset_path1: {len(set1)}\n"
        f"Total files in dataset_path2: {len(set2)}\n"
        f"Common files:                 {len(common)}\n"
        f"Files only in dataset_path1:  {len(only1)}\n"
        f"Files only in dataset_path2:  {len(only2)}\n"
        f"Filesize mismatches:          {filesize_mismatches}\n"
        f"Checksum mismatches:          {checksum_mismatches}\n"
        f"Deeply compared files:        {deeply_compared}\n"
        f"Deeply mismatched files:      {len(deeply_mismatched)}\n"
    )

    if len(deeply_mismatched) + len(only1) + len(only2) == 0:
        cprint("\n=> Datasets match!", color='green', attrs=('bold',))
    else:
        if deeply_mismatched:
            print("Deeply mismatched files:")
            for f in deeply_mismatched:
                print(f"\t{f} (for details, see above)")

        if only1:
            print("Files only in dataset_path1 (min_fraction=0.01):")
            for d, cnt in _summarize_missing_by_directory(
                only1, min_fraction=0.01
            ):
                print(f"\t{cnt: 8d} file(s): {d}")

        if only2:
            print("Files only in dataset_path2 (min_fraction=0.01):")
            for d, cnt in _summarize_missing_by_directory(
                only2, min_fraction=0.01
            ):
                print(f"\t{cnt: 8d} file(s): {d}")

        cprint("\n=> Datasets do NOT match!", color='red', attrs=('bold',))


if __name__ == '__main__':
    main()
