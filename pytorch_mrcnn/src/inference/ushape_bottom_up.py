"""
Colon Crypt Segmentation (PyTorch) - Step 3: U-Shape Extraction ("Bottom-Up" Patches)

This script cuts each accepted/fragmented crypt out of its tile and stands it upright. It fits an
ellipse to the mask, rotates it vertical, picks the U direction from the two halves, rotates again,
flips it if the opening points down, keeps the tile pixels under the mask and crops to the crypt.

The steps are those of the TF script src/inference/ushape_bottom_up.py, without imgaug (it breaks on
NumPy 2) and with the differences of this environment removed, so TF's masks give TF's patches byte for
byte:
  * imgaug PadToAspectRatio(1.5, "center") + Resize(768x1024, cubic) reproduced exactly (pad_and_resize)
  * OpenCV 4.5's answer (1 label) for an empty image half in direc_det; OpenCV 4.13 segfaults there
  * the random 8000-point subsample of large masks is seeded per crypt from its file name (--unseeded: off)
  * after the run, every accepted/fragmented mask must have a patch, otherwise the script exits with 1

    <us_split_root>/<rel>/<category>/<tile>_<j>.png + <original_root>/<rel>/<tile>.png
        -> <output_root>/<rel>/<category>/<tile>_<j>.png


Usage (from pytorch_mrcnn/):
    python src/inference/ushape_bottom_up.py --us_split_root inference/us_split \\
        --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data --output_root inference/us_final

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import faulthandler
import glob
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import imutils
import numpy as np


# --- Runtime settings ---
# Dump a Python traceback on native segfaults (SIGSEGV/SIGABRT). Without this,
# a segfault in libopencv terminates silently — makes the bad input impossible
# to identify. With it, you get a stack trace pointing at the failing line.
faulthandler.enable()

# HPC-safe OpenCV threading. On multi-core login/compute nodes, OpenCV's
# default thread pool often races with numpy's BLAS threads and causes
# intermittent segfaults in warpAffine / fitEllipse / findContours. Pinning to
# a single thread here is ~5% slower but eliminates the class of crashes.
cv2.setNumThreads(1)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


# --- Default Settings ---
_TARGET_ASPECT = 1.5   # width / height of the model input
_TARGET_H = 768        # model input height
_TARGET_W = 1024       # model input width
MAX_FIT_POINTS = 8000  # larger masks are subsampled to this many points for the ellipse fit


def _points_are_fittable(pts, min_points=5, min_rank_gap=1.0):
    """
    Check that a point cloud can be given to cv2.fitEllipse.

    cv2.fitEllipse segfaults on fully collinear or near-collinear points even when len(pts) >= 5, so such
    inputs are rejected up front.

    Args:
        pts: points from cv2.findNonZero ([N, 1, 2]) or None
        min_points: minimum number of points
        min_rank_gap: minimum smaller singular value of the centred points

    Returns:
        True only if the points have a 2-D spread (do not lie on a line)
    """
    if pts is None or len(pts) < min_points:
        return False
    arr = pts.reshape(-1, 2).astype(np.float64)
    # Spread along the two principal axes. If the smaller singular value is
    # effectively zero, the points lie on a line -> fitEllipse can segfault.
    try:
        _, s, _ = np.linalg.svd(arr - arr.mean(axis=0), full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    return bool(s[-1] > min_rank_gap)


# ---------------------------------------------------------------------------------------------
# Pure numpy + cv2 replacement for the imgaug block of the TF script
# ---------------------------------------------------------------------------------------------
def _pad_to_aspect_imgaug(img, aspect):
    """
    Zero-pad to the width/height ratio `aspect`, as imgaug 0.4.0 PadToAspectRatio(aspect, position="center")
    (floor/ceil padding, split int(0.5 * total)).

    Args:
        img: H x W (x C) image
        aspect: target width / height

    Returns:
        The padded image
    """
    h, w = img.shape[:2]
    if w / h < aspect:
        diff = aspect * h - w
        total = int(np.ceil(diff / 2)) + int(np.floor(diff / 2))
        left = int(0.5 * total)
        return cv2.copyMakeBorder(img, 0, 0, left, total - left, cv2.BORDER_CONSTANT, value=0)
    if w / h > aspect:
        diff = w / aspect - h
        total = int(np.floor(diff / 2)) + int(np.ceil(diff / 2))
        top = int(0.5 * total)
        return cv2.copyMakeBorder(img, top, total - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
    return img


def pad_and_resize(img):
    """
    Replace iaa.PadToAspectRatio(1.5, center) + iaa.Resize(768x1024) (imgaug's default interpolation is cubic).

    Args:
        img: H x W x 3 image

    Returns:
        768 x 1024 x 3 image
    """
    img = _pad_to_aspect_imgaug(img, _TARGET_ASPECT)
    return cv2.resize(img, (_TARGET_W, _TARGET_H), interpolation=cv2.INTER_CUBIC)


def _n_components(binary):
    """
    Label count of cv2.connectedComponentsWithStats; 1 for an empty image, as in OpenCV 4.5 (4.13 segfaults).

    Args:
        binary: uint8 binary image

    Returns:
        Number of labels (background included)
    """
    if binary is None or binary.size == 0:
        return 1
    return cv2.connectedComponentsWithStats(binary)[0]


# ---------------------------------------------------------------------------------------------
# Orientation: ellipse fit on all foreground pixels, two rotations, upright check
# ---------------------------------------------------------------------------------------------
def find_ellipse(image):
    """
    Fit an ellipse to all foreground pixels of a mask.

    Args:
        image: BGR mask image

    Returns:
        Tuple of (largest external contour, ellipse ((cx, cy), (w, h), angle), grey image)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(mask)
    if not _points_are_fittable(pts):
        return np.zeros((0, 1, 2), dtype=np.int32), ([0, 0], [0, 0]), gray
    if len(pts) > MAX_FIT_POINTS:
        idx = np.random.choice(len(pts), MAX_FIT_POINTS, replace=False)
        pts = pts[idx]
    ellipse = cv2.fitEllipse(pts.astype(np.float32))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea) if contours else np.zeros((0, 1, 2), np.int32)
    return cnt, ellipse, gray


def rotate1(image, elps):
    """
    First rotation, from the ellipse angle: by 270 - angle degrees if the angle is > 90, by 90 - angle
    otherwise (none if it is 0); then fit the ellipse again on the rotated image.

    Args:
        image: BGR image (mask or tile)
        elps: ellipse of the mask (from find_ellipse)

    Returns:
        Tuple of (rotated image, rotated grey image with the new ellipse drawn, new ellipse, flag);
        flag is 1 if no rotation was done or the new fit failed
    """
    flag = 0
    angle = elps[2]
    if angle > 90:
        rotated = imutils.rotate_bound(image, 270 - angle)
    elif angle == 0:
        flag = 1
        rotated = image
    else:
        rotated = imutils.rotate_bound(image, 90 - angle)

    rotated_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thr = cv2.threshold(rotated_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pts = cv2.findNonZero(thr)
    if not _points_are_fittable(pts):
        flag = 1
        return rotated, rotated_gray, ([0, 0], [0, 0]), flag
    if len(pts) > MAX_FIT_POINTS:
        idx = np.random.choice(len(pts), MAX_FIT_POINTS, replace=False)
        pts = pts[idx]
    try:
        ellipse2 = cv2.fitEllipse(pts.astype(np.float32))
    except cv2.error:
        flag = 1
        return rotated, rotated_gray, ([0, 0], [0, 0]), flag

    cv2.ellipse(rotated_gray, ellipse2, (255, 255, 255), 10)
    return rotated, rotated_gray, ellipse2, flag


def direc_det(image):
    """
    Pick the direction of the second rotation from the two halves of the mask after the first rotation.

    The rotated mask is split at the centre of its new ellipse (or in the middle if there is none), and
    the background components of each half are counted (inverted Otsu threshold); 'left' if the left
    half has more, otherwise 'right'.

    Args:
        image: BGR mask image

    Returns:
        Tuple of (mask after the first rotation, ellipse of the mask, 'left' or 'right')
    """
    _, elps, _ = find_ellipse(image)
    rotated, rotated_gray, ellipse2, flag = rotate1(image, elps)
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

    if flag == 0 and isinstance(ellipse2, tuple):
        cx = int(ellipse2[0][0])
    else:
        cx = gray.shape[1] // 2

    left_ori = gray[:, :cx]
    right_ori = gray[:, cx:]

    _, thresh1 = cv2.threshold(left_ori, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(right_ori, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nlabels_left = _n_components(thresh1)
    nlabels_right = _n_components(thresh2)

    direction = 'left' if nlabels_left > nlabels_right else 'right'
    return rotated, elps, direction


def rotate2(img, direction):
    """
    Second rotation: by -90 degrees for 'left', by +90 degrees for 'right'.

    Args:
        img: BGR image
        direction: 'left' or 'right' (from direc_det)

    Returns:
        Rotated image
    """
    return imutils.rotate_bound(img, -90 if direction == 'left' else 90)


def fit_box(original):
    """
    Crop an image to the bounding box of its pixels that are non-zero in all channels.

    Args:
        original: H x W x 3 image

    Returns:
        Cropped image (the input if it has no such pixel)
    """
    mask = original > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return original
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return original[x0:x1, y0:y1]


def opening_is_up(mask_bgr, top_frac=0.45, bottom_frac=0.45):
    """
    Decide whether the U opening points up: the row with the fewest foreground pixels (smoothed) lies in
    the top part of the mask.

    Args:
        mask_bgr: BGR mask image
        top_frac: the opening is up if the thinnest row is in the top `top_frac` of the mask
        bottom_frac: the opening is down if the thinnest row is in the bottom `bottom_frac`

    Returns:
        True if the opening points up (or it cannot be decided)
    """
    gray = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    ys, xs = np.where(bw > 0)
    if len(ys) == 0:
        return True
    y0, y1 = ys.min(), ys.max()
    roi = bw[y0:y1 + 1, :]
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    row_counts = (roi // 255).sum(axis=1).astype(np.float32)
    k = 9 if len(row_counts) >= 9 else max(3, (len(row_counts) // 2) * 2 + 1)
    kernel = np.ones(k, dtype=np.float32) / k
    row_counts_s = np.convolve(row_counts, kernel, mode='same')
    if len(row_counts_s) == 0:
        return True
    ymin_idx = int(np.argmin(row_counts_s))
    h = len(row_counts_s)
    pos = ymin_idx / max(h - 1, 1)
    if pos <= top_frac:
        return True
    if pos >= 1.0 - bottom_frac:
        return False
    return True


def ensure_u_upright(mask_bgr, rgb_bgr):
    """
    Rotate mask and tile by 180 degrees if the U opening points down.

    Args:
        mask_bgr: BGR mask image
        rgb_bgr: BGR tile image (same frame as the mask)

    Returns:
        Tuple of (mask, tile), upright
    """
    if not opening_is_up(mask_bgr):
        mask_bgr = cv2.rotate(mask_bgr, cv2.ROTATE_180)
        rgb_bgr = cv2.rotate(rgb_bgr, cv2.ROTATE_180)
    return mask_bgr, rgb_bgr


# ---------------------------------------------------------------------------------------------
# One crypt, one slide folder, the whole tree
# ---------------------------------------------------------------------------------------------
def _process_single_mask(rgb_full, mask_path, des_dir, seeded=True):
    """
    Worker: cut the upright patch of one crypt out of its tile and save it.

    Kept as a module-level function so it is picklable for ProcessPoolExecutor. Each worker re-applies
    single-threaded OpenCV (workers inherit environment variables, but cv2.setNumThreads must be set
    inside the worker process).

    Args:
        rgb_full: tile file
        mask_path: crypt mask file (<tile>_<j>.png)
        des_dir: output folder
        seeded: seed np.random from the mask file name, so that the 8000-point subsample in
            find_ellipse/rotate1 (masks larger than 8000 pixels) is the same on every run

    Returns:
        Short status string
    """
    if seeded:
        np.random.seed(zlib.crc32(os.path.basename(mask_path).encode()))
    cv2.setNumThreads(1)
    try:
        mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask_img is None:
            return f"skip (unreadable mask) {mask_path}"
        if mask_img.size == 0 or min(mask_img.shape[:2]) < 3:
            return f"skip (tiny {mask_img.shape}) {mask_path}"

        rgb_src = cv2.imread(rgb_full, cv2.IMREAD_COLOR)
        if rgb_src is None:
            return f"skip (unreadable rgb) {rgb_full}"

        # tile in the frame of the mask (see the known issue in the header)
        mh, mw = mask_img.shape[:2]
        rgb_resize = cv2.resize(rgb_src, (mw, mh))
        rgb_img_aug = pad_and_resize(rgb_resize)

        # same rotations for mask and tile
        rotated_mask1, elps, direction = direc_det(mask_img)
        rgb_rot1, _, _, _ = rotate1(rgb_img_aug, elps)
        mask_after_r2 = rotate2(rotated_mask1, direction)
        rgb_after_r2 = rotate2(rgb_rot1, direction)

        mask_fixed, rgb_fixed = ensure_u_upright(mask_after_r2, rgb_after_r2)

        # tile pixels under the mask, cropped to the crypt
        mask_gray = cv2.cvtColor(mask_fixed, cv2.COLOR_BGR2GRAY)
        _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        res = cv2.bitwise_and(rgb_fixed, rgb_fixed, mask=mask_bin)
        res = fit_box(res)

        out_path = os.path.join(des_dir, os.path.basename(mask_path))
        ok = cv2.imwrite(out_path, res)
        return f"saved ok={ok} {out_path}"
    except Exception as e:
        return f"ERROR {mask_path}: {e!r}"


def iterate_dir_save_accepted(mask_directory, original_directory, des_dir, workers=1, seeded=True):
    """
    Extract the patches of all crypt masks in one category folder of one slide.

    Args:
        mask_directory: us_split/<rel>/<category>/ with <tile>_<j>.png masks
        original_directory: folder with the tiles of the slide
        des_dir: output folder us_final/<rel>/<category>/
        workers: parallel worker processes (1 = serial)
        seeded: per-crypt seed of the point subsample (see _process_single_mask)
    """
    os.makedirs(des_dir, exist_ok=True)
    print("[PROCESS] images:", original_directory)
    print("[PROCESS] masks :", mask_directory)
    print("[OUTPUT ] dest  :", des_dir)
    print(f"[PROCESS] workers={workers}")

    # Build the full job list: (rgb_full, mask_path, des_dir) tuples.
    jobs = []
    for rgb_fname in os.listdir(original_directory):
        base = os.path.splitext(os.path.basename(rgb_fname))[0]
        mask_names = glob.glob(os.path.join(mask_directory, base + '_*.png'))
        if not mask_names:
            continue
        rgb_full = os.path.join(original_directory, rgb_fname)
        if not os.path.isfile(rgb_full):
            print("  ! Missing RGB:", rgb_full)
            continue
        for mp in mask_names:
            if mp.lower().endswith(".png"):
                jobs.append((rgb_full, mp, des_dir, seeded))

    if not jobs:
        print("  (no jobs)")
        return

    # Serial fast path — preserves previous behavior exactly when workers<=1.
    if workers <= 1:
        for rgb_full, mp, d, s in jobs:
            print("  [process]", mp, flush=True)
            print("  ->", _process_single_mask(rgb_full, mp, d, s))
        return

    # Parallel path. Each worker pins cv2 to 1 thread; the Pool gives us
    # process-level parallelism, which is segfault-safe (unlike cv2 threads).
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_process_single_mask, r, m, d, s): m for (r, m, d, s) in jobs}
        done = 0
        total = len(futures)
        for fut in as_completed(futures):
            done += 1
            mp = futures[fut]
            try:
                status = fut.result()
            except Exception as e:
                status = f"ERROR {mp}: {e!r}"
            print(f"  [{done}/{total}] {status}", flush=True)


def process_us_split_tree(us_split_root, original_root, output_root,
                          categories=('accepted', 'fragmented'), mask_ext="*.png",
                          workers=1, seeded=True):
    """
    Walk the split tree and extract the patches of every category folder, mirroring the structure.

    Args:
        us_split_root: output root of step 2
        original_root: tile root with the same relative paths
        output_root: output root of this step
        categories: category folders to process
        mask_ext: mask file pattern (kept for compatibility; the masks are matched as <tile>_*.png)
        workers: parallel worker processes per folder
        seeded: per-crypt seed of the point subsample
    """
    us_split_root = os.path.abspath(us_split_root)
    original_root = os.path.abspath(original_root)
    output_root = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    print("[WALK] us_split_root:", us_split_root)
    hit = 0
    for dirpath, dirnames, _ in os.walk(us_split_root):
        leaf_name = os.path.basename(dirpath)
        if leaf_name not in categories:
            continue

        rel_parent = os.path.relpath(os.path.dirname(dirpath), us_split_root)
        images_dir = os.path.join(original_root, rel_parent)
        if not os.path.isdir(images_dir):
            print("  ! Missing original images dir for", dirpath, "->", images_dir)
            continue

        dest_dir = os.path.join(output_root, rel_parent, leaf_name)
        os.makedirs(dest_dir, exist_ok=True)

        iterate_dir_save_accepted(dirpath, images_dir, dest_dir, workers=workers, seeded=seeded)
        hit += 1

    if hit == 0:
        print("[WARN] No category folders found under:", us_split_root)
    else:
        print(f"\n[DONE] Processed {hit} folder(s). Output root: {output_root}")


def check_complete(us_split_root, output_root, categories):
    """
    Check that every mask in us_split/<rel>/<category>/ has a patch in us_final/<rel>/<category>/.

    Args:
        us_split_root: output root of step 2
        output_root: output root of this step
        categories: category folders that were processed

    Returns:
        True if every folder is complete
    """
    def count_png(d):
        return sum(1 for f in os.listdir(d) if f.lower().endswith(".png")) if os.path.isdir(d) else 0
    bad, total_in, total_out = [], 0, 0
    for dirpath, _, _ in os.walk(us_split_root):
        if os.path.basename(dirpath) not in categories:
            continue
        rel = os.path.relpath(dirpath, us_split_root)
        n_in, n_out = count_png(dirpath), count_png(os.path.join(output_root, rel))
        total_in += n_in
        total_out += n_out
        if n_in != n_out:
            bad.append((rel, n_in, n_out))
    print(f"[check] {total_out} of {total_in} crypt patches written")
    for rel, a, b in bad:
        print(f"[check] INCOMPLETE {rel}: {b} of {a}")
    if not bad:
        print("[check] every folder complete")
    return not bad


def main():
    parser = argparse.ArgumentParser(
        description="Extract oriented patches for accepted/fragmented masks, mirroring folder structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--us_split_root", required=True,
                        help="Root with subfolders having accepted/ and fragmented/ (from the splitter).")
    parser.add_argument("--original_root", required=True,
                        help="Root with original images mirroring the relative paths.")
    parser.add_argument("--output_root", required=True,
                        help="Destination root to save extracted patches with mirrored structure.")

    # Options
    parser.add_argument("--categories", nargs="+", default=["accepted", "fragmented"],
                        help="Which categories to process (default: accepted fragmented).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel worker processes (default 1, as verified). Each worker pins cv2 to 1 "
                             "thread; with the per-crypt seed the output does not depend on this.")
    parser.add_argument("--unseeded", action="store_true",
                        help="do not seed the per-crypt random point subsampling (the TF script's behaviour)")
    args = parser.parse_args()

    process_us_split_tree(args.us_split_root, args.original_root, args.output_root,
                          categories=tuple(args.categories),
                          workers=args.workers, seeded=not args.unseeded)
    if not check_complete(args.us_split_root, args.output_root, tuple(args.categories)):
        sys.exit(1)


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/inference/ushape_bottom_up.py --us_split_root inference/us_split \
#       --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data --output_root inference/us_final
#
#   # one protocol only, 8 worker processes (same output, the subsample is seeded per crypt)
#   python src/inference/ushape_bottom_up.py --us_split_root inference/us_split/Boiling-5min \
#       --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data/Boiling-5min \
#       --output_root inference/us_final/Boiling-5min --workers 8
