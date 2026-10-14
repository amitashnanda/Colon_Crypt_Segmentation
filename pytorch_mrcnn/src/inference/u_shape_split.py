"""
Colon Crypt Segmentation (PyTorch) - Step 2: U-Shape Split of the Crypt Masks

This script sorts every predicted crypt mask into one of three folders, from the skeleton end points,
the connected components and the large holes of the mask:
    accepted/     clean U shape
    fragmented/   U shape in pieces
    rejected/     closed O-shaped glands, noise and unclear cases

It is the classifier of the TF script src/inference/u_shape_split.py. The only difference is that
remove_small_objects is the scikit-image 0.15 version (TF environment, see below); with it every one of
the 3,852 TF crypt masks lands in the same folder as in TF (src/validation/check_split_parity.py).

Each <inf_root>/<rel>/predicted_mask/ gives <out_root>/<rel>/{accepted,fragmented,rejected}/; the masks
are copied, not moved, so the step can be re-run.

Usage (from pytorch_mrcnn/):
    python src/inference/u_shape_split.py --inf_root inference/inf --out_root inference/us_split

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os
import shutil

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label
from skimage.morphology import skeletonize


# --- Default Settings (classifier tunables) ---
MIN_OBJ_AREA       = 32          # remove tiny white specks
CLOSE_KERNEL_SIZE  = 3           # 3x3 closing to kill pinholes
HOLE_MIN_ABS       = 64          # ignore holes smaller than this (in pixels)
HOLE_MIN_FRAC      = 0.01        # ...or smaller than this *foreground* fraction
GOOD_U_ENDPTS_MIN  = 2           # endpoints in [2,4] → good U
GOOD_U_ENDPTS_MAX  = 4


def remove_small_objects(ar, min_size=64, connectivity=1, in_place=False):
    """
    scikit-image 0.15 remove_small_objects for boolean masks: removes components *smaller than* min_size.

    Newer scikit-image (0.26 in torch_env) also removes components *equal to* min_size, which empties a
    32-pixel crypt mask (MIN_OBJ_AREA = 32) that TF keeps and classifies.

    Args:
        ar: bool mask
        min_size: smallest component size that is kept
        connectivity: neighbourhood connectivity (1 = 4-connected in 2-D)
        in_place: modify `ar` instead of a copy

    Returns:
        The cleaned bool mask
    """
    out = ar if in_place else ar.copy()
    if min_size == 0:
        return out
    selem = ndi.generate_binary_structure(ar.ndim, connectivity)
    ccs = np.zeros_like(ar, dtype=np.int32)
    ndi.label(ar, selem, output=ccs)
    too_small = np.bincount(ccs.ravel()) < min_size
    out[too_small[ccs]] = 0
    return out


def read_mask_binary(path):
    """
    Read a mask as a bool foreground, lightly cleaned for classification.

    Args:
        path: mask PNG

    Returns:
        H x W bool mask, or None if the file cannot be read
    """
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    # Otsu binarization
    _, binv = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binv = (binv > 0)

    # Remove very small white specks
    if binv.any():
        binv = remove_small_objects(binv, min_size=MIN_OBJ_AREA)

    # Light morphological closing to eliminate pinholes (won't bridge big gaps)
    ksz = CLOSE_KERNEL_SIZE
    if ksz and ksz > 1:
        ker = np.ones((ksz, ksz), np.uint8)
        binv = cv2.morphologyEx(binv.astype(np.uint8)*255, cv2.MORPH_CLOSE, ker) > 0

    return binv


def count_endpoints(skel_bool):
    """
    Count the end points of an 8-connected skeleton (pixels with exactly one neighbour).

    Args:
        skel_bool: bool skeleton

    Returns:
        Number of end points
    """
    if skel_bool is None or skel_bool.sum() == 0:
        return 0
    skel_u8 = skel_bool.astype(np.uint8)
    kernel = np.array([[1,1,1],
                       [1,0,1],
                       [1,1,1]], dtype=np.uint8)
    nbrs = cv2.filter2D(skel_u8, ddepth=cv2.CV_16S, kernel=kernel, borderType=cv2.BORDER_CONSTANT)
    return int(np.sum((skel_u8 == 1) & (nbrs == 1)))


def count_large_holes(binary_mask):
    """
    Count the holes of a mask, ignoring tiny ones.

    The inverted mask is labelled; every component that does not touch the border is a hole. Only holes
    with area >= max(HOLE_MIN_ABS, HOLE_MIN_FRAC * foreground area) count as large.

    Args:
        binary_mask: bool mask

    Returns:
        Tuple of (number of large holes, number of holes)
    """
    if binary_mask is None:
        return 0, 0
    fg_area = int(binary_mask.sum())
    if fg_area == 0:
        return 0, 0

    min_hole_area = max(HOLE_MIN_ABS, int(HOLE_MIN_FRAC * fg_area))

    inv = ~binary_mask
    lbl = label(inv, connectivity=2)  # 0 = background
    nlabels = int(lbl.max())
    if nlabels == 0:
        return 0, 0

    # labels that touch the border (not holes)
    border_labels = np.unique(np.concatenate([lbl[0,:], lbl[-1,:], lbl[:,0], lbl[:,-1]]))
    hole_labels = [i for i in range(1, nlabels+1) if i not in border_labels]

    total_holes = len(hole_labels)
    if total_holes == 0:
        return 0, 0

    # count only "large" holes
    large = 0
    for i in hole_labels:
        area = int((lbl == i).sum())
        if area >= min_hole_area:
            large += 1
    return large, total_holes


def classify_mask(binary_mask):
    """
    Classify one crypt mask.

    Rules, checked in this order:
        o_shape (loop):  single component AND >= 1 large hole AND end points <= 1
        fragmented_u:    no large holes AND (components >= 2 OR end points >= 3)
        good_u:          no large holes AND single component AND end points in [2, 4]
        rejected:        anything else (empty, very noisy, odd border cases)

    Args:
        binary_mask: bool mask (from read_mask_binary)

    Returns:
        Tuple of (class name, dict with the measured features)
    """
    if binary_mask is None or binary_mask.sum() == 0:
        return "rejected", {"reason": "empty"}

    num_comp = int(label(binary_mask, connectivity=2).max())
    large_holes, total_holes = count_large_holes(binary_mask)

    skel = skeletonize(binary_mask)
    endpoints = count_endpoints(skel)

    # Strict O-rule (AND)
    if (num_comp == 1) and (large_holes >= 1) and (endpoints <= 1):
        return "o_shape", {"components": num_comp, "large_holes": large_holes,
                           "total_holes": total_holes, "endpoints": endpoints}

    # Fragmented U (no large holes but multiple comps or many endpoints)
    if (large_holes == 0) and ((num_comp >= 2) or (endpoints >= 3)):
        return "fragmented_u", {"components": num_comp, "large_holes": large_holes,
                                "total_holes": total_holes, "endpoints": endpoints}

    # Good U (no large holes, one component, right number of endpoints)
    if (large_holes == 0) and (num_comp >= 1) and (GOOD_U_ENDPTS_MIN <= endpoints <= GOOD_U_ENDPTS_MAX):
        return "good_u", {"components": num_comp, "large_holes": large_holes,
                          "total_holes": total_holes, "endpoints": endpoints}

    # Anything else (very noisy, odd border cases)
    return "rejected", {"components": num_comp, "large_holes": large_holes,
                        "total_holes": total_holes, "endpoints": endpoints, "reason": "uncertain"}


def split_directory_single(mask_dir, out_dir, pattern="*.png"):
    """
    Split one predicted_mask/ folder into accepted/, fragmented/ and rejected/.

    Each mask is COPIED (not moved), so it is safe to re-run.

    Args:
        mask_dir: folder with the masks of one slide
        out_dir: folder that gets accepted/, fragmented/ and rejected/
        pattern: mask file pattern
    """
    acc_dir = os.path.join(out_dir, "accepted")
    frag_dir = os.path.join(out_dir, "fragmented")
    rej_dir  = os.path.join(out_dir, "rejected")
    os.makedirs(acc_dir, exist_ok=True)
    os.makedirs(frag_dir, exist_ok=True)
    os.makedirs(rej_dir,  exist_ok=True)

    paths = sorted(glob.glob(os.path.join(mask_dir, pattern)))
    print(f"[INFO] Found {len(paths)} mask(s) in: {mask_dir}")

    n_good = n_frag = n_rej = 0
    for p in paths:
        binv = read_mask_binary(p)
        cls, meta = classify_mask(binv)
        base = os.path.basename(p)

        if cls == "good_u":
            dst = os.path.join(acc_dir, base); n_good += 1
        elif cls == "fragmented_u":
            dst = os.path.join(frag_dir, base); n_frag += 1
        else:  # 'o_shape' or 'rejected'
            dst = os.path.join(rej_dir, base);  n_rej  += 1

        shutil.copy2(p, dst)
        print(f"  • {base:>30s} -> {cls:12s}  {meta}")

    print("\n[SUMMARY]")
    print(f"  accepted (good U):   {n_good}")
    print(f"  fragmented U:        {n_frag}")
    print(f"  rejected (O/other):  {n_rej}")
    print(f"  Output root: {out_dir}")


def split_tree(inf_root, out_root, mask_subdir="predicted_mask", pattern="*.png"):
    """
    Mirror the folder structure of `inf_root` and split every `mask_subdir` folder found in it.

    For every folder that contains a `mask_subdir` subfolder with masks, the split is written to
    out_root/<path of that folder relative to inf_root>/{accepted,fragmented,rejected}/.

    Args:
        inf_root: inference output root (step 1)
        out_root: output root of the split
        mask_subdir: name of the mask folders
        pattern: mask file pattern
    """
    inf_root = os.path.abspath(inf_root)
    out_root = os.path.abspath(out_root)
    os.makedirs(out_root, exist_ok=True)

    print(f"[WALK] Scanning: {inf_root}")
    hits = 0
    for dirpath, dirnames, _ in os.walk(inf_root):
        if mask_subdir in dirnames:
            mask_dir = os.path.join(dirpath, mask_subdir)
            # only proceed if there are masks
            if not glob.glob(os.path.join(mask_dir, pattern)):
                continue
            rel_dir = os.path.relpath(dirpath, inf_root)  # parent of 'predicted_mask'
            out_dir = os.path.join(out_root, rel_dir)
            os.makedirs(out_dir, exist_ok=True)
            print(f"\n[PROCESS] {mask_dir}")
            print(f"[OUTPUT ] {out_dir}")
            split_directory_single(mask_dir, out_dir, pattern=pattern)
            hits += 1

    if hits == 0:
        print("[WARN] No '{}' folders with masks found under: {}".format(mask_subdir, inf_root))
    else:
        print(f"\n[DONE] Processed {hits} mask folder(s). Output root: {out_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Split predicted masks into accepted/fragmented/rejected, mirroring folder structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--inf_root", required=True,
                        help="Root directory containing inference results (folders with a 'predicted_mask/' subfolder).")
    parser.add_argument("--out_root", required=True,
                        help="Output root where the mirrored split folders will be created.")
    args = parser.parse_args()

    split_tree(args.inf_root, args.out_root, mask_subdir="predicted_mask", pattern="*.png")


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/inference/u_shape_split.py --inf_root inference/inf --out_root inference/us_split
#   -> inference/us_split/<protocol>/<slide>/{accepted,fragmented,rejected}/<tile>_<j>.png
