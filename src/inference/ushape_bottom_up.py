"""
Colon Crypt Segmentation - Step 3: U-Shape Extraction ("Bottom-Up" Patches)

This script cuts each accepted/fragmented crypt out of its tile and stands it upright. It fits an
ellipse to the mask, rotates it, picks the direction of the second rotation from the two halves,
rotates again, flips the patch if the U opening points down, keeps the tile pixels under the mask and
crops to the crypt.

    <us_split_root>/<rel>/<category>/<tile>_<j>.png + <original_root>/<rel>/<tile>.png
        -> <output_root>/<rel>/<category>/<tile>_<j>.png

The ellipse fit of masks larger than 8000 pixels uses a random 8000-point subsample (np.random,
unseeded, as in the paper run), so a few large crypts can give slightly different patches from run to
run. pytorch_mrcnn/src/inference/ushape_bottom_up.py seeds it per crypt.

Known issue (kept so that the results match the paper): the tile is resized to the mask size before the
imgaug pad + resize, so for tiles that are not 4:3 the patch frame is shifted against the mask.

Usage (TF environment, from the repository root):
    python src/inference/ushape_bottom_up.py --us_split_root inference/us_split \\
        --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data --output_root inference/us_final

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os

import cv2
import numpy as np
import imutils
import imgaug.augmenters as iaa


# --- Default Settings ---
MAX_FIT_POINTS = 8000                    # larger masks are subsampled to this many points for the ellipse fit
INPUT_ASPECT = 1.5                       # the tile gets the model's input frame: padded to width / height 1.5 ...
INPUT_HEIGHT, INPUT_WIDTH = 768, 1024    # ... and resized to 1024 x 768


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
    if pts is None or len(pts) < 5:
        return np.zeros((0,1,2), dtype=np.int32), ([0,0],[0,0]), gray
    if len(pts) > MAX_FIT_POINTS:
        idx = np.random.choice(len(pts), MAX_FIT_POINTS, replace=False)
        pts = pts[idx]
    ellipse = cv2.fitEllipse(pts.astype(np.float32))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(contours, key=cv2.contourArea) if contours else np.zeros((0,1,2), np.int32)
    return cnt, ellipse, gray


def rotate1(image, elps):
    """
    First rotation, from the ellipse angle: by 270 - angle degrees if the angle is > 90, by 90 - angle
    otherwise (none if it is 0); then fit the ellipse again on all foreground pixels of the rotated image.

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
    if pts is None or len(pts) < 5:
        flag = 1
        return rotated, rotated_gray, ([0,0],[0,0]), flag
    if len(pts) > MAX_FIT_POINTS:
        idx = np.random.choice(len(pts), MAX_FIT_POINTS, replace=False)
        pts = pts[idx]
    try:
        ellipse2 = cv2.fitEllipse(pts.astype(np.float32))
    except:
        flag = 1
        return rotated, rotated_gray, ([0,0],[0,0]), flag

    cv2.ellipse(rotated_gray, ellipse2, (255,255,255), 10)  # optional overlay
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

    left_ori  = gray[:, :cx]
    right_ori = gray[:,  cx:]

    _, thresh1 = cv2.threshold(left_ori,  0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh2 = cv2.threshold(right_ori, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    nlabels_left,  *_ = cv2.connectedComponentsWithStats(thresh1)
    nlabels_right, *_ = cv2.connectedComponentsWithStats(thresh2)

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
    roi = bw[y0:y1+1, :]
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    row_counts = (roi // 255).sum(axis=1).astype(np.float32)
    k = 9 if len(row_counts) >= 9 else max(3, (len(row_counts)//2)*2 + 1)
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
        rgb_bgr  = cv2.rotate(rgb_bgr,  cv2.ROTATE_180)
    return mask_bgr, rgb_bgr


def pad_and_resize_imgaug(img):
    """
    Give an image the model's input frame: imgaug PadToAspectRatio(1.5, "center") + Resize(768 x 1024).

    Args:
        img: H x W x 3 image

    Returns:
        768 x 1024 x 3 image
    """
    augmentation = iaa.Sequential([
        iaa.PadToAspectRatio(INPUT_ASPECT, position="center"),
        iaa.Resize({"height": INPUT_HEIGHT, "width": INPUT_WIDTH}),
    ], random_order=False)
    return augmentation.augment_image(img)


# ---------------------------------------------------------------------------------------------
# One category folder of one slide, and the whole tree
# ---------------------------------------------------------------------------------------------
def iterate_dir_save_accepted(mask_directory, original_directory, des_dir):
    """
    Extract the upright patches of all crypt masks in one category folder of one slide.

    Args:
        mask_directory: us_split/<rel>/<category>/ with <tile>_<j>.png masks
        original_directory: folder with the tiles of the slide
        des_dir: output folder us_final/<rel>/<category>/
    """
    os.makedirs(des_dir, exist_ok=True)
    print("[PROCESS] images:", original_directory)
    print("[PROCESS] masks :", mask_directory)
    print("[OUTPUT ] dest  :", des_dir)

    for rgb_fname in os.listdir(original_directory):
        base = os.path.splitext(os.path.basename(rgb_fname))[0]
        mask_names = glob.glob(os.path.join(mask_directory, base + '_*.png'))
        if not mask_names:
            continue

        rgb_full = os.path.join(original_directory, rgb_fname)
        rgb_src = cv2.imread(rgb_full, cv2.IMREAD_COLOR)
        if rgb_src is None:
            print("  ! Could not read RGB:", rgb_full)
            continue

        for mask_path in mask_names:
            if not mask_path.lower().endswith(".png"):
                continue
            try:
                mask_img = cv2.imread(mask_path, cv2.IMREAD_COLOR)
                if mask_img is None:
                    print("  ! Could not read mask:", mask_path)
                    continue

                # tile in the frame of the mask (see the known issue in the header)
                mh, mw = mask_img.shape[:2]
                rgb_resize = cv2.resize(rgb_src, (mw, mh))
                rgb_img_aug = pad_and_resize_imgaug(rgb_resize)

                # orientation: same rotations for mask and tile
                rotated_mask1, elps, direction = direc_det(mask_img)
                rgb_rot1, _, _, _ = rotate1(rgb_img_aug, elps)
                mask_after_r2 = rotate2(rotated_mask1, direction)
                rgb_after_r2  = rotate2(rgb_rot1,     direction)

                # upright fix
                mask_fixed, rgb_fixed = ensure_u_upright(mask_after_r2, rgb_after_r2)

                # extract patch: tile pixels under the mask, cropped to the crypt
                mask_gray = cv2.cvtColor(mask_fixed, cv2.COLOR_BGR2GRAY)
                _, mask_bin = cv2.threshold(mask_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                res = cv2.bitwise_and(rgb_fixed, rgb_fixed, mask=mask_bin)
                res = fit_box(res)

                out_path = os.path.join(des_dir, os.path.basename(mask_path))
                ok = cv2.imwrite(out_path, res)
                print("  -> saved:", out_path, "ok:", ok)

            except Exception as e:
                print("  ! Error on", mask_path, ":", repr(e))
                continue


def process_us_split_tree(us_split_root, original_root, output_root,
                          categories=('accepted','fragmented'), mask_ext="*.png"):
    """
    Walk the split tree and extract the patches of every category folder, mirroring the structure.

    Args:
        us_split_root: root containing .../<rel_path>/(accepted|fragmented|rejected)/ (step 2)
        original_root: root containing the tiles in .../<rel_path>/
        output_root: where to save the patches, mirroring .../<rel_path>/(accepted|fragmented)/
        categories: category folders to process
        mask_ext: mask file pattern (kept for compatibility; the masks are matched as <tile>_*.png)
    """
    us_split_root = os.path.abspath(us_split_root)
    original_root = os.path.abspath(original_root)
    output_root   = os.path.abspath(output_root)
    os.makedirs(output_root, exist_ok=True)

    print("[WALK] us_split_root:", us_split_root)
    hit = 0
    for dirpath, dirnames, _ in os.walk(us_split_root):
        leaf_name = os.path.basename(dirpath)
        if leaf_name not in categories:
            continue  # we only process accepted/fragmented folders themselves

        # relative path up to the parent of the category folder
        rel_parent = os.path.relpath(os.path.dirname(dirpath), us_split_root)
        # the corresponding original images folder
        images_dir = os.path.join(original_root, rel_parent)
        if not os.path.isdir(images_dir):
            print("  ! Missing original images dir for", dirpath, "->", images_dir)
            continue

        # destination mirrors the same relative path and category
        dest_dir = os.path.join(output_root, rel_parent, leaf_name)
        os.makedirs(dest_dir, exist_ok=True)

        iterate_dir_save_accepted(dirpath, images_dir, dest_dir)
        hit += 1

    if hit == 0:
        print("[WARN] No category folders found under:", us_split_root)
    else:
        print(f"\n[DONE] Processed {hit} folder(s). Output root: {output_root}")


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
    parser.add_argument("--categories", nargs="+", default=["accepted","fragmented"],
                        help="Which categories to process (default: accepted fragmented).")
    args = parser.parse_args()

    process_us_split_tree(args.us_split_root, args.original_root, args.output_root,
                          categories=tuple(args.categories))


if __name__ == "__main__":
    main()


# Example usage (from the repository root, TF environment):
#
#   python src/inference/ushape_bottom_up.py --us_split_root inference/us_split \
#       --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data --output_root inference/us_final
#
#   # one protocol only (the one-command runner starts one such process per protocol)
#   python src/inference/ushape_bottom_up.py --us_split_root inference/us_split/Boiling-5min \
#       --original_root dataset/Whole_Slides_Segments_Processed/CDX2_Data/Boiling-5min \
#       --output_root inference/us_final/Boiling-5min
