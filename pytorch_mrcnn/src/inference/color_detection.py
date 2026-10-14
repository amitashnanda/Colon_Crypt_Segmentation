"""
Colon Crypt Segmentation (PyTorch) - Step 4: Colour Detection Along Each Crypt

This script measures the colour profile along every extracted crypt patch and writes one summary row
per slide. It is the same code as the TF script src/inference/color_detection.py (same steps and
parameters).

For each <src>/<rel>/ holding accepted/ and/or fragmented/ patches, every patch is
  1. resized to 400 x 1200 (width x height),
  2. reduced to its nuclei (Otsu threshold on the non-zero pixels, small opening + closing),
  3. split into blue (hue 80-140) and brown (hue 0-40 or 150-180) nuclei pixels,
  4. turned into per-row blue and brown counts, smoothed and normalised (blue + brown = 1 per row).
The normalised profiles of all patches of the slide are summed.

Outputs in <dest>/<rel>/:
    mask_and/<patch>.png          nuclei-only patch
    blue_brown_mask/<patch>.pdf   patch, nuclei, blue, brown and its colour profile
    all_in_one.pdf                summed (smoothed) profile of the slide
    final.txt                     number of crypts and the summed blue/brown profile means over the
                                  top/bottom 50% and 25% of the height (one row per slide)

Usage (from pytorch_mrcnn/):
    python src/inference/color_detection.py --src inference/us_final --dest inference/final

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d  # scipy.ndimage.filters is deprecated in current SciPy


# --- Default Settings ---
TARGET_SIZE = (400, 1200)                  # (width, height) of every patch; profiles run along the height
CATEGORY_DIRS = ('accepted', 'fragmented', 'fragments')   # patch folders read in each slide folder
MORPH_KERNEL = np.ones((2, 2), np.uint8)   # opening + closing of the nuclei mask
# HSV ranges (OpenCV hue 0-180) as (h_low, h_high, s_low, s_high, v_low, v_high)
BROWN_HSV_RANGES = ((0, 40, 1, 254, 1, 254), (150, 180, 1, 254, 1, 254))   # DAB-brown nuclei
BLUE_HSV_RANGE = (80, 140, 1, 254, 1, 254)                                 # haematoxylin-blue nuclei
PATCH_SIGMA = 15     # Gaussian smoothing of the per-row counts of one patch
SLIDE_SIGMA = 10     # Gaussian smoothing of the summed profile (all_in_one.pdf only)
EPS = 1e-6           # avoids 0/0 in rows without blue or brown pixels
FINAL_HEADERS = [
    'directory_name', 'number_of_images',
    'avg_blue_top_50%',  'avg_blue_bottom_50%',
    'avg_blue_top_25%',  'avg_blue_bottom_25%',
    'avg_brown_top_50%', 'avg_brown_bottom_50%',
    'avg_brown_top_25%', 'avg_brown_bottom_25%',
]


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------
def show_image_plt(im1, im2, im3, im4, blue_count, brown_count, labels, save_path):
    """
    Save the per-patch figure: four images and the blue/brown profile along the height.

    Args:
        im1, im2: RGB images (patch, nuclei only)
        im3, im4: grey images (blue mask, brown mask)
        blue_count, brown_count: per-row profiles
        labels: five panel titles
        save_path: PDF file to write
    """
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=1, ncols=5, figsize=(12, 3))
    ax1.imshow(im1); ax1.set_title(labels[0]); ax1.axis('off')
    ax2.imshow(im2); ax2.set_title(labels[1]); ax2.axis('off')
    ax3.imshow(im3, cmap='gray'); ax3.set_title(labels[2]); ax3.axis('off')
    ax4.imshow(im4, cmap='gray'); ax4.set_title(labels[3]); ax4.axis('off')
    ax5.plot(blue_count[::-1],  range(len(blue_count[::-1])),  label='blue')
    ax5.plot(brown_count[::-1], range(len(brown_count[::-1])), label='brown')
    ax5.set_title(labels[4]); ax5.legend(); ax5.axis('off')
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, bbox_inches='tight')
    plt.close(fig)


def mask_builder(image_bgr, hl, hh, sl, sh, vl, vh):
    """
    Mask of the pixels inside an HSV box.

    Args:
        image_bgr: BGR image
        hl, hh: hue range (OpenCV 0-180)
        sl, sh: saturation range
        vl, vh: value range

    Returns:
        uint8 mask, 255 inside the range
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    lower_bound = np.array([hl, sl, vl], dtype=np.uint8)
    upper_bound = np.array([hh, sh, vh], dtype=np.uint8)
    return cv2.inRange(hsv, lower_bound, upper_bound)


def halves_quarters(arr):
    """
    Means of a profile over its top/bottom half and top/bottom quarter.

    Args:
        arr: 1-D profile (index 0 = top of the crypt)

    Returns:
        Tuple of (top 50%, bottom 50%, top 25%, bottom 25%) means
    """
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(np.mean(arr[: n//2])),
        float(np.mean(arr[n//2 :])),
        float(np.mean(arr[: n//4])),
        float(np.mean(arr[(3*n)//4 :]))
    )


# ---------------------------------------------------------------------------------------------
# One slide folder
# ---------------------------------------------------------------------------------------------
def process_accepted_fragmented(parent_dir, dest_dir, target_size=TARGET_SIZE):
    """
    Colour profiles of all patches of one slide folder, its plots and its final.txt row.

    Args:
        parent_dir: folder that contains 'accepted/' and/or 'fragmented/' (or 'fragments/')
        dest_dir: mirror of parent_dir under the output root
        target_size: (width, height) every patch is resized to
    """
    # which leaf folders to include
    cat_dirs = []
    for name in CATEGORY_DIRS:
        p = os.path.join(parent_dir, name)
        if os.path.isdir(p):
            cat_dirs.append(p)
    if not cat_dirs:
        return  # nothing to do here

    os.makedirs(dest_dir, exist_ok=True)
    mask_and_dir = os.path.join(dest_dir, "mask_and")
    bbm_dir      = os.path.join(dest_dir, "blue_brown_mask")
    os.makedirs(mask_and_dir, exist_ok=True)
    os.makedirs(bbm_dir, exist_ok=True)

    # OpenCV resize expects (width, height); profiles are vertical (height)
    W, H = target_size               # width, height
    all_blue  = np.zeros(H, dtype=np.float64)
    all_brown = np.zeros(H, dtype=np.float64)
    total_images = 0

    for cat in cat_dirs:
        im_names = sorted(glob.glob(os.path.join(cat, "*.png")))
        for im_name in im_names:
            img = cv2.imread(im_name, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue

            # resize to portrait: width=W, height=H
            img = cv2.resize(img, (W, H))
            img_original = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if np.all(gray == 0):
                continue

            # nuclei-ish mask (Otsu on non-zero)
            nz = gray[gray != 0]
            if nz.size == 0:
                continue
            thr, _ = cv2.threshold(nz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            nuclei_mask = np.logical_and(0 < gray, gray <= thr).astype(np.uint8)

            # small clean
            opening = cv2.morphologyEx(nuclei_mask, cv2.MORPH_OPEN, MORPH_KERNEL)
            closing = cv2.morphologyEx(opening,     cv2.MORPH_CLOSE, MORPH_KERNEL)

            # apply mask
            img_nuclei_only = cv2.bitwise_and(img, img, mask=closing)
            cv2.imwrite(os.path.join(mask_and_dir, os.path.basename(im_name)), img_nuclei_only)
            img_rgb_nuclei_only = cv2.cvtColor(img_nuclei_only, cv2.COLOR_BGR2RGB)

            # HSV color masks within nuclei
            brown_mask = np.logical_or(
                mask_builder(img_nuclei_only, *BROWN_HSV_RANGES[0]) > 0,
                mask_builder(img_nuclei_only, *BROWN_HSV_RANGES[1]) > 0
            ).astype(np.uint8) * 255
            blue_mask  = mask_builder(img_nuclei_only, *BLUE_HSV_RANGE)

            # per-row counts (length = H)
            brown_count = np.sum(brown_mask == 255, axis=1).astype(np.float64)
            blue_count  = np.sum(blue_mask  == 255, axis=1).astype(np.float64)

            # smooth
            brown_s = gaussian_filter1d(brown_count, sigma=PATCH_SIGMA)
            blue_s  = gaussian_filter1d(blue_count,  sigma=PATCH_SIGMA)

            # row-wise normalize for profile shape
            total = blue_s + brown_s + EPS
            blue_norm  = blue_s  / total
            brown_norm = brown_s / total

            # per-image figure
            plot_path = os.path.join(bbm_dir, os.path.splitext(os.path.basename(im_name))[0] + '.pdf')
            show_image_plt(
                img_original, img_rgb_nuclei_only, blue_mask, brown_mask,
                blue_norm, brown_norm,
                labels=["original", "nuclei_mask", "blue", "brown", "color pattern"],
                save_path=plot_path
            )

            # accumulate SUM profiles (vertical)
            all_blue  += blue_norm
            all_brown += brown_norm
            total_images += 1

    # --------- per-folder aggregated plot ---------
    if total_images > 0:
        all_brown_smooth = gaussian_filter1d(all_brown, sigma=SLIDE_SIGMA)
        all_blue_smooth  = gaussian_filter1d(all_blue,  sigma=SLIDE_SIGMA)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(all_blue_smooth[::-1],  range(len(all_blue_smooth[::-1])),  label='blue')
        ax.plot(all_brown_smooth[::-1], range(len(all_brown_smooth[::-1])), label='brown')
        ax.legend()
        ax.set_xlabel('Summed Color Intensity (normalized per image, then summed)')
        ax.set_ylabel('Image Height (Top to Bottom)')
        ax.set_title(f'Aggregated Color Profile\n{os.path.basename(parent_dir)}')
        ax.grid(True)
        fig.savefig(os.path.join(dest_dir, 'all_in_one.pdf'), bbox_inches='tight')
        plt.close(fig)

    # --------- per-folder final.txt ---------
    base_name  = os.path.basename(parent_dir.rstrip(os.sep))
    final_path = os.path.join(dest_dir, 'final.txt')
    with open(final_path, 'w') as f:
        f.write('\t'.join(FINAL_HEADERS) + '\n')

        b_top50,  b_bot50,  b_top25,  b_bot25  = halves_quarters(all_blue)
        br_top50, br_bot50, br_top25, br_bot25 = halves_quarters(all_brown)

        row = [
            base_name, str(total_images),
            str(b_top50),  str(b_bot50),  str(b_top25),  str(b_bot25),
            str(br_top50), str(br_bot50), str(br_top25), str(br_bot25),
        ]
        f.write('\t'.join(row) + '\n')

    print(f"[DONE] {parent_dir} -> {dest_dir}  (images: {total_images})")


# ---------------------------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------------------------
def walk_and_process(src_root, dest_root):
    """
    Process every slide folder under src_root, mirroring the structure under dest_root.

    A slide folder is any folder that contains 'accepted/' and/or 'fragmented/' (or 'fragments/').

    Args:
        src_root: output root of step 3 (U-shape patches)
        dest_root: output root of this step
    """
    src_root  = os.path.abspath(src_root)
    dest_root = os.path.abspath(dest_root)
    os.makedirs(dest_root, exist_ok=True)

    hits = 0
    for dirpath, dirnames, filenames in os.walk(src_root):
        has_acc = 'accepted'   in dirnames
        has_fra = 'fragmented' in dirnames or 'fragments' in dirnames
        if not (has_acc or has_fra):
            continue

        rel = os.path.relpath(dirpath, src_root)
        out_dir = os.path.join(dest_root, rel)
        os.makedirs(out_dir, exist_ok=True)

        process_accepted_fragmented(dirpath, out_dir)
        hits += 1

    if hits == 0:
        print(f"[WARN] No folders with 'accepted/' or 'fragmented/' under: {src_root}")
    else:
        print(f"[SUMMARY] Wrote artifacts for {hits} folder(s) under: {dest_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Colour profile (blue/brown nuclei) along every crypt patch and one final.txt row per slide.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", required=True, help="Root containing subfolders with accepted/ and/or fragmented/")
    parser.add_argument("--dest", required=True, help="Output root to mirror structure and write artifacts + final.txt")
    args = parser.parse_args()

    walk_and_process(args.src, args.dest)


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/inference/color_detection.py --src inference/us_final --dest inference/final
#   -> inference/final/<protocol>/<slide>/final.txt (+ mask_and/, blue_brown_mask/, all_in_one.pdf)
#
#   # one protocol only (the one-command runner starts one such process per protocol)
#   python src/inference/color_detection.py --src inference/us_final/Boiling-5min --dest inference/final/Boiling-5min
