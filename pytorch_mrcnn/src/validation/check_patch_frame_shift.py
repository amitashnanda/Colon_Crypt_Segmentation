"""
Colon Crypt Segmentation (PyTorch) - Validation: Patch-Frame Shift of the U-Shape Step

This script measures the patch-frame issue shared by the TF and the PyTorch U-shape step. Inference
predicts the masks on the tile padded to W/H = 1.5 and resized to 1024 x 768 (model frame).
ushape_bottom_up.py (TF and PyTorch) first resizes the tile to 1024 x 768, then pads and resizes again
(patch frame). The two frames agree only for 4:3 tiles. For every crypt mask, the script computes how
far the crypt sits from its mask in the patch frame (pixels of the 1024 x 768 frame), and prints the
tile aspect ratios and the distribution of the shift.

Usage (from pytorch_mrcnn/):
    python src/validation/check_patch_frame_shift.py --root ../inference

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os
import re
import struct

import cv2
import numpy as np


# --- Default Settings ---
PROTOCOLS = ["Boiling-2min", "Boiling-5min", "Boiling-15min", "Pressure-3min"]
CDX2_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "dataset", "Whole_Slides_Segments_Processed", "CDX2_Data")


def png_size(path):
    """
    Read the size of a PNG from its header (no decoding).

    Args:
        path: PNG file

    Returns:
        Tuple of (width, height)
    """
    with open(path, "rb") as fh:
        w, h = struct.unpack(">II", fh.read(24)[16:24])
    return w, h


def imgaug_pad(w, h, aspect=1.5):
    """
    Padding of imgaug PadToAspectRatio(aspect, 'center') for a w x h image.

    Args:
        w, h: image size
        aspect: target width / height

    Returns:
        Tuple of (left pad, top pad, padded width, padded height)
    """
    if w / h < aspect:
        d = aspect * h - w
        tot = int(np.ceil(d / 2)) + int(np.floor(d / 2))
        return int(0.5 * tot), 0, w + tot, h
    if w / h > aspect:
        d = w / aspect - h
        tot = int(np.floor(d / 2)) + int(np.ceil(d / 2))
        return 0, int(0.5 * tot), w, h + tot
    return 0, 0, w, h


def main():
    parser = argparse.ArgumentParser(
        description="Shift between each crypt mask and its tissue in the patch frame of the U-shape step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="results root with inf/<protocol>/<slide>/predicted_mask")
    parser.add_argument("--images", default=CDX2_DATA, help="CDX2_Data root (default: pytorch_mrcnn/dataset/.../CDX2_Data)")
    args = parser.parse_args()

    img_root = args.images
    l2, t2, wp2, hp2 = imgaug_pad(1024, 768)
    aspects, shifts = [], []
    for p in PROTOCOLS:
        for s in sorted(os.listdir(os.path.join(img_root, p))):
            if s == "Mouse-DSS" or not os.path.isdir(os.path.join(img_root, p, s)):
                continue
            # tile sizes of the slide
            sizes = {}
            for f in os.listdir(os.path.join(img_root, p, s)):
                if f.lower().endswith(".png"):
                    sizes[os.path.splitext(f)[0]] = png_size(os.path.join(img_root, p, s, f))
            aspects += [w / h for w, h in sizes.values()]
            # shift of every crypt mask of the slide
            mdir = os.path.join(args.root, "inf", p, s, "predicted_mask")
            for f in os.listdir(mdir):
                m = re.match(r"(.*)_(\d+)\.png$", f)
                if not m or m.group(1) not in sizes:
                    continue
                w, h = sizes[m.group(1)]
                l, t, wp, hp = imgaug_pad(w, h)
                ys, xs = np.nonzero(cv2.imread(os.path.join(mdir, f), cv2.IMREAD_GRAYSCALE))
                if len(xs) == 0:
                    continue
                xm, ym = xs.mean(), ys.mean()
                xt, yt = xm * wp / 1024 - l, ym * hp / 768 - t           # tile point under the mask centre
                xu = (xt * 1024 / w + l2) * 1024 / wp2                    # where that tissue is in the patch frame
                yu = (yt * 768 / h + t2) * 768 / hp2
                shifts.append(float(np.hypot(xu - xm, yu - ym)))
    a, d = np.array(aspects), np.array(shifts)
    print(f"tiles: {len(a)}; 4:3 tiles: {int(np.sum(np.abs(a - 4 / 3) < 0.002))}; aspect W/H {a.min():.2f} to {a.max():.2f}")
    print(f"crypt masks: {len(d)}; shift median {np.median(d):.1f} px; 90th percentile {np.quantile(d, 0.9):.1f} px; "
          f"max {d.max():.1f} px; share > 10 px {100 * np.mean(d > 10):.1f}%")


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/validation/check_patch_frame_shift.py --root ../inference          # TF masks
#   python src/validation/check_patch_frame_shift.py --root inference             # PyTorch masks
