"""
Colon Crypt Segmentation - Colour Detection for the Mouse DSS Patches

This script is the mouse DSS variant of the colour step (step 4). It computes, for every extracted
crypt patch of the boiling and the pressure set, the brown intensity "top 25% minus bottom 25%" of the
crypt, and compares the two sets with a Welch t-test.

Per patch: resize, keep the nuclei (Otsu threshold on the non-zero pixels, small opening + closing),
count brown (hue 0-40 or 150-180) and blue (hue 80-140) nuclei pixels per row, smooth, normalise
(brown / (brown + blue)) and take the mean over the top quarter minus the mean over the bottom quarter.

Outputs in --out (inference/Mouse/ for the paper):
    per_image_brown_intensity.tsv    one value per patch (read by src/inference/testing_data_analysis.ipynb)
    summary_and_pvalue.tsv           counts, means and the Welch t-test
    violin_boiling_vs_pressure.png   with --plot (needs seaborn)

Usage (from the repository root):
    python src/inference/color_detection_mouse.py --boiling_root <dir with patch_b_us> \\
        --pressure_root <dir with patch_p_us> --out inference/Mouse --plot

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
import cv2
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind


# --- Default Settings ---
# (height, width) as unpacked in brown_intensity_topminusbottom: patches are resized to 1200 x 400
# (width x height), unlike color_detection.py (400 x 1200); kept as used for the mouse results
TARGET_SIZE = (400, 1200)
SMOOTH_SIGMA = 15       # Gaussian smoothing of the per-row counts
MORPH_KERNEL = np.ones((2, 2), np.uint8)   # opening + closing of the nuclei mask
# HSV ranges (OpenCV hue 0-180) as (h_low, h_high, s_low, s_high, v_low, v_high)
BROWN_HSV_RANGES = ((0, 40, 1, 254, 1, 254), (150, 180, 1, 254, 1, 254))
BLUE_HSV_RANGE = (80, 140, 1, 254, 1, 254)
VALUE_COLUMN = "Brown Intensity(25% top - 25% bottom)"


# ---------------------------------------------------------------------------------------------
# Colour measurement of one patch
# ---------------------------------------------------------------------------------------------
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
    lower = np.array([hl, sl, vl], dtype=np.uint8)
    upper = np.array([hh, sh, vh], dtype=np.uint8)
    return cv2.inRange(hsv, lower, upper)


def nuclei_mask_bgr(img_bgr):
    """
    Nuclei mask of a patch: Otsu threshold on the non-zero grey pixels, then opening + closing (2 x 2),
    as in color_detection.py.

    Args:
        img_bgr: BGR patch

    Returns:
        uint8 mask (1 = nucleus)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if np.all(gray == 0):
        return np.zeros_like(gray, np.uint8)
    nz = gray[gray != 0]
    if nz.size == 0:
        return np.zeros_like(gray, np.uint8)
    thr, _ = cv2.threshold(nz, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    nm = np.logical_and(0 < gray, gray <= thr).astype(np.uint8)
    nm = cv2.morphologyEx(nm, cv2.MORPH_OPEN,  MORPH_KERNEL)
    nm = cv2.morphologyEx(nm, cv2.MORPH_CLOSE, MORPH_KERNEL)
    return nm


def brown_intensity_topminusbottom(img_bgr, target_size=TARGET_SIZE, smooth_sigma=SMOOTH_SIGMA):
    """
    Brown intensity (top 25% - bottom 25%) of one patch.

    Args:
        img_bgr: BGR patch
        target_size: (height, width) unpacked as H, W; the patch is resized to (W, H) with INTER_AREA
        smooth_sigma: Gaussian smoothing of the per-row counts

    Returns:
        Mean normalised brown over the top quarter minus the mean over the bottom quarter
    """
    H, W = target_size  # (height, width)
    img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    nm = nuclei_mask_bgr(img)
    img_nuclei = cv2.bitwise_and(img, img, mask=nm)

    # Brown = (0..40) U (150..180); Blue = 80..140
    brown_mask = np.logical_or(
        mask_builder(img_nuclei, *BROWN_HSV_RANGES[0]) > 0,
        mask_builder(img_nuclei, *BROWN_HSV_RANGES[1]) > 0
    ).astype(np.uint8) * 255
    blue_mask  = mask_builder(img_nuclei, *BLUE_HSV_RANGE)

    brown_count = np.sum(brown_mask == 255, axis=1).astype(np.float64)
    blue_count  = np.sum(blue_mask  == 255, axis=1).astype(np.float64)

    brown_s = gaussian_filter1d(brown_count, sigma=smooth_sigma)
    blue_s  = gaussian_filter1d(blue_count,  sigma=smooth_sigma)

    total = brown_s + blue_s
    total[total == 0] = 1e-6
    brown_norm = brown_s / total

    n = len(brown_norm)
    if n == 0:
        return 0.0
    top25    = float(np.mean(brown_norm[: n//4]))
    bottom25 = float(np.mean(brown_norm[(3*n)//4 :]))
    return top25 - bottom25


# ---------------------------------------------------------------------------------------------
# Folder scanning
# ---------------------------------------------------------------------------------------------
def find_patch_dir(root, expect='b'):
    """
    Find the patch folder of one protocol.

    Inside `root` a single patch folder is expected: one whose name contains 'patch_b_us' for boiling,
    'patch_p_us' for pressure (case-insensitive). If `root` is that folder itself, it is used as is.

    Args:
        root: protocol root or patch folder
        expect: 'b' (boiling) or 'p' (pressure)

    Returns:
        Path of the patch folder (the first in sorted order if there are several)
    """
    root = os.path.abspath(root)
    last = os.path.basename(root).lower()
    if (expect == 'b' and 'patch_b_us' in last) or (expect == 'p' and 'patch_p_us' in last):
        return root  # already the patch dir

    # otherwise search one level deep
    candidates = []
    for name in os.listdir(root):
        p = os.path.join(root, name)
        if not os.path.isdir(p):
            continue
        low = name.lower()
        if expect == 'b' and 'patch_b_us' in low:
            candidates.append(p)
        if expect == 'p' and 'patch_p_us' in low:
            candidates.append(p)
    if len(candidates) == 0:
        raise FileNotFoundError(f"No expected patches folder found under {root} (expect={'patch_b_us' if expect=='b' else 'patch_p_us'})")
    if len(candidates) > 1:
        # pick first deterministically
        candidates.sort()
    return candidates[0]


def collect_metrics_from_patch_dir(patch_dir):
    """
    Brown intensity of every PNG patch in the accepted/ and fragmented/ subfolders of a patch folder.

    Args:
        patch_dir: folder with accepted/ and fragmented/

    Returns:
        List of (patch path, value)
    """
    vals = []
    # consider only these leafs
    for leaf in ('accepted', 'fragmented'):
        leaf_dir = os.path.join(patch_dir, leaf)
        if not os.path.isdir(leaf_dir):
            continue
        for p in glob.glob(os.path.join(leaf_dir, "*.png")):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                val = brown_intensity_topminusbottom(img)
                vals.append((p, val))
            except Exception:
                # skip any odd failures, keep pipeline going
                continue
    return vals  # list of (path, value)


# ---------------------------------------------------------------------------------------------
# Tables, test and plot
# ---------------------------------------------------------------------------------------------
def save_violin_plot(df, p, out_dir):
    """
    Violin + swarm plot of the per-patch values of both protocols (needs seaborn).

    Args:
        df: per-patch table (protocol, image_path, value)
        p: Welch p-value (shown in the title if finite)
        out_dir: output folder
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(style="whitegrid", context="notebook")
        plot_df = df.copy()
        plot_df["protocol"] = plot_df["protocol"].map({"boiling":"Boiling","pressure":"Pressure"})
        fig, ax = plt.subplots(figsize=(6.5,5), dpi=300)
        order = ["Boiling","Pressure"]
        sns.violinplot(data=plot_df, x="protocol", y=VALUE_COLUMN,
                       order=order, inner="quartile", linewidth=0.8, cut=0, ax=ax)
        sns.swarmplot(data=plot_df, x="protocol", y=VALUE_COLUMN,
                      order=order, color="k", alpha=0.35, size=4, ax=ax)
        ax.set_xlabel("Protocol")
        ax.set_ylabel("Brown Intensity (top 25% − bottom 25%)")
        title = "Boiling vs Pressure"
        if np.isfinite(p):
            title += f"  (Welch p = {p:.3g})"
        ax.set_title(title)
        plt.tight_layout()
        out_png = os.path.join(out_dir, "violin_boiling_vs_pressure.png")
        plt.savefig(out_png, dpi=300)
        plt.close(fig)
        print(f"[SAVE] {out_png}")
    except Exception as e:
        print("[WARN] Plotting skipped:", e)


def run(boiling_root, pressure_root, out_dir, plot=False):
    """
    Measure both protocols, write the per-patch table and the Welch t-test summary (and the plot).

    Args:
        boiling_root: boiling root (contains patch_b_us) or the patch folder itself
        pressure_root: pressure root (contains patch_p_us) or the patch folder itself
        out_dir: output folder for the TSV files and the plot
        plot: also save the violin plot
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- 1. Locate the patch folders and measure every patch ---
    b_patch = find_patch_dir(boiling_root, expect='b')
    p_patch = find_patch_dir(pressure_root, expect='p')

    print("[INFO] Boiling patch dir :", b_patch)
    print("[INFO] Pressure patch dir:", p_patch)

    b_list = collect_metrics_from_patch_dir(b_patch)
    p_list = collect_metrics_from_patch_dir(p_patch)
    b_vals = np.array([v for _, v in b_list], dtype=float)
    p_vals = np.array([v for _, v in p_list], dtype=float)

    print(f"[INFO] Counts -> Boiling: {len(b_vals)}   Pressure: {len(p_vals)}")
    print(f"[INFO] Means  -> Boiling: {b_vals.mean():.6f}   Pressure: {p_vals.mean():.6f}")

    # --- 2. Per-patch table ---
    df = pd.DataFrame({
        "protocol": (["boiling"]*len(b_vals)) + (["pressure"]*len(p_vals)),
        "image_path": [p for p,_ in b_list] + [p for p,_ in p_list],
        VALUE_COLUMN: np.r_[b_vals, p_vals]
    })
    per_image_path = os.path.join(out_dir, "per_image_brown_intensity.tsv")
    df.to_csv(per_image_path, sep="\t", index=False)
    print(f"[SAVE] {per_image_path}")

    # --- 3. Welch t-test and summary ---
    if len(b_vals) >= 2 and len(p_vals) >= 2:
        t, p = ttest_ind(b_vals, p_vals, equal_var=False)
    else:
        t, p = np.nan, np.nan
        print("[WARN] Need ≥2 per group for a valid Welch t-test; reporting NaN.")

    summary = pd.DataFrame([{
        "n_boiling": len(b_vals),
        "n_pressure": len(p_vals),
        "mean_boiling": float(np.mean(b_vals) if len(b_vals) else np.nan),
        "mean_pressure": float(np.mean(p_vals) if len(p_vals) else np.nan),
        "diff_mean_boiling_minus_pressure": float((np.mean(b_vals) - np.mean(p_vals)) if len(b_vals) and len(p_vals) else np.nan),
        "welch_t": float(t) if np.isfinite(t) else np.nan,
        "p_value_welch": float(p) if np.isfinite(p) else np.nan
    }])
    summary_path = os.path.join(out_dir, "summary_and_pvalue.tsv")
    summary.to_csv(summary_path, sep="\t", index=False)
    print(f"[SAVE] {summary_path}")

    # --- 4. Optional plot ---
    if plot and len(df) > 0:
        save_violin_plot(df, p, out_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-image Brown Intensity (top25%−bottom25%) and Welch p-value for boiling vs pressure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--boiling_root", required=True,
                        help="Path to boiling root (contains patch_b_us) OR directly to patch_b_us")
    parser.add_argument("--pressure_root", required=True,
                        help="Path to pressure root (contains patch_p_us) OR directly to patch_p_us")
    parser.add_argument("--out", required=True, help="Output directory for TSVs (and optional plot)")

    # Options
    parser.add_argument("--plot", action="store_true", help="Save violin+swarm plot")
    args = parser.parse_args()

    run(args.boiling_root, args.pressure_root, args.out, plot=args.plot)


if __name__ == "__main__":
    main()


# Example usage (from the repository root):
#
#   python src/inference/color_detection_mouse.py --boiling_root <mouse boiling root>/ \
#       --pressure_root <mouse pressure root>/ --out inference/Mouse --plot
#   -> inference/Mouse/per_image_brown_intensity.tsv, summary_and_pvalue.tsv, violin_boiling_vs_pressure.png
