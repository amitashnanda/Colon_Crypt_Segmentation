"""
Colon Crypt Segmentation (PyTorch) - Mouse DSS Statistics: Per-Patch Brown Gradient and Welch Tests

This script computes, for every crypt patch of the mouse DSS experiment (boiling and pressure), the brown intensity
"top 25% minus bottom 25%" and compares the two protocols with Welch t-tests. The measurement of a patch is the
code of the TF script src/inference/color_detection_mouse.py (resize to 1200 x 400, nuclei by Otsu threshold and a
small opening + closing, brown hue 0-40 or 150-180 and blue hue 80-140 per row, Gaussian smoothing, brown /
(brown + blue), mean of the top quarter minus mean of the bottom quarter). Unlike that script it:
  * lists the patches in sorted order, so the tables do not depend on the file order of the filesystem;
  * applies the patch selection of the stored TF analysis by file name, from src/analysis/mouse_patch_selection.tsv:
    the 39 boiling patches of the stored TF analysis (inference/Mouse/boiling/patch_b_us_copy) and the 4 patches
    the mouse cells of src/inference/testing_data_analysis.ipynb drop before their test (rows 7, 11, 51, 54 of the
    stored table).

Outputs in --out:
    per_image_brown_intensity.tsv       the selected patches (protocol, image_path, value), as the stored TF table
    per_image_brown_intensity_all.tsv   every patch, with the columns selected and notebook_excluded
    summary_and_pvalue.tsv              counts, means and Welch test of the selected patches (stored column order)
    notebook_pvalue.tsv                 the same test without the 4 excluded patches (the notebook's p-value)
    violin_boiling_vs_pressure.png      violin + swarm plot of the selected patches (unless --no_plot)

Usage (torch_env, from pytorch_mrcnn/; src/inference/run_mouse_pipeline.py runs it as its last step):
    python src/analysis/mouse_patch_statistics.py --boiling_us inference/Mouse/boiling/patch_b_us \\
        --pressure_us inference/Mouse/pressure/patch_p_us --out inference/Mouse

Author: Amitash Nanda
Date: 2026-10-13
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import ttest_ind


# --- Default Settings ---
HERE = os.path.dirname(os.path.abspath(__file__))
SELECTION = os.path.join(HERE, "mouse_patch_selection.tsv")
CATEGORIES = ("accepted", "fragmented")
# measurement, as in src/inference/color_detection_mouse.py
TARGET_SIZE = (400, 1200)   # (height, width) as unpacked in brown_intensity_topminusbottom
SMOOTH_SIGMA = 15           # Gaussian smoothing of the per-row counts
MORPH_KERNEL = np.ones((2, 2), np.uint8)
BROWN_HSV_RANGES = ((0, 40, 1, 254, 1, 254), (150, 180, 1, 254, 1, 254))
BLUE_HSV_RANGE = (80, 140, 1, 254, 1, 254)
VALUE_COLUMN = "Brown Intensity(25% top - 25% bottom)"
# column order of the stored inference/Mouse/summary_and_pvalue.tsv
SUMMARY_COLUMNS = ["diff_mean_boiling_minus_pressure", "mean_boiling", "mean_pressure", "n_boiling", "n_pressure",
                   "p_value_welch", "welch_t"]


# ---------------------------------------------------------------------------------------------
# Measurement of one patch (the code of src/inference/color_detection_mouse.py)
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
    Nuclei mask of a patch: Otsu threshold on the non-zero grey pixels, then opening + closing (2 x 2).

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
    nm = cv2.morphologyEx(nm, cv2.MORPH_OPEN, MORPH_KERNEL)
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
    H, W = target_size
    img = cv2.resize(img_bgr, (W, H), interpolation=cv2.INTER_AREA)

    nm = nuclei_mask_bgr(img)
    img_nuclei = cv2.bitwise_and(img, img, mask=nm)

    brown_mask = np.logical_or(
        mask_builder(img_nuclei, *BROWN_HSV_RANGES[0]) > 0,
        mask_builder(img_nuclei, *BROWN_HSV_RANGES[1]) > 0
    ).astype(np.uint8) * 255
    blue_mask = mask_builder(img_nuclei, *BLUE_HSV_RANGE)

    brown_count = np.sum(brown_mask == 255, axis=1).astype(np.float64)
    blue_count = np.sum(blue_mask == 255, axis=1).astype(np.float64)

    brown_s = gaussian_filter1d(brown_count, sigma=smooth_sigma)
    blue_s = gaussian_filter1d(blue_count, sigma=smooth_sigma)

    total = brown_s + blue_s
    total[total == 0] = 1e-6
    brown_norm = brown_s / total

    n = len(brown_norm)
    if n == 0:
        return 0.0
    top25 = float(np.mean(brown_norm[: n // 4]))
    bottom25 = float(np.mean(brown_norm[(3 * n) // 4:]))
    return top25 - bottom25


# ---------------------------------------------------------------------------------------------
# Patches, selection, tables
# ---------------------------------------------------------------------------------------------
def load_selection(path):
    """
    Read the patch selection list.

    Args:
        path: TSV with the columns protocol, category, patch, selected, notebook_excluded ('#' lines are comments)

    Returns:
        DataFrame indexed by (protocol, patch)
    """
    sel = pd.read_csv(path, sep="\t", comment="#")
    return sel.set_index(["protocol", "patch"])


def measure_patches(protocol, us_dir, root):
    """
    Measure every patch of one protocol, in sorted order.

    Args:
        protocol: 'boiling' or 'pressure'
        us_dir: U-shape output folder with accepted/ and fragmented/
        root: folder the image paths of the tables are written relative to

    Returns:
        List of dicts (protocol, category, patch, image_path, value)
    """
    rows = []
    for cat in CATEGORIES:
        for p in sorted(glob.glob(os.path.join(us_dir, cat, "*.png"))):
            img = cv2.imread(p, cv2.IMREAD_COLOR)
            if img is None:
                sys.exit("cannot read %s" % p)
            rows.append({"protocol": protocol, "category": cat, "patch": os.path.basename(p),
                         "image_path": os.path.relpath(os.path.abspath(p), root),
                         VALUE_COLUMN: brown_intensity_topminusbottom(img)})
    return rows


def apply_selection(df, sel, allow_missing):
    """
    Add the columns selected and notebook_excluded, matched by protocol and file name.

    Args:
        df: per-patch table of all patches
        sel: selection list (load_selection)
        allow_missing: only warn (instead of stopping) if a selected patch of the list is not in df

    Returns:
        The table with the two columns (patches that are not in the list: not selected)
    """
    keys = list(zip(df.protocol, df.patch))
    unknown = [k for k in keys if k not in sel.index]
    for k in unknown:
        print("[WARN] patch not in the selection list (treated as not selected): %s %s" % k)
    df["selected"] = [int(sel.loc[k, "selected"]) if k in sel.index else 0 for k in keys]
    df["notebook_excluded"] = [int(sel.loc[k, "notebook_excluded"]) if k in sel.index else 0 for k in keys]
    present = set(keys)
    missing = [k for k in sel.index[sel.selected == 1] if k not in present]
    if missing:
        msg = "%d selected patches of the list are missing: %s" % (len(missing), ", ".join("%s/%s" % k for k in missing))
        if not allow_missing:
            sys.exit(msg)
        print("[WARN] " + msg)
    return df


def welch(df):
    """
    Welch t-test boiling vs pressure (as src/inference/color_detection_mouse.py: t of boiling minus pressure).

    Args:
        df: per-patch table with the columns protocol and VALUE_COLUMN

    Returns:
        dict with counts, means, mean difference, t and p
    """
    b = df[df.protocol == "boiling"][VALUE_COLUMN].values
    p = df[df.protocol == "pressure"][VALUE_COLUMN].values
    t, pv = ttest_ind(b, p, equal_var=False) if len(b) >= 2 and len(p) >= 2 else (np.nan, np.nan)
    return {"n_boiling": len(b), "n_pressure": len(p), "mean_boiling": float(np.mean(b)), "mean_pressure": float(np.mean(p)),
            "diff_mean_boiling_minus_pressure": float(np.mean(b) - np.mean(p)), "welch_t": float(t), "p_value_welch": float(pv)}


def save_violin_plot(df, p, path, title="Boiling vs Pressure"):
    """
    Violin + swarm plot of the per-patch values of both protocols (the layout of the stored TF figure).

    Args:
        df: per-patch table (protocol, VALUE_COLUMN)
        p: Welch p-value shown in the title
        path: PNG to write
        title: title before the p-value
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid", context="notebook")
    plot_df = df.copy()
    plot_df["protocol"] = plot_df["protocol"].map({"boiling": "Boiling", "pressure": "Pressure"})
    order = ["Boiling", "Pressure"]
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=300)
    # one colour per protocol and the violin areas normalised together, as seaborn < 0.13 drew it without hue
    sns.violinplot(data=plot_df, x="protocol", y=VALUE_COLUMN, hue="protocol", order=order, hue_order=order,
                   palette=sns.color_palette("deep", 2), legend=False, inner="quartile", linewidth=0.8, cut=0,
                   density_norm="area", common_norm=True, ax=ax)
    sns.swarmplot(data=plot_df, x="protocol", y=VALUE_COLUMN, order=order, color="k", alpha=0.35, size=4, ax=ax)
    ax.set_xlabel("Protocol")
    ax.set_ylabel("Brown Intensity (top 25% − bottom 25%)")
    ax.set_title(title + ("  (Welch p = %.3g)" % p if np.isfinite(p) else ""))
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close(fig)
    print("[SAVE] %s" % path)


def main():
    parser = argparse.ArgumentParser(
        description="Per-patch brown gradient (top 25% - bottom 25%) and Welch tests of the mouse DSS patches, "
                    "with the patch selection of the stored TF analysis applied by file name.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--boiling_us", required=True, help="boiling U-shape patches (folder with accepted/ fragmented/)")
    parser.add_argument("--pressure_us", required=True, help="pressure U-shape patches (folder with accepted/ fragmented/)")
    parser.add_argument("--out", required=True, help="output folder of the tables and the plot")
    parser.add_argument("--root", default=None, help="write image paths relative to this folder (default: --out)")

    # Options
    parser.add_argument("--selection", default=SELECTION, help="patch selection list (default: %(default)s)")
    parser.add_argument("--allow_missing", action="store_true", help="only warn if a selected patch is missing")
    parser.add_argument("--no_plot", action="store_true", help="do not draw the violin plots")
    args = parser.parse_args()

    out = os.path.abspath(args.out)
    root = os.path.abspath(args.root or args.out)
    os.makedirs(out, exist_ok=True)

    # --- 1. Measure every patch, in sorted order, and apply the selection by file name ---
    rows = measure_patches("boiling", args.boiling_us, root) + measure_patches("pressure", args.pressure_us, root)
    df = apply_selection(pd.DataFrame(rows), load_selection(args.selection), args.allow_missing)
    df[["protocol", "category", "patch", "image_path", VALUE_COLUMN, "selected", "notebook_excluded"]].to_csv(
        os.path.join(out, "per_image_brown_intensity_all.tsv"), sep="\t", index=False)
    sel = df[df.selected == 1]
    sel[["protocol", "image_path", VALUE_COLUMN]].to_csv(os.path.join(out, "per_image_brown_intensity.tsv"),
                                                         sep="\t", index=False)
    print("[INFO] patches: boiling %d (selected %d), pressure %d (selected %d)" % (
        (df.protocol == "boiling").sum(), (sel.protocol == "boiling").sum(),
        (df.protocol == "pressure").sum(), (sel.protocol == "pressure").sum()))

    # --- 2. Welch test of the selected patches (the stored summary) ---
    s = welch(sel)
    pd.DataFrame([s])[SUMMARY_COLUMNS].to_csv(os.path.join(out, "summary_and_pvalue.tsv"), sep="\t", index=False)
    print("[INFO] selected patches: Welch p = %.6g (t = %.4g)" % (s["p_value_welch"], s["welch_t"]))

    # --- 3. The notebook's test: without the 4 excluded patches ---
    nb = sel[sel.notebook_excluded == 0]
    w = welch(nb)
    excluded = ";".join("%s/%s" % k for k in zip(sel[sel.notebook_excluded == 1].protocol, sel[sel.notebook_excluded == 1].patch))
    pd.DataFrame([dict(w, excluded_patches=excluded)])[["excluded_patches"] + SUMMARY_COLUMNS].to_csv(
        os.path.join(out, "notebook_pvalue.tsv"), sep="\t", index=False)
    print("[INFO] without the 4 patches the notebook drops (%s): Welch p = %.6g" % (excluded, w["p_value_welch"]))

    # --- 4. Plots ---
    if not args.no_plot:
        save_violin_plot(sel, s["p_value_welch"], os.path.join(out, "violin_boiling_vs_pressure.png"))
        save_violin_plot(nb, w["p_value_welch"], os.path.join(out, "violin_boiling_vs_pressure_notebook.png"),
                         title="Boiling vs Pressure, notebook selection")


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/, torch_env):
#
#   # the PyTorch mouse patches (written by src/inference/run_mouse_pipeline.py)
#   python src/analysis/mouse_patch_statistics.py --boiling_us inference/Mouse/boiling/patch_b_us \
#       --pressure_us inference/Mouse/pressure/patch_p_us --out inference/Mouse
#
#   # the stored TF mouse patches, into a new folder (reproduces the values of ../inference/Mouse/per_image_brown_intensity.tsv)
#   python src/analysis/mouse_patch_statistics.py --boiling_us ../inference/Mouse/boiling/patch_b_us \
#       --pressure_us ../inference/Mouse/pressure/patch_p_us --out mouse_stats_tf
