"""
Colon Crypt Segmentation - Mouse DSS: Welch t-Test and Violin Plot of the Per-Patch Brown Gradient

This script is the standalone version of the mouse cells of src/inference/testing_data_analysis.ipynb (cells 10-19,
which stay as they are).

Environment: the PyTorch environment (pytorch_mrcnn/torch_env.yml: seaborn 0.13.2, matplotlib 3.10.9, pandas 3.0.2,
scipy 1.14.1); it needs only pandas, scipy, seaborn and matplotlib, and no GPU. The TF environment
(environment_cdx2_tf1_ref.yml) has no seaborn. With seaborn 0.13 the violins are coloured through hue (the call is
updated as in pytorch_mrcnn/src/analysis/plot_pvalues.py); with older seaborn the notebook's own call is used.
On inference/Mouse/per_image_brown_intensity.tsv it prints p = 0.0385 (0.03845932719726146, the notebook's value)
and draws the figure of plots/mouse_pvalue.png: same size (6000 x 3600 at 600 dpi), colours, layout, violins, points
and text. That file was drawn by the notebook with matplotlib 3.2.1, so the new figure is not pixel-identical: 0.30% of
the pixels differ in the PyTorch environment (antialiasing and glyph rendering of matplotlib 3.10.9), and 0.09% with
matplotlib 3.2.1 and seaborn 0.9-0.11 added to the TF environment (only the rendering of the text glyphs).

Usage (from the repository root):
    python src/inference/plot_pvalues_mouse.py --tsv inference/Mouse/per_image_brown_intensity.tsv \\
        --out plots/mouse_pvalue.png

Author: Amitash Nanda
Date: 2026-10-13
"""

import argparse
import colorsys
import os

import matplotlib
matplotlib.use("Agg")  # no display on compute nodes; must be set before pyplot is imported
import matplotlib.colors as mc  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import ttest_ind  # noqa: E402


# --- Default Settings ---
VALUE_COLUMN = "Brown Intensity(25% top - 25% bottom)"
PROTOCOLS = ['boiling', 'pressure']
# the 4 patches the notebook drops (rows 7, 11, 51, 54 of inference/Mouse/per_image_brown_intensity.tsv)
EXCLUDE = ["patch_0015_pr_b3_2.png", "patch_0014_pr_b3_1.png", "patch_0003_14.png", "patch_0005_26.png"]
acolor = ["#00CC00", "#D8A03D", "#EC008C", 'cyan', "#B741DC", "#808285",
          'blue', 'black', 'green', 'red', 'orange', 'brown', 'pink', 'purple']   # palette of the notebook
SNS_HUE_PALETTE = tuple(int(v) for v in sns.__version__.split(".")[:2]) >= (0, 13)   # seaborn >= 0.13


def load_patches(path, exclude):
    """
    Read a per-patch table and leave out the excluded patches.

    Args:
        path: per_image_brown_intensity.tsv (columns protocol, image_path, VALUE_COLUMN)
        exclude: file names of the patches to leave out

    Returns:
        Tuple (table without the excluded patches, in the order of the file; table of the left-out rows)
    """
    df = pd.read_csv(path, sep='\t')
    names = df['image_path'].map(os.path.basename)
    dropped = df[names.isin(exclude)]
    missing = sorted(set(exclude) - set(names))
    if missing:
        print("[WARN] patches to leave out that are not in the table: %s" % ", ".join(missing))
    return df[~names.isin(exclude)], dropped


def adj_light(color, l=1, s=1):
    """
    Scale the lightness and saturation of a colour (notebook helper).

    Args:
        color: matplotlib colour name or hex string
        l: lightness factor
        s: saturation factor

    Returns:
        (r, g, b) tuple
    """
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, l * c[1])), max(0, min(1, s * c[2])))


def welch_pvalue(df_scores):
    """
    Welch t-test of pressure against boiling.

    Args:
        df_scores: DataFrame with columns 'score' and 'category'

    Returns:
        p-value
    """
    pressure = df_scores[df_scores['category'] == 'pressure']['score']
    boiling = df_scores[df_scores['category'] == 'boiling']['score']
    return ttest_ind(pressure, boiling, equal_var=False, nan_policy='omit')[1]


def plot_violin(data, atypes, out_path, dpi=600, w=10, h=6, width=0.95):
    """
    Violin + swarm plot of the per-patch values with the Welch p-value (cell 19 of the notebook).

    Args:
        data: one list of per-patch values per protocol
        atypes: protocol names, in the order of `data`
        out_path: PNG file to write
        dpi: resolution
        w, h: figure size in inches
        width: violin width

    Returns:
        p-value (pressure vs boiling)
    """
    df = pd.DataFrame()
    df['score'] = [k for i in range(len(data)) for k in data[i]]
    df['category'] = [atypes[i] for i in range(len(data)) for k in data[i]]
    pval = welch_pvalue(df)
    sns.set()
    sns.set_style("white")
    sns.set_style({'text.color': '.5', 'xtick.color': '.5', 'ytick.color': '.5', 'axes.labelcolor': '.5'})
    sns.set_context("notebook")
    palette = [adj_light(c, 1.5, 1) for c in acolor]
    sns.set_palette(palette)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    fig.subplots_adjust(bottom=0.125)   # the notebook's inline backend sets figure.subplot.bottom = 0.125
    buffer = 0.4
    rng = max(map(max, data)) - min(map(min, data))
    ax.set_ylim(min(map(min, data)) - buffer * rng, max(map(max, data)) + buffer * rng)
    # seaborn >= 0.13 colours the violins only through hue and then normalises each violin on its own; older
    # seaborn takes the palette set above and gives all violins the same area (density_norm="area", common_norm)
    hue = dict(hue="category", palette=palette, legend=False, dodge=False, density_norm="area",
               common_norm=True) if SNS_HUE_PALETTE else {}
    ax = sns.violinplot(x="category", y="score", inner='quartile', linewidth=0.5, width=width, ax=ax, data=df,
                        order=atypes, **hue)
    ax = sns.swarmplot(x="category", y="score", color='blue', alpha=0.2, ax=ax, data=df, order=atypes)
    ax.set_xlabel("Staining Protocol")
    ax.set_ylabel('Brown Intensity (25% top - 25% bottom)')
    ax.yaxis.grid(True, linestyle='-', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    text = f"p={pval:.3g}" + ('*' if pval < 0.05 else '')
    plt.text(0, max(df[df['category'] == 'boiling']['score']), text, horizontalalignment='center', size=12, color='0.3')
    ax.yaxis.grid(True, clip_on=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, format='png', dpi=dpi)
    plt.close(fig)
    return pval


def main():
    parser = argparse.ArgumentParser(
        description="Welch t-test (pressure vs boiling) and violin plot of the per-patch brown gradient of the mouse "
                    "DSS patches, without the 4 patches the notebook drops.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--tsv", required=True, help="per-patch table, e.g. inference/Mouse/per_image_brown_intensity.tsv")
    parser.add_argument("--out", required=True, help="output PNG")

    # Options
    parser.add_argument("--exclude", nargs="*", default=EXCLUDE,
                        help="file names of the patches to leave out (default: the 4 patches the notebook drops; "
                             "give --exclude without names to keep all)")
    args = parser.parse_args()

    # --- 1. Per-patch values of both protocols, without the excluded patches ---
    df, dropped = load_patches(args.tsv, args.exclude)
    for _, r in dropped.iterrows():
        print(f"  left out: {r['protocol']:8s} {os.path.basename(r['image_path'])}")
    sel = df[df['protocol'].isin(PROTOCOLS)]
    protocols = sel['protocol'].unique()
    data = [sel[sel['protocol'] == p][VALUE_COLUMN].tolist() for p in protocols]

    # --- 2. Test and plot ---
    pval = plot_violin(data, protocols, args.out)
    print(f"saved {args.out}")
    print("  " + ", ".join(f"{p} {len(d)} patches" for p, d in zip(protocols, data)))
    print(f"  pressure vs boiling: Welch p = {pval:.3g} ({pval!r})")


if __name__ == "__main__":
    main()


# Example usage (from the repository root, PyTorch environment):
#
#   # the mouse figure of the paper from the stored table (prints p = 0.0385)
#   python src/inference/plot_pvalues_mouse.py --tsv inference/Mouse/per_image_brown_intensity.tsv \
#       --out plots/mouse_pvalue.png
#
#   # all 84 patches (no patch left out)
#   python src/inference/plot_pvalues_mouse.py --tsv inference/Mouse/per_image_brown_intensity.tsv \
#       --out mouse_all.png --exclude
