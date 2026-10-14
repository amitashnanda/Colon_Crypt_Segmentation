"""
Colon Crypt Segmentation (PyTorch) - Step 6: Welch t-Tests and Violin Plot

This script computes the statistics and the figure of the paper from a per-slide results table, with
the normalisation, statistic, test and plot of the TF notebook src/inference/testing_data_analysis.ipynb
(only the seaborn calls are updated for seaborn 0.13):
  * every column is divided by the number of crypts of the slide,
  * the per-slide value is brown top 25% minus brown bottom 25%,
  * Pressure-3min is compared with each Boiling protocol (Welch t-test, unequal variances),
  * a violin + swarm plot shows the four protocols with the p-values.

Usage (from pytorch_mrcnn/):
    python src/analysis/plot_pvalues.py --res inference/res_tfparity.txt \\
        --out inference/plots/human_pytorch_tfparity.png

Author: Amitash Nanda
Date: 2026-10-09
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
COLUMNS = ['Staining Protocol', 'size', 'bf50', 'be50', 'bf25', 'be25', 'rf50', 're50', 'rf25', 're25']
BOILING = ['Boiling-2min', 'Boiling-5min', 'Boiling-15min']
acolor = ["#00CC00", "#D8A03D", "#EC008C", 'cyan', "#B741DC", "#808285",
          'blue', 'black', 'green', 'red', 'orange', 'brown', 'pink', 'purple']   # palette of the notebook


def load_res(path):
    """
    Read a per-slide table and add the per-slide value.

    Args:
        path: table (res_new.txt, res_tfparity.txt, ...)

    Returns:
        DataFrame with the columns divided by the number of crypts and the column
        'Brown Intensity(25% top - 25% bottom)'
    """
    df = pd.read_csv(path, sep=r'\s+', names=COLUMNS, header=None, index_col=False)
    df['size'] = df['size'].astype(float)
    df.iloc[:, 1:] = df.iloc[:, 1:].div(df['size'].values, axis=0)
    df['Brown Intensity(25% top - 25% bottom)'] = df['rf25'] - df['re25']
    return df


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


def welch_pvalues(df_scores):
    """
    Welch t-test of Pressure-3min against each Boiling protocol.

    Args:
        df_scores: DataFrame with columns 'score' and 'category'

    Returns:
        List of p-values in the order of BOILING
    """
    pressure = df_scores[df_scores['category'] == 'Pressure-3min']['score']
    return [ttest_ind(pressure, df_scores[df_scores['category'] == b]['score'], equal_var=False, nan_policy='omit')[1]
            for b in BOILING]


def plot_violin(data, atypes, out_path, dpi=600, w=10, h=6, width=0.95):
    """
    Violin + swarm plot of the per-slide values with the Welch p-values.

    Args:
        data: one list of per-slide values per protocol
        atypes: protocol names, in the order of `data`
        out_path: PNG file to write
        dpi: resolution
        w, h: figure size in inches
        width: violin width

    Returns:
        List of p-values (Pressure-3min vs each Boiling protocol)
    """
    df = pd.DataFrame()
    df['score'] = [k for i in range(len(data)) for k in data[i]]
    df['category'] = [atypes[i] for i in range(len(data)) for k in data[i]]
    pvals = welch_pvalues(df)
    sns.set()
    sns.set_style("white")
    sns.set_style({'text.color': '.5', 'xtick.color': '.5', 'ytick.color': '.5', 'axes.labelcolor': '.5'})
    sns.set_context("notebook")
    palette = [adj_light(c, 1.5, 1) for c in acolor]
    sns.set_palette(palette)
    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    buffer = 0.4
    rng = max(map(max, data)) - min(map(min, data))
    ax.set_ylim(min(map(min, data)) - buffer * rng, max(map(max, data)) + buffer * rng)
    ax = sns.violinplot(x="category", y="score", hue="category", palette=palette, legend=False, dodge=False,
                        inner='quartile', linewidth=0.5, width=width, ax=ax, data=df, order=atypes)
    ax = sns.swarmplot(x="category", y="score", color='blue', alpha=0.2, ax=ax, data=df, order=atypes)
    ax.set_xlabel("Staining Protocol")
    ax.set_ylabel('Brown Intensity (25% top - 25% bottom)')
    ax.yaxis.grid(True, linestyle='-', alpha=0.5)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    for i, b in enumerate(BOILING):
        text = f"p={pvals[i]:.3g}" + ('*' if pvals[i] < 0.05 else '')
        plt.text(i, max(df[df['category'] == b]['score']), text, horizontalalignment='center', size=12, color='0.3')
    ax.yaxis.grid(True, clip_on=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, format='png', dpi=dpi)
    plt.close(fig)
    return pvals


def main():
    parser = argparse.ArgumentParser(
        description="Welch t-tests (Pressure-3min vs each Boiling protocol) and violin plot of a per-slide table.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--res", required=True, help="res table (e.g. res_new.txt or res_tfparity.txt)")
    parser.add_argument("--out", required=True, help="output PNG")
    args = parser.parse_args()

    # --- 1. Per-slide values of the four protocols ---
    df = load_res(args.res)
    sel = df[df['Staining Protocol'].isin(BOILING + ['Pressure-3min'])]
    protocols = sel['Staining Protocol'].unique()
    data = [sel[sel['Staining Protocol'] == p]['Brown Intensity(25% top - 25% bottom)'].tolist() for p in protocols]

    # --- 2. Tests and plot ---
    pvals = plot_violin(data, protocols, args.out)
    print(f"saved {args.out}")
    for b, p in zip(BOILING, pvals):
        print(f"  Pressure-3min vs {b}: p = {p:.3g}")


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/analysis/plot_pvalues.py --res inference/res_tfparity.txt \
#       --out inference/plots/human_pytorch_tfparity.png
#
#   # the TF table of the paper
#   python src/analysis/plot_pvalues.py --res ../inference/res_new.txt --out /tmp/human_tf_res_new.png
