"""
Colon Crypt Segmentation (PyTorch) - Step 5: Per-Slide Results Table

This script collects the colour-detection output (step 4) into the per-slide results table ("res"
file). It writes one row per slide, read from final/<protocol>/<slide>/final.txt:
    <protocol> <number_of_crypts> <8 colour columns>
in protocol order Boiling-2min, Boiling-5min, Boiling-15min, Pressure-3min and sorted slide order;
Mouse-DSS is skipped. The format is that of the TF tables read by plot_pvalues.py and
src/inference/testing_data_analysis.ipynb. Run on TF's final/ (../inference/final), it reproduces the
TF table res_new.txt byte for byte.

Usage (from pytorch_mrcnn/):
    python src/analysis/build_res_table.py --final_root inference/final --out inference/res_tfparity.txt

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os


# --- Default Settings ---
PROTOCOLS = ["Boiling-2min", "Boiling-5min", "Boiling-15min", "Pressure-3min"]   # row order of the table


def build_table(final_root, out_path, exclude):
    """
    Write the per-slide table.

    Args:
        final_root: output root of step 4 (final/<protocol>/<slide>/final.txt)
        out_path: table file to write
        exclude: slide folder names to leave out

    Returns:
        Number of rows (slides) written
    """
    rows = 0
    with open(out_path, "w") as out:
        for protocol in PROTOCOLS:
            pdir = os.path.join(final_root, protocol)
            if not os.path.isdir(pdir):
                continue
            for sub in sorted(os.listdir(pdir)):
                if sub in exclude:
                    continue
                f = os.path.join(pdir, sub, "final.txt")
                if not os.path.isfile(f):
                    continue
                # skip the header line; replace the slide name by the protocol
                for line in open(f).read().splitlines()[1:]:
                    if line.strip():
                        out.write(protocol + "\t" + "\t".join(line.split("\t")[1:]) + "\n")
                        rows += 1
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Per-slide results table from final/<protocol>/<slide>/final.txt.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--final_root", required=True, help="output root of the colour step, e.g. inference/final")
    parser.add_argument("--out", required=True, help="table to write, e.g. inference/res_tfparity.txt")

    # Options
    parser.add_argument("--exclude", nargs="*", default=["Mouse-DSS"], help="slide folders to leave out")
    args = parser.parse_args()

    rows = build_table(args.final_root, args.out, args.exclude)
    print(f"wrote {args.out} ({rows} slides)")


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   python src/analysis/build_res_table.py --final_root inference/final --out inference/res_tfparity.txt
#
#   # TF's colour output gives TF's table (byte-identical to ../inference/res_new.txt)
#   python src/analysis/build_res_table.py --final_root ../inference/final --out /tmp/res_new_check.txt
