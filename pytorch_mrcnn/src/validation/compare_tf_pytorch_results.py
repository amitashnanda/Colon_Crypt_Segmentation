"""
Colon Crypt Segmentation (PyTorch) - Validation: Compare Two Result Roots at Every Stage

This script compares two pipeline result roots (for example TF vs PyTorch) at every stage. A root is a
results folder like inference/ (TF, repository root) or pytorch_mrcnn/inference/:
    inf/<protocol>/<slide>/predicted_mask/*.png, us_split/..., us_final/..., final/<protocol>/<slide>/final.txt
Any stage that is missing in a root is skipped. Mouse-DSS is excluded.

Reports: crypt masks (identical masks; greedy IoU >= 0.5 matching), crypt counts per stage and slide,
byte-identical U-shape patches, the per-slide brown values and their agreement, and the Welch p-values.
The one-command PyTorch runner calls it at the end (-> comparison_with_tf.json).

Usage (torch_env, from pytorch_mrcnn/):
    python src/validation/compare_tf_pytorch_results.py --a ../inference --a_name TF \\
        --b inference --b_name PyTorch --json validation/compare/tf_original_vs_torch_parity.json

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import collections
import json
import os
import re

import cv2
import numpy as np
from scipy import stats


# --- Default Settings ---
PROTOCOLS = ["Boiling-2min", "Boiling-5min", "Boiling-15min", "Pressure-3min"]
BOILING = PROTOCOLS[:3]
EXCLUDE = {"Mouse-DSS"}
COUNT_KEYS = ["n_masks", "split_accepted", "split_fragmented", "split_rejected", "final_accepted", "final_fragmented", "final_n"]


# ---------------------------------------------------------------------------------------------
# Reading a result root
# ---------------------------------------------------------------------------------------------
def slides(root, stage="inf"):
    """
    Args:
        root: result root
        stage: stage folder to list (inf, us_split, ...)

    Returns:
        List of (protocol, slide), Mouse-DSS left out
    """
    out = []
    for p in PROTOCOLS:
        d = os.path.join(root, stage, p)
        if os.path.isdir(d):
            out += [(p, s) for s in sorted(os.listdir(d)) if s not in EXCLUDE and os.path.isdir(os.path.join(d, s))]
    return out


def n_png(d):
    """
    Args:
        d: folder

    Returns:
        Number of PNG files in it, or None if it does not exist
    """
    return sum(1 for f in os.listdir(d) if f.endswith(".png")) if os.path.isdir(d) else None


def tile_masks(d):
    """
    Group the mask files of a predicted_mask/ folder by tile.

    Args:
        d: predicted_mask/ folder (<tile>_<j>.png files)

    Returns:
        Dict tile -> {file name: path}
    """
    by = collections.defaultdict(dict)
    if os.path.isdir(d):
        for f in os.listdir(d):
            m = re.match(r"(.*)_(\d+)\.png$", f)
            if m:
                by[m.group(1)][f] = os.path.join(d, f)
    return by


def read_bool(p):
    """
    Args:
        p: mask PNG

    Returns:
        bool mask (pixel > 0)
    """
    return cv2.imread(p, cv2.IMREAD_GRAYSCALE) > 0


def final_value(root, p, s):
    """
    Per-slide value of a slide: (brown top 25% - brown bottom 25%) / number of crypts, from final.txt.

    Args:
        root: result root
        p: protocol
        s: slide

    Returns:
        Dict with n (number of crypts) and value, or None if there is no final.txt
    """
    f = os.path.join(root, "final", p, s, "final.txt")
    if not os.path.isfile(f):
        return None
    v = open(f).read().strip().splitlines()[1].split("\t")
    n = float(v[1])
    return dict(n=int(n), value=(float(v[8]) - float(v[9])) / n)


def stage_counts(root, p, s):
    """
    Crypt counts of one slide at every stage, and its per-slide value.

    Args:
        root: result root
        p: protocol
        s: slide

    Returns:
        Dict with split_*, final_*, n_masks, final_n and value (None where a stage is missing)
    """
    st = {}
    for c in ("accepted", "fragmented", "rejected"):
        st["split_" + c] = n_png(os.path.join(root, "us_split", p, s, c))
    for c in ("accepted", "fragmented"):
        st["final_" + c] = n_png(os.path.join(root, "us_final", p, s, c))
    st["n_masks"] = n_png(os.path.join(root, "inf", p, s, "predicted_mask"))
    fv = final_value(root, p, s)
    st["final_n"] = fv["n"] if fv else None
    st["value"] = fv["value"] if fv else None
    return st


# ---------------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------------
def compare_masks(dir_a, dir_b):
    """
    Compare the crypt masks of one slide: identical masks and greedy IoU >= 0.5 matching per tile.

    Args:
        dir_a, dir_b: predicted_mask/ folders of the slide in the two roots

    Returns:
        Dict of counts (a, b, matched, identical_masks, pixel_diff, pixels_a, tiles, tiles_identical) and
        recall, precision, f1, iou_median, iou_min
    """
    ta, tb = tile_masks(dir_a), tile_masks(dir_b)
    r = collections.Counter()
    ious = []
    for tile in sorted(set(ta) | set(tb)):
        A = {k: read_bool(v) for k, v in ta.get(tile, {}).items()}
        B = {k: read_bool(v) for k, v in tb.get(tile, {}).items()}
        r["a"] += len(A)
        r["b"] += len(B)
        same_names = set(A) == set(B)
        tile_identical = same_names and all(np.array_equal(A[k], B[k]) for k in A)
        r["tiles"] += 1
        r["tiles_identical"] += int(tile_identical)
        # all overlapping pairs, best IoU first
        areas_a = {k: int(v.sum()) for k, v in A.items()}
        areas_b = {k: int(v.sum()) for k, v in B.items()}
        pairs = []
        for ka, ma in A.items():
            for kb, mb in B.items():
                inter = int(np.logical_and(ma, mb).sum())
                if inter:
                    union = areas_a[ka] + areas_b[kb] - inter
                    pairs.append((inter / union if union else 0.0, ka, kb))
        pairs.sort(reverse=True)
        # greedy one-to-one matching
        ua, ub = set(), set()
        for v, ka, kb in pairs:
            if v < 0.5 or ka in ua or kb in ub:
                continue
            ua.add(ka)
            ub.add(kb)
            ious.append(v)
            r["matched"] += 1
            r["identical_masks"] += int(np.array_equal(A[ka], B[kb]))
            r["pixel_diff"] += int(np.sum(A[ka] != B[kb]))
            r["pixels_a"] += areas_a[ka]
    rec = r["matched"] / r["a"] if r["a"] else float("nan")
    pre = r["matched"] / r["b"] if r["b"] else float("nan")
    return dict(r, recall=rec, precision=pre, f1=2 * rec * pre / (rec + pre) if r["matched"] else 0.0,
                iou_median=float(np.median(ious)) if ious else float("nan"),
                iou_min=float(np.min(ious)) if ious else float("nan"))


def compare_patches(dir_a, dir_b):
    """
    Compare the U-shape patches of one category folder pixel by pixel.

    Args:
        dir_a, dir_b: us_final/<protocol>/<slide>/<category>/ folders in the two roots

    Returns:
        Counter with common, identical, only_a, only_b
    """
    r = collections.Counter()
    if not (os.path.isdir(dir_a) and os.path.isdir(dir_b)):
        return r
    fa, fb = set(os.listdir(dir_a)), set(os.listdir(dir_b))
    for f in fa & fb:
        a = cv2.imread(os.path.join(dir_a, f), cv2.IMREAD_COLOR)
        b = cv2.imread(os.path.join(dir_b, f), cv2.IMREAD_COLOR)
        r["common"] += 1
        r["identical"] += int(a is not None and b is not None and a.shape == b.shape and np.array_equal(a, b))
    r["only_a"] += len(fa - fb)
    r["only_b"] += len(fb - fa)
    return r


def welch(values):
    """
    Welch t-test of Pressure-3min against each Boiling protocol.

    Args:
        values: list of (protocol, per-slide value)

    Returns:
        Dict Boiling protocol -> p-value
    """
    pr = [v for p, v in values if p == "Pressure-3min"]
    return {b: float(stats.ttest_ind(pr, [v for p, v in values if p == b], equal_var=False).pvalue) for b in BOILING}


def compare_roots(A, B):
    """
    Compare every slide of root A with the same slide in root B.

    Args:
        A, B: result roots

    Returns:
        Tuple of (per-slide rows, summed mask counts, mask counts per protocol)
    """
    rows = []
    agg_masks = collections.Counter()
    proto_masks = {}
    for p, s in slides(A):
        row = dict(protocol=p, slide=s)
        # crypt masks
        m = compare_masks(os.path.join(A, "inf", p, s, "predicted_mask"), os.path.join(B, "inf", p, s, "predicted_mask"))
        row["masks"] = {k: (float(v) if isinstance(v, float) else v) for k, v in m.items()}
        for k in ("a", "b", "matched", "identical_masks", "pixel_diff", "pixels_a", "tiles", "tiles_identical"):
            agg_masks[k] += m[k]
        pm = proto_masks.setdefault(p, collections.Counter())
        for k in ("a", "b", "matched", "identical_masks"):
            pm[k] += m[k]
        # counts per stage and per-slide value
        for tag, root in (("a", A), ("b", B)):
            row[tag] = stage_counts(root, p, s)
        # U-shape patches
        pc = collections.Counter()
        for c in ("accepted", "fragmented"):
            pc.update(compare_patches(os.path.join(A, "us_final", p, s, c), os.path.join(B, "us_final", p, s, c)))
        row["patches"] = dict(pc)
        rows.append(row)
    return rows, agg_masks, proto_masks


def report(rows, agg_masks, proto_masks, A, B, a_name, b_name):
    """
    Print the comparison and build its summary.

    Args:
        rows, agg_masks, proto_masks: output of compare_roots
        A, B: result roots
        a_name, b_name: names of the two roots in the report

    Returns:
        Summary dict (written to --json)
    """
    # crypt masks
    print(f"=== {a_name} ({A}) vs {b_name} ({B}) ===")
    a, b = agg_masks["a"], agg_masks["b"]
    rec = agg_masks["matched"] / a if a else 0
    pre = agg_masks["matched"] / b if b else 0
    print(f"crypt masks: {a_name} {a}, {b_name} {b}; matched {agg_masks['matched']} (recall {rec:.4f}, precision {pre:.4f}); "
          f"identical masks {agg_masks['identical_masks']}; differing pixels in matched masks {agg_masks['pixel_diff']} of {agg_masks['pixels_a']}; "
          f"identical tiles {agg_masks['tiles_identical']}/{agg_masks['tiles']}")
    for p in PROTOCOLS:
        if p in proto_masks:
            pm = proto_masks[p]
            print(f"   {p:14s} masks {pm['a']}/{pm['b']}  matched {pm['matched']}  identical {pm['identical_masks']}")

    # counts per stage
    diffs = collections.Counter()
    for r in rows:
        for k in COUNT_KEYS:
            if r["a"][k] is not None and r["b"][k] is not None and r["a"][k] != r["b"][k]:
                diffs[k] += 1
    print("slides whose counts differ, per stage:", {k: diffs.get(k, 0) for k in COUNT_KEYS}, f"(of {len(rows)} slides)")

    # U-shape patches
    pt = collections.Counter()
    for r in rows:
        pt.update(r["patches"])
    if pt:
        print(f"U-shape patches with the same file name: {pt['common']}; byte-identical {pt['identical']}; "
              f"only in {a_name} {pt['only_a']}; only in {b_name} {pt['only_b']}")

    # per-slide values and p-values
    vals = [(r["protocol"], r["a"]["value"], r["b"]["value"]) for r in rows if r["a"]["value"] is not None and r["b"]["value"] is not None]
    summary = dict(masks=dict(agg_masks, recall=rec, precision=pre), per_protocol={k: dict(v) for k, v in proto_masks.items()},
                   count_diffs=dict(diffs), patches=dict(pt), rows=rows)
    if vals:
        x = np.array([v[1] for v in vals])
        y = np.array([v[2] for v in vals])
        d = y - x
        sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        pa = welch([(p, v) for p, v, _ in vals])
        pb = welch([(p, v) for p, _, v in vals])
        print(f"per-slide value ({len(vals)} slides): largest |difference| {np.abs(d).max():.2e}; mean bias {d.mean():+.2e}; "
              f"Pearson r {stats.pearsonr(x, y)[0]:.6f}")
        print("Welch p-values (Pressure-3min vs):")
        for k in BOILING:
            print(f"   {k:14s} {a_name} {pa[k]:.6g}   {b_name} {pb[k]:.6g}")
        summary.update(values=dict(n=len(vals), max_abs_diff=float(np.abs(d).max()), bias=float(d.mean()), sd=sd,
                                   r=float(stats.pearsonr(x, y)[0]), loa=[float(d.mean() - 1.96 * sd), float(d.mean() + 1.96 * sd)]),
                       pvalues={a_name: pa, b_name: pb})
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Compare two pipeline result roots (e.g. TF vs PyTorch) at every stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Roots
    parser.add_argument("--a", required=True, help="first result root, e.g. ../inference (TF)")
    parser.add_argument("--b", required=True, help="second result root, e.g. inference (PyTorch)")
    parser.add_argument("--a_name", default="A", help="name of the first root in the report")
    parser.add_argument("--b_name", default="B", help="name of the second root in the report")

    # Output
    parser.add_argument("--json", default=None, help="optional: write the full comparison here")
    args = parser.parse_args()

    A, B = args.a, args.b
    rows, agg_masks, proto_masks = compare_roots(A, B)
    summary = report(rows, agg_masks, proto_masks, A, B, args.a_name, args.b_name)
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump(summary, fh, indent=1, default=float)
        print("wrote", args.json)


if __name__ == "__main__":
    main()


# Example usage (torch_env, from pytorch_mrcnn/):
#
#   # TF results against the PyTorch results
#   python src/validation/compare_tf_pytorch_results.py --a ../inference --a_name TF \
#       --b inference --b_name PyTorch --json validation/compare/tf_original_vs_torch_parity.json
#
#   # TF results against a TF re-run on this machine (noise floor)
#   python src/validation/compare_tf_pytorch_results.py --a ../inference --a_name TF_original \
#       --b validation/tf_reference --b_name TF_rerun --json validation/compare/tf_original_vs_tf_rerun.json
