"""
Colon Crypt Segmentation (PyTorch) - Validation: Stage-by-Stage Comparison with TF

This script compares the PyTorch stage outputs with the TF reference, tile by tile. It reads the per-tile
.npz files written by src/validation/tf_reference_inference.py (<dest>/stages) and by
src/inference/my_inference.py --stages_dir, and reports, per stage, how often the two agree:
proposals (1000 ROIs), class probabilities, detections, raw 28 x 28 masks and the final boxes and masks.

Usage (torch_env, from pytorch_mrcnn/):
    python src/validation/compare_stages_with_tf.py --tf validation/tf_reference/stages \\
        --torch validation/torch_stages --json validation/compare/stages_torch_vs_tf_rerun.json

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import json
import os

import numpy as np


def unpack(z):
    """
    Final masks of one tile from its stage file (stored bit-packed).

    Args:
        z: loaded .npz of one tile

    Returns:
        [H, W, N] bool masks
    """
    shape = tuple(z["mask_shape"])
    if len(shape) < 3 or shape[2] == 0:
        return np.zeros(shape if len(shape) == 3 else (768, 1024, 0), bool)
    return np.unpackbits(z["masks"], axis=-1, count=shape[-1]).astype(bool).reshape(shape)


def new_summary():
    """
    Returns:
        Empty running summary (counts and largest differences over all tiles)
    """
    return dict(tiles=0, rois_identical=0, rois_maxdiff=0.0, rois_rows_off=0, det_same_n=0, det_same_all=0,
                det_box_maxdiff=0.0, score_maxdiff=0.0, final_same_n=0, final_same_boxes=0, mask_px_diff=0,
                mask_px_total=0, tiles_identical=0, raw_mask_maxdiff=0.0, cls_maxdiff=0.0)


def compare_tile(a, b, agg):
    """
    Compare the stage outputs of one tile and add them to the running summary.

    Args:
        a: TF stage file of the tile (loaded .npz)
        b: PyTorch stage file of the tile
        agg: running summary (updated in place)

    Returns:
        Tuple of (final result identical, one-line description of the differences)
    """
    agg["tiles"] += 1
    # proposals (ROI layer)
    ra, rb = a["rois"], b["rois"]
    d = np.abs(ra.astype(np.float64) - rb.astype(np.float64))
    rows_off = int(np.sum(d.max(axis=1) > 1e-4))
    agg["rois_identical"] += int(rows_off == 0)
    agg["rois_rows_off"] += rows_off
    agg["rois_maxdiff"] = max(agg["rois_maxdiff"], float(d.max()))
    if rows_off == 0:
        agg["cls_maxdiff"] = max(agg["cls_maxdiff"], float(np.abs(a["mrcnn_class"] - b["mrcnn_class"]).max()))
    # detections (and the raw masks of matching detections)
    da, db = a["detections"], b["detections"]
    same_n = len(da) == len(db)
    agg["det_same_n"] += int(same_n)
    same_all = False
    if same_n and len(da):
        same_cls = np.array_equal(da[:, 4], db[:, 4])
        bd = float(np.abs(da[:, :4] - db[:, :4]).max())
        sd = float(np.abs(da[:, 5] - db[:, 5]).max())
        agg["det_box_maxdiff"] = max(agg["det_box_maxdiff"], bd)
        agg["score_maxdiff"] = max(agg["score_maxdiff"], sd)
        same_all = same_cls and bd < 1e-3
        if same_all:
            agg["raw_mask_maxdiff"] = max(agg["raw_mask_maxdiff"], float(np.abs(a["mrcnn_mask"] - b["mrcnn_mask"]).max()))
    elif same_n:
        same_all = True
    agg["det_same_all"] += int(same_all)
    # final boxes and masks
    ma, mb = unpack(a), unpack(b)
    fa = np.array_equal(a["class_ids"], b["class_ids"]) and ma.shape == mb.shape
    agg["final_same_n"] += int(fa)
    identical = False
    if fa:
        agg["final_same_boxes"] += int(np.array_equal(a["boxes"], b["boxes"]))
        px = int(np.sum(ma != mb))
        agg["mask_px_diff"] += px
        agg["mask_px_total"] += int(ma.sum())
        identical = px == 0 and np.array_equal(a["boxes"], b["boxes"])
    agg["tiles_identical"] += int(identical)
    detail = (f"rois rows off {rows_off} (max diff {d.max():.2e}); detections TF {len(da)} vs PT {len(db)}; "
              f"final TF {len(a['class_ids'])} vs PT {len(b['class_ids'])}"
              + (f"; mask pixels differ {int(np.sum(ma != mb))}" if fa else ""))
    return identical, detail


def print_summary(agg):
    """
    Print the per-stage summary.

    Args:
        agg: running summary after all tiles
    """
    n = agg["tiles"]
    print(f"tiles compared: {n}")
    print(f"proposals identical (all 1000 rows within 1e-4): {agg['rois_identical']}/{n}; rows off in total {agg['rois_rows_off']}; "
          f"largest coordinate difference {agg['rois_maxdiff']:.2e}")
    print(f"class probabilities, tiles with identical proposals: largest difference {agg['cls_maxdiff']:.2e}")
    print(f"detections: same count {agg['det_same_n']}/{n}; same classes and boxes (<1e-3) {agg['det_same_all']}/{n}; "
          f"largest box diff {agg['det_box_maxdiff']:.2e}; largest score diff {agg['score_maxdiff']:.2e}")
    print(f"raw 28x28 masks on matching detections: largest difference {agg['raw_mask_maxdiff']:.2e}")
    print(f"final results: same detections {agg['final_same_n']}/{n}; same pixel boxes {agg['final_same_boxes']}/{n}; "
          f"mask pixels differing {agg['mask_px_diff']} of {agg['mask_px_total']}; tiles identical {agg['tiles_identical']}/{n}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PyTorch stage outputs (.npz per tile) with the TF reference, stage by stage.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--tf", required=True, help="TF stage files, e.g. validation/tf_reference/stages")
    parser.add_argument("--torch", required=True, help="PyTorch stage files, e.g. validation/torch_stages")
    parser.add_argument("--json", default=None, help="optional: write the summary here")

    # Options
    parser.add_argument("--show", type=int, default=8, help="print details for up to N differing tiles")
    args = parser.parse_args()

    # tiles present on both sides
    tiles = sorted(os.path.relpath(p, args.torch) for p in glob.glob(os.path.join(args.torch, "**", "*.npz"), recursive=True))
    tiles = [t for t in tiles if os.path.exists(os.path.join(args.tf, t))]

    agg = new_summary()
    shown = 0
    for t in tiles:
        a = np.load(os.path.join(args.tf, t))
        b = np.load(os.path.join(args.torch, t))
        identical, detail = compare_tile(a, b, agg)
        if not identical and shown < args.show:
            shown += 1
            print(f"  {t}: {detail}")
    print_summary(agg)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(agg, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()


# Example usage (torch_env, from pytorch_mrcnn/):
#
#   python src/validation/compare_stages_with_tf.py --tf validation/tf_reference/stages \
#       --torch validation/torch_stages --json validation/compare/stages_torch_vs_tf_rerun.json
