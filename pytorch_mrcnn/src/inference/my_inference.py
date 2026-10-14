"""
Colon Crypt Segmentation (PyTorch) - Step 1: Mask R-CNN Inference

This script runs the PyTorch Mask R-CNN over a tree of colour-normalised, Hough-aligned tiles. The
network is the TF-parity engine in tfparity_maskrcnn.py with the converted weights
models/notransfer_torch.pth. Input, settings and output layout match the TF script
src/inference/my_inference.py at the repository root, and so do the crypt masks.

For every <src>/<rel>/<tile>.png it writes:
    <dest>/<rel>/predicted_images/<tile>.png     detections drawn on the tile (masks, boxes, scores)
    <dest>/<rel>/predicted_mask/<tile>_<j>.png   one 0/255 mask per crypt (class 1), j = detection index

Slide folders named in --skip (default Mouse-DSS) are left out. With --stages_dir, the output of
every network stage is also saved per tile (.npz), for src/validation/compare_stages_with_tf.py.

Device (--device): cuda for an NVIDIA GPU (default), cpu for a computer without one (e.g. a Mac), mps for the
GPU of an Apple-silicon Mac (not tested).

Usage (from pytorch_mrcnn/, torch_env, GPU node):
    python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \\
        --dest inference/inf --weights models/notransfer_torch.pth

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import colorsys
import glob
import os
import random
import sys
import time

import cv2
import matplotlib
matplotlib.use("Agg")  # no display on compute nodes; must be set before pyplot is imported
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from skimage.measure import find_contours  # noqa: E402

# TF-parity Mask R-CNN engine (same folder)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tfparity_maskrcnn import CLASS_NAMES, TFParityMaskRCNN, load_and_preprocess  # noqa: E402


# --- Default Settings ---
SEED = 123             # the TF script re-seeds Python's random module with 123 before every tile
FIGSIZE = (16, 16)     # size of the predicted_images figures in inches (display_instances default)
CRYPT_CLASS_ID = 1     # class ids: 0 bg, 1 crypt, 2 gland; only crypt masks are written


# ---------------------------------------------------------------------------------------------
# predicted_images: display_instances() of the TF library (mrcnn/visualize.py), as the TF script
# calls it
# ---------------------------------------------------------------------------------------------
def random_colors(N, bright=True):
    """
    Generate N distinct colours in HSV space, in random order (TF mrcnn/visualize.py).

    Args:
        N: number of colours
        bright: full brightness (1.0) if True, otherwise 0.7

    Returns:
        List of N (r, g, b) tuples in [0, 1], shuffled with Python's random module
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """
    Blend one instance mask into the image, in place (TF mrcnn/visualize.py).

    Args:
        image: H x W x 3 image array
        mask: H x W mask, 1 inside the instance
        color: (r, g, b) in [0, 1]
        alpha: opacity of the colour

    Returns:
        The blended image
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def save_predicted_image(image, boxes, masks, class_ids, scores, title, out_path, figsize=FIGSIZE):
    """
    Draw the detections of one tile as the TF script does and save the figure.

    Same drawing as display_instances(image, rois, masks, class_ids, ['bg', 'crypt', 'gland'], scores,
    title=<tile name>, dest=predicted_images/): a dashed box, a caption, the filled mask and its outline
    for every detection. Python's random module is seeded as in the TF loop, so the instance colours are
    the TF ones.

    Args:
        image: 768 x 1024 x 3 uint8 RGB tile (output of load_and_preprocess)
        boxes: [N, 4] boxes (y1, x1, y2, x2) in pixels
        masks: [H, W, N] bool masks
        class_ids: [N] class ids
        scores: [N] confidence scores
        title: figure title (the tile file name)
        out_path: PNG file to write
        figsize: figure size in inches
    """
    random.seed(SEED)
    N = boxes.shape[0]
    fig, ax = plt.subplots(1, figsize=figsize)
    colors = random_colors(N)
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        if not np.any(boxes[i]):
            continue
        # dashed bounding box and "<class> <score>" caption
        y1, x1, y2, x2 = boxes[i]
        ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, alpha=0.7, linestyle="dashed",
                                       edgecolor=color, facecolor='none'))
        label = CLASS_NAMES[class_ids[i]]
        caption = "{} {:.3f}".format(label, scores[i]) if scores[i] else label
        ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="none")
        # filled mask and its outline
        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)
        # pad so that masks touching the image edge still give closed outlines
        padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        for verts in find_contours(padded_mask, 0.5):
            verts = np.fliplr(verts) - 1
            ax.add_patch(Polygon(verts, facecolor="none", edgecolor=color))
    ax.imshow(masked_image.astype(np.uint8))
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Tiles and outputs
# ---------------------------------------------------------------------------------------------
def list_tiles(src, skip, only=None, limit=0):
    """
    List the tiles to process.

    Args:
        src: tile root, <src>/<protocol>/<slide>/<tile>.png
        skip: slide folder names to leave out (e.g. ["Mouse-DSS"])
        only: optional list of tile paths relative to src; if given, only these are kept
        limit: if > 0, keep only the first `limit` tiles

    Returns:
        Sorted list of tile paths
    """
    paths = sorted(glob.glob(os.path.join(src, "**", "*.png"), recursive=True))
    paths = [p for p in paths if not any(f"{os.sep}{s}{os.sep}" in p for s in skip)]
    if only:
        want = set(only)
        paths = [p for p in paths if os.path.relpath(p, src) in want]
    if limit:
        paths = paths[:limit]
    return paths


def save_stage_dump(stages_dir, rel_dir, stem, r):
    """
    Save the stage outputs of one tile for the tile-by-tile comparison with the TF reference.

    Args:
        stages_dir: root of the stage dumps
        rel_dir: folder of the tile relative to the tile root (<protocol>/<slide>)
        stem: tile file name without extension
        r: result of TFParityMaskRCNN.detect(..., return_stages=True)
    """
    sd = os.path.join(stages_dir, rel_dir)
    os.makedirs(sd, exist_ok=True)
    st = r["stages"]
    masks = r["masks"]
    np.savez_compressed(os.path.join(sd, f"{stem}.npz"), rois=st["rois"], mrcnn_class=st["mrcnn_class"],
                        mrcnn_bbox=st["mrcnn_bbox"], detections=st["detections"], mrcnn_mask=st["mrcnn_mask"],
                        boxes=r["rois"], class_ids=r["class_ids"], scores=r["scores"],
                        masks=np.packbits(masks.astype(bool), axis=-1) if masks.size else np.zeros((0,), np.uint8),
                        mask_shape=np.array(masks.shape))


def main():
    parser = argparse.ArgumentParser(
        description="PyTorch Mask R-CNN inference with TF parity (output layout of the TF "
                    "src/inference/my_inference.py).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", required=True,
                        help="tile root, e.g. dataset/Whole_Slides_Segments_Processed/CDX2_Data")
    parser.add_argument("--dest", required=True, help="output root (mirrors --src), e.g. inference/inf")
    parser.add_argument("--weights", required=True, help="converted state dict, e.g. models/notransfer_torch.pth")
    parser.add_argument("--device", default="cuda",
                        help="cuda: NVIDIA GPU (default); cpu: no NVIDIA GPU, e.g. a Mac; "
                             "mps: Apple-silicon GPU on a Mac (not tested)")

    # Tile selection
    parser.add_argument("--skip", nargs="*", default=["Mouse-DSS"], help="slide folders to skip")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N tiles (testing)")
    parser.add_argument("--only", nargs="*", default=None, help="optional: relative tile paths to process")

    # Outputs and engine options
    parser.add_argument("--no_images", action="store_true", help="write predicted_mask/ only, no predicted_images/")
    parser.add_argument("--nms_ties", default="heap", choices=["heap", "index"],
                        help="order in which NMS visits exactly equal scores: 'heap' (default) follows the "
                             "std::priority_queue of TF's CPU kernel and reproduces TF's proposals; 'index' visits "
                             "them in index order")
    parser.add_argument("--stages_dir", default=None, help="optional: save per-tile stage outputs here")
    args = parser.parse_args()

    # --- 1. Load the model and list the tiles ---
    model = TFParityMaskRCNN(args.weights, device=args.device, nms_ties=args.nms_ties)
    paths = list_tiles(args.src, args.skip, args.only, args.limit)
    print(f"[info] {len(paths)} tiles; device {args.device}; NMS ties '{args.nms_ties}'", flush=True)

    # --- 2. Detect on every tile; write crypt masks, drawings and (optionally) stage outputs ---
    t0 = time.time()
    for i, p in enumerate(paths):
        rel_dir = os.path.dirname(os.path.relpath(p, args.src))
        name = os.path.basename(p)
        stem = os.path.splitext(name)[0]
        mask_dir = os.path.join(args.dest, rel_dir, "predicted_mask")
        os.makedirs(mask_dir, exist_ok=True)

        image = load_and_preprocess(p)
        r = model.detect(image, return_stages=bool(args.stages_dir))

        # one 0/255 PNG per crypt; j is the index of the detection
        for j, cid in enumerate(r["class_ids"]):
            if cid == CRYPT_CLASS_ID:
                cv2.imwrite(os.path.join(mask_dir, f"{stem}_{j}.png"), (r["masks"][:, :, j] * 255).astype(np.uint8))
        if not args.no_images:
            img_dir = os.path.join(args.dest, rel_dir, "predicted_images")
            os.makedirs(img_dir, exist_ok=True)
            save_predicted_image(image, r["rois"], r["masks"], r["class_ids"], r["scores"], name, os.path.join(img_dir, name))
        if args.stages_dir:
            save_stage_dump(args.stages_dir, rel_dir, stem, r)

        if (i + 1) % 50 == 0 or i == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(paths)}] {os.path.relpath(p, args.src)}  {len(r['class_ids'])} det  ({el / (i + 1):.2f} s/tile)", flush=True)
    print(f"[done] {len(paths)} tiles in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()


# Example usage (from pytorch_mrcnn/):
#
#   # all human tiles, as in inference/inf
#   python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \
#       --dest inference/inf --weights models/notransfer_torch.pth
#
#   # same, plus the stage outputs used by src/validation/compare_stages_with_tf.py
#   python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \
#       --dest inference/inf --weights models/notransfer_torch.pth --stages_dir validation/torch_stages
#
#   # quick test on the first 5 tiles, masks only
#   python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \
#       --dest /tmp/inf_test --weights models/notransfer_torch.pth --limit 5 --no_images
