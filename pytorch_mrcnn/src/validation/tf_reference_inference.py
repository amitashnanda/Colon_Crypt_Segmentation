"""
Colon Crypt Segmentation (PyTorch) - Validation: TF Reference Inference with Stage Outputs

This script runs exactly the TF workflow of src/inference/my_inference.py (same BowlConfig, same
imgaug preprocessing, same model.detect internals, same mask files) with the TF package mrcnn/ of the
repository root, and additionally saves per tile: the proposals (ROI layer), the classifier
probabilities and box deltas, the raw detections and the raw 28 x 28 mask-head output. With these,
every PyTorch stage can be checked against TF (src/validation/compare_stages_with_tf.py).

Outputs:
    <dest>/inf/<protocol>/<slide>/predicted_mask/<tile>_<j>.png
    <dest>/stages/<protocol>/<slide>/<tile>.npz

Usage (TF1 environment cdx2_tf1_ref, CPU, about 1 s per tile on a Perlmutter node; from pytorch_mrcnn/):
    python src/validation/tf_reference_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \\
        --dest validation/tf_reference --model models/notransfer.h5 --skip Mouse-DSS

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # TF1 GPU builds do not support A100; use CPU

import numpy as np

# TF Mask R-CNN package (mrcnn/ at the repository root)
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))  # repository root, holds the TF package mrcnn/
sys.path.insert(0, REPO)

import tensorflow as tf  # noqa: E402
import keras.backend as K  # noqa: E402
from keras.models import Model as KModel  # noqa: E402
import skimage.io  # noqa: E402
from skimage import img_as_ubyte  # noqa: E402
import imgaug.augmenters as iaa  # noqa: E402
import cv2  # noqa: E402

from mrcnn import model as modellib  # noqa: E402  (TF mrcnn/model.py)
from mrcnn.config import Config  # noqa: E402  (TF mrcnn/config.py)


class BowlConfig(Config):
    """Inference configuration, identical to BowlConfig in the TF script src/inference/my_inference.py."""
    NAME = "Inference"
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD = 0.4
    DETECTION_MIN_CONFIDENCE = 0.7
    LEARNING_RATE = 0.0001
    NUM_CLASSES = 1 + 1 + 1
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)
    TRAIN_ROIS_PER_IMAGE = 200
    USE_MINI_MASK = True


def load_and_preprocess(path):
    """
    Load one tile with the steps of the TF script src/inference/my_inference.py.

    Args:
        path: tile file

    Returns:
        768 x 1024 x 3 uint8 RGB image
    """
    img = skimage.io.imread(path)
    if len(img.shape) < 3:
        img = img_as_ubyte(img)
        img = np.expand_dims(img, 2)
        img = img[:, :, [0, 0, 0]]
    img = img[:, :, :3]
    aug = iaa.Sequential([iaa.PadToAspectRatio(1.5, position="center"),
                          iaa.Resize({"height": 768, "width": 1024})], random_order=False)
    return aug.augment_image(img)


def build_probe(model):
    """
    Keras model with the same inputs as the inference model that also outputs the intermediate stages.

    Args:
        model: modellib.MaskRCNN in inference mode with the weights loaded

    Returns:
        Keras model with outputs [detections, masks, ROIs, class probabilities, box deltas]
    """
    km = model.keras_model
    outs = [km.get_layer("mrcnn_detection").output, km.get_layer("mrcnn_mask").output,
            km.get_layer("ROI").output, km.get_layer("mrcnn_class").output, km.get_layer("mrcnn_bbox").output]
    return KModel(inputs=km.inputs, outputs=outs)


def main():
    parser = argparse.ArgumentParser(
        description="TF reference inference that also saves the stage outputs of every tile.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", required=True, help="CDX2_Data root (PNG tiles in <protocol>/<slide>/)")
    parser.add_argument("--dest", required=True, help="output root (gets inf/ and stages/)")
    parser.add_argument("--model", required=True, help="Keras weights, e.g. models/notransfer.h5")

    # Options
    parser.add_argument("--skip", nargs="*", default=["Mouse-DSS"], help="slide folders to skip")
    parser.add_argument("--threads", type=int, default=64, help="TF intra-op threads")
    parser.add_argument("--limit", type=int, default=0, help="process only the first N tiles (testing)")
    args = parser.parse_args()

    # --- 1. Model and probe ---
    K.set_session(tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=args.threads,
                                                   inter_op_parallelism_threads=2)))
    cfg = BowlConfig()
    model = modellib.MaskRCNN(mode="inference", config=cfg, model_dir=args.dest)
    model.load_weights(os.path.abspath(args.model), by_name=True)
    probe = build_probe(model)

    # --- 2. Tiles ---
    paths = sorted(glob.glob(os.path.join(args.src, "**", "*.png"), recursive=True))
    paths = [p for p in paths if not any(f"{os.sep}{s}{os.sep}" in p for s in args.skip)]
    if args.limit:
        paths = paths[:args.limit]
    print(f"[info] {len(paths)} tiles", flush=True)

    # --- 3. Detect on every tile; write the crypt masks and the stage outputs ---
    t0 = time.time()
    for i, p in enumerate(paths):
        rel_dir = os.path.dirname(os.path.relpath(p, args.src))
        stem = os.path.splitext(os.path.basename(p))[0]
        mask_dir = os.path.join(args.dest, "inf", rel_dir, "predicted_mask")
        npz_dir = os.path.join(args.dest, "stages", rel_dir)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(npz_dir, exist_ok=True)

        image = load_and_preprocess(p)
        # exactly what MaskRCNN.detect() does
        molded, metas, windows = model.mold_inputs([image])
        anchors = model.get_anchors(molded[0].shape)
        anchors = np.broadcast_to(anchors, (cfg.BATCH_SIZE,) + anchors.shape)
        det, mmask, rois, mclass, mbbox = probe.predict([molded, metas, anchors], verbose=0)
        boxes, class_ids, scores, masks = model.unmold_detections(det[0], mmask[0], image.shape, molded[0].shape, windows[0])
        # mask files exactly as my_inference.py names them
        for j, cid in enumerate(class_ids):
            if cid == 1:
                cv2.imwrite(os.path.join(mask_dir, f"{stem}_{j}.png"), (masks[:, :, j] * 255).astype(np.uint8))
        # stage outputs (detections and masks without the zero padding)
        n = int(np.argmax(det[0][:, 4] == 0)) if np.any(det[0][:, 4] == 0) else det.shape[1]
        np.savez_compressed(os.path.join(npz_dir, f"{stem}.npz"),
                            rois=rois[0], mrcnn_class=mclass[0], mrcnn_bbox=mbbox[0],
                            detections=det[0][:n], mrcnn_mask=mmask[0][:n],
                            boxes=boxes, class_ids=class_ids, scores=scores,
                            masks=np.packbits(masks.astype(bool), axis=-1) if masks.size else np.zeros((0,), np.uint8),
                            mask_shape=np.array(masks.shape))
        if (i + 1) % 25 == 0 or i == 0:
            el = time.time() - t0
            print(f"  [{i + 1}/{len(paths)}] {os.path.relpath(p, args.src)}  {len(class_ids)} det  "
                  f"({el / (i + 1):.2f} s/tile)", flush=True)
    print(f"[done] {len(paths)} tiles in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()


# Example usage (TF environment, from pytorch_mrcnn/):
#
#   python src/validation/tf_reference_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \
#       --dest validation/tf_reference --model models/notransfer.h5 --skip Mouse-DSS
#   bash src/validation/run_tf_reference_downstream.sh        # split, U-shape, colour, table on that output
#
#   # quick test on 10 tiles
#   python src/validation/tf_reference_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data \
#       --dest /tmp/tf_ref_test --model models/notransfer.h5 --limit 10
