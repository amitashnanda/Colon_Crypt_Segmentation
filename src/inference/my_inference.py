"""
Colon Crypt Segmentation - Step 1: Mask R-CNN Inference (TF/Keras)

This script runs the trained crypt/gland Mask R-CNN (Matterport Mask R-CNN in TF/Keras, mrcnn/) over a
tree of colour-normalised, Hough-aligned tiles. It produced the CDX2 results of the paper.

Each tile is padded to the aspect ratio 1.5 and resized to 1024 x 768 (imgaug), then detected with
BowlConfig (ResNet-50, minimum confidence 0.7, NMS 0.4, up to 512 instances). Python, NumPy and
TensorFlow are seeded with 123 before every tile. For every <src>/<rel>/<tile>.png it writes:
    <dest>/<rel>/predicted_images/<tile>.png     detections drawn on the tile (masks, boxes, scores)
    <dest>/<rel>/predicted_mask/<tile>_<j>.png   one 0/255 mask per crypt (class 1), j = detection index

A relative --model path is taken from the working directory, so run from the repository root.

Usage (TF environment, environment_cdx2_tf1_ref.yml):
    python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data/ \\
        --dest inference/inf/ --model models/notransfer.h5

Author: Amitash Nanda
Date: 2026-10-09
"""

# Make the TF Mask R-CNN package (mrcnn/ at the repository root) importable from any working directory
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import glob
import random
import time

# TensorFlow (and the Mask R-CNN library) are imported before imgaug and OpenCV, as in the original script
import numpy as np
import tensorflow as tf
import skimage.io
from skimage import img_as_ubyte

from mrcnn import model as modellib
from mrcnn.config import Config
from mrcnn.visualize import display_instances

import imgaug.augmenters as iaa
import matplotlib.pyplot as plt
import cv2 as cv


# --- Default Settings ---
SEED = 123                               # Python, NumPy and TF seed, re-applied before every tile
CLASS_NAMES = ['bg', 'crypt', 'gland']   # class ids 0, 1, 2 of the trained model
CRYPT_CLASS_ID = 1                       # only crypt masks are written to predicted_mask/
INPUT_ASPECT = 1.5                       # tiles are padded to width / height = 1.5 ...
INPUT_HEIGHT, INPUT_WIDTH = 768, 1024    # ... and resized to 1024 x 768 before detection


class BowlConfig(Config):
    """
    Inference configuration of the crypt/gland model (ResNet-50, 3 classes: bg, crypt, gland).

    The name comes from the nuclei (Kaggle Data Science Bowl) recipe this project started from. The
    training-only values are kept so that the printed configuration is the one of the paper run.
    """
    # Give the configuration a recognizable name
    NAME = "Inference"
    BACKBONE = "resnet50"
    IMAGE_RESIZE_MODE = "square"
    # No augmentation
    ZOOM = False
    ASPECT_RATIO = 1
    MIN_ENLARGE = 1
    IMAGE_MIN_SCALE = False  # not used

    # molded input: smallest side scaled to 768, padded to a 1024 x 1024 square
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    DETECTION_MAX_INSTANCES = 512
    DETECTION_NMS_THRESHOLD =  0.4
    DETECTION_MIN_CONFIDENCE = 0.7

    LEARNING_RATE = 0.0001

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 + 1  # background + crypt + gland

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 200

    USE_MINI_MASK = True


def set_seeds(seed):
    """
    Seed Python's random module, NumPy and TensorFlow (graph-level seed).

    Args:
        seed: seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)


def load_and_preprocess(image_path):
    """
    Load one tile and give it the model input format.

    Grey tiles get three channels, an alpha channel is dropped, then imgaug pads the tile to the aspect
    ratio 1.5 (centred) and resizes it to 1024 x 768 (bicubic).

    Args:
        image_path: tile file

    Returns:
        768 x 1024 x 3 uint8 RGB image
    """
    original_image = skimage.io.imread(image_path)

    # grey (single-channel) tiles -> 3 channels
    if len(original_image.shape) < 3:
        original_image = img_as_ubyte(original_image)
        original_image = np.expand_dims(original_image, 2)
        original_image = original_image[:, :, [0, 0, 0]]
    original_image = original_image[:, :, :3]

    augmentation = iaa.Sequential([
        iaa.PadToAspectRatio(INPUT_ASPECT, position="center"),
        iaa.Resize({"height": INPUT_HEIGHT, "width": INPUT_WIDTH}),
    ], random_order=False)
    return augmentation.augment_image(original_image)


def save_crypt_masks(r, image_id, mask_dir):
    """
    Write one 0/255 PNG per crypt detection.

    Args:
        r: result of model.detect for one tile (rois, masks, class_ids, scores)
        image_id: tile file name (<tile>.png)
        mask_dir: predicted_mask/ folder of the slide
    """
    pred_masks = r['masks']
    for j, class_id in enumerate(r['class_ids']):
        if class_id == CRYPT_CLASS_ID:
            mask_filename = image_id[:-len(".png")] + "_" + str(j) + ".png"
            mask_path = os.path.join(mask_dir, mask_filename)
            cv.imwrite(mask_path, pred_masks[:, :, j] * 255)


def main():
    parser = argparse.ArgumentParser(
        description="Mask R-CNN inference (TF/Keras) over a tree of colour-normalised, Hough-aligned tiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--src", help="tile root, e.g. dataset/Whole_Slides_Segments_Processed/CDX2_Data/",
                        type=str, required=True)
    parser.add_argument("--dest", help="output root (mirrors --src), e.g. inference/inf/", type=str, required=True)

    # Model
    parser.add_argument("--model", help="Keras weights, e.g. models/notransfer.h5 (relative to the working directory)",
                        type=str, required=True)
    args = parser.parse_args()

    # --- 1. Seeds and configuration ---
    set_seeds(SEED)
    inference_config = BowlConfig()
    inference_config.display()

    # --- 2. Model in inference mode with the trained weights ---
    model_dir = os.getcwd()
    model_path = args.model
    src_path = args.src
    dest_path = args.dest

    print("Loading weights from ", model_path)
    os.makedirs(dest_path, exist_ok=True)
    start_time = time.time()

    model = modellib.MaskRCNN(mode="inference",
                              config=inference_config,
                              model_dir=model_dir)
    model.load_weights(os.path.join(model_dir, model_path), by_name=True)

    # --- 3. All tiles under --src ---
    all_image_paths = glob.glob(os.path.join(src_path, "**/*.png"), recursive=True)
    n_images = len(all_image_paths)
    print(f"Found {n_images} images in {src_path}")

    # --- 4. Detect on every tile; write the drawing and the crypt masks, mirroring the folders ---
    for i, image_path in enumerate(all_image_paths):
        rel_path = os.path.relpath(image_path, src_path)
        rel_dir = os.path.dirname(rel_path)
        image_id = os.path.basename(image_path)

        dest_subdir = os.path.join(dest_path, rel_dir)
        predicted_mask_dir = os.path.join(dest_subdir, "predicted_mask")
        predicted_images_dir = os.path.join(dest_subdir, "predicted_images")
        os.makedirs(predicted_mask_dir, exist_ok=True)
        os.makedirs(predicted_images_dir, exist_ok=True)

        print(f'Start detect {i+1}/{n_images}  {rel_path}')

        # same seeds before every tile
        set_seeds(SEED)

        original_image = load_and_preprocess(image_path)
        results = model.detect([original_image], verbose=1)
        r = results[0]

        # predicted_images/<tile>.png (display_instances saves the figure), then free the figures
        display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                          CLASS_NAMES, r['scores'],
                          title=image_id, dest=predicted_images_dir + "/")
        plt.close("all")

        # predicted_mask/<tile>_<j>.png
        save_crypt_masks(r, image_id, predicted_mask_dir)

    end_time = time.time()
    elapsed_time = (end_time - start_time) / 3600
    print(f'Time required to process {n_images} images: {elapsed_time:.4f} hours')


if __name__ == "__main__":
    main()


# Example usage (from the repository root, TF environment):
#
#   # all tiles, as in inference/inf (Mouse-DSS included; the per-slide table leaves it out)
#   python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data/ \
#       --dest inference/inf/ --model models/notransfer.h5
#
#   # one slide
#   python src/inference/my_inference.py --src dataset/Whole_Slides_Segments_Processed/CDX2_Data/Boiling-5min/S.19.4704/ \
#       --dest inference_test/inf/Boiling-5min/S.19.4704/ --model models/notransfer.h5
