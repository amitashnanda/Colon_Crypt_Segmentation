"""
Colon Crypt Segmentation - Gland-Only Mask R-CNN (Earlier Experiment)

This script trains a Mask R-CNN for glands only (one class) on <dataset>/{train,valid} (layout of
my_gland_dataset.py), initialised from the COCO weights (all heads re-initialised). All layers are trained
to epoch 30 (LR), 50 (LR/10) and 75 (LR/30) with flips, 90-degree and small rotations and brightness
changes. Python, NumPy and TensorFlow are seeded with 123. The weights are saved as logs/<run>/final.h5.

Usage (TF environment, from the repository root; mask_rcnn_coco.h5 is in models/):
    python src/training/my_train_gland.py --dataset dataset/gland --coco models/mask_rcnn_coco.h5

Author: Amitash Nanda
Date: 2026-10-09
"""

# Make the TF Mask R-CNN package (mrcnn/ at the repository root) importable from any working directory
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import argparse
import random
import time

import numpy as np
import tensorflow as tf
from imgaug import augmenters as iaa

from mrcnn import model as modellib
from mrcnn.config import Config
from my_gland_dataset import GlandDataset


# --- Default Settings ---
SEED = 123
DATASET = "dataset/gland"            # with train/ and valid/
COCO_WEIGHTS = "mask_rcnn_coco.h5"   # in the working directory, as in the original run


class GlandConfig(Config):
    """
    Training configuration of the gland-only model (2 classes: background, gland; 512 x 512 random crops).
    """
    # Give the configuration a recognizable name
    NAME = "001"

    IMAGE_RESIZE_MODE = "crop"
    # Augmentation parameters
    ASPECT_RATIO = 1.3  # Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1.2  # Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.5  # Maximum zoom per image, note that this will be randomized
    IMAGE_MIN_SCALE = False  # not used

    ROT_RANGE = 10.
    CHANNEL_SHIFT_RANGE = 15

    LEARNING_RATE = 0.001
    # Train on 4 GPUs with 1 image per GPU (batch size 4)
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + gland

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = 664 // IMAGES_PER_GPU
    VALIDATION_STEPS = 1

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512


def set_seeds(seed):
    """
    Seed NumPy, TensorFlow (graph-level seed) and Python's random module.

    Args:
        seed: seed value
    """
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)


def load_datasets(root_dir, base_dir):
    """
    Load the training and validation sets.

    Args:
        root_dir: working directory
        base_dir: dataset folder with train/ and valid/ (relative to root_dir)

    Returns:
        Tuple of (training GlandDataset, validation GlandDataset), prepared
    """
    train_dir = os.path.join(root_dir, base_dir + '/train/Images')
    print(train_dir)

    # Get train IDs
    train_ids = next(os.walk(train_dir))[2]

    # Training dataset
    dataset_train = GlandDataset()
    dataset_train.load_bowl(train_ids, base_dir + '/train')
    dataset_train.prepare()

    # Validation dataset
    val_dir = os.path.join(root_dir, base_dir + '/valid/Images')
    valid_ids = next(os.walk(val_dir))[2]
    dataset_val = GlandDataset()
    dataset_val.load_bowl(valid_ids, base_dir + '/valid')
    dataset_val.prepare()
    return dataset_train, dataset_val


def main():
    parser = argparse.ArgumentParser(
        description="Train a gland-only Mask R-CNN from the COCO weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", default=DATASET, help="dataset folder with train/ and valid/ (default: %(default)s)")
    parser.add_argument("--coco", default=COCO_WEIGHTS, help="COCO weights (default: %(default)s)")
    args = parser.parse_args()

    # --- 1. Seeds and configuration ---
    set_seeds(SEED)
    bowl_config = GlandConfig()
    bowl_config.display()

    # --- 2. Datasets ---
    root_dir = os.getcwd()
    dataset_train, dataset_val = load_datasets(root_dir, args.dataset)

    # --- 3. Model from the COCO weights, heads re-initialised ---
    model_dir = os.path.join(root_dir, "logs")
    coco_model_path = os.path.join(root_dir, args.coco)
    model = modellib.MaskRCNN(mode="training", config=bowl_config,
                              model_dir=model_dir)
    model.load_weights(coco_model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    start_time = time.time()

    # --- 4. Training schedule (http://imgaug.readthedocs.io/en/latest/source/augmenters.html) ---
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.OneOf([iaa.Affine(rotate=0),
                   iaa.Affine(rotate=90),
                   iaa.Affine(rotate=180),
                   iaa.Affine(rotate=270)]),
        iaa.Sometimes(0.5,iaa.Affine(rotate=(-10,10))),
        iaa.Add((-15, 15), per_channel=1)
    ])

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers="all")

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE / 10,
                epochs=50,
                augmentation=augmentation,
                layers="all")

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE / 30,
                epochs=75,
                augmentation=augmentation,
                layers="all")

    # --- 5. Final weights ---
    elapsed_time = (time.time() - start_time) / 3600
    print(f"training time: {elapsed_time:.2f} hours")
    print(model.log_dir)
    model_path = os.path.join(model.log_dir, 'final.h5')
    model.keras_model.save_weights(model_path)


if __name__ == "__main__":
    main()


# Example usage (TF environment, from the repository root):
#
#   python src/training/my_train_gland.py --dataset dataset/gland --coco models/mask_rcnn_coco.h5
