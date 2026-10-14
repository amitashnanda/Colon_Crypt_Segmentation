"""
Colon Crypt Segmentation - Nuclei Training Recipe, Stage 1 (Kaggle Data Science Bowl 2018)

This script is stage 1 of the nuclei training recipe this project started from: a one-class (nuclei)
Mask R-CNN on 512 x 512 random crops of the Kaggle DSB 2018 stage-1 training set (layout of
my_bowl_dataset.py), initialised from the COCO weights (heads re-initialised). All layers are trained to
epoch 30 (LR), 50 (LR/10) and 75 (LR/30), without augmentation; the validation set is the training set.
Python, NumPy and TensorFlow are seeded with 123. The weights are saved as logs/<run>/final.h5;
my_train_2.py and my_train_3.py continue from them.

Usage (TF environment, from the repository root):
    python src/training/my_train_1.py --train_dir dataset/kaggle-dsbowl-2018-dataset-fixes/stage1_train \\
        --coco models/mask_rcnn_coco.h5

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

from mrcnn import model as modellib
from mrcnn.config import Config
from my_bowl_dataset import BowlDataset


# --- Default Settings ---
SEED = 123
TRAIN_DIR = "dataset/kaggle-dsbowl-2018-dataset-fixes/stage1_train"
COCO_WEIGHTS = "mask_rcnn_coco.h5"   # in the working directory, as in the original run
# (https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5)


class BowlConfig(Config):
    """
    Training configuration of stage 1 (2 classes: background, nuclei; 512 x 512 random crops).
    """
    # Give the configuration a recognizable name
    NAME = "001"

    IMAGE_RESIZE_MODE = "crop"
    # Augmentation parameters
    ASPECT_RATIO = 1.3 # Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1.2 # Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.5 # Maximum zoom per image, note that this will be randomized
    IMAGE_MIN_SCALE = False # not used

    ROT_RANGE = 10.
    CHANNEL_SHIFT_RANGE = 15

    LEARNING_RATE = 0.001
    # Train on 4 GPUs with 1 image per GPU (batch size 4)
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + nuclei

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8 , 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 600

    STEPS_PER_EPOCH = 664//IMAGES_PER_GPU
    # the validation set is the training set, so validation is kept short
    VALIDATION_STEPS =1

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


def main():
    parser = argparse.ArgumentParser(
        description="Nuclei recipe, stage 1: train all layers from the COCO weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_dir", default=TRAIN_DIR, help="stage-1 training set (default: %(default)s)")
    parser.add_argument("--coco", default=COCO_WEIGHTS, help="COCO weights (default: %(default)s)")
    args = parser.parse_args()

    # --- 1. Seeds and configuration ---
    set_seeds(SEED)
    bowl_config = BowlConfig()
    bowl_config.display()

    # --- 2. Dataset (one folder per image); the validation set is the training set ---
    root_dir = os.getcwd()
    train_dir = os.path.join(root_dir, args.train_dir)
    print(train_dir)

    # Get train IDs
    train_ids = next(os.walk(train_dir))[1]
    train_ids = [os.path.join(train_dir, train_id) for train_id in train_ids]

    dataset_train = BowlDataset()
    dataset_train.load_bowl(train_ids)
    dataset_train.prepare()

    dataset_val = BowlDataset()
    dataset_val.load_bowl(train_ids)
    dataset_val.prepare()

    # --- 3. Model from the COCO weights, heads re-initialised ---
    model_dir = os.path.join(root_dir, "logs")
    coco_model_path = os.path.join(root_dir, args.coco)
    model = modellib.MaskRCNN(mode="training", config=bowl_config,
                              model_dir=model_dir)
    model.load_weights(coco_model_path, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])

    start_time = time.time()

    # --- 4. Training schedule, no augmentation ---
    augmentation=False

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers="all")

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE/10,
                epochs=50,
                augmentation=augmentation,
                layers="all")

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE/30,
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
#   python src/training/my_train_1.py --train_dir dataset/kaggle-dsbowl-2018-dataset-fixes/stage1_train \
#       --coco models/mask_rcnn_coco.h5
#   -> logs/001<timestamp>/final.h5 (pass it to my_train_2.py --init)
