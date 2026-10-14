"""
Colon Crypt Segmentation - Nuclei Training Recipe, Stage 3 (Kaggle Data Science Bowl 2018)

This script is stage 3 of the nuclei training recipe: it resumes from the stage-2 weights
(logs/00220190520T1522/final.h5 by default) with the same size-balanced sampling as my_train_2.py and
stronger built-in scaling/rotation/colour-shift parameters, and trains all layers for 5 epochs at LR/30,
without imgaug augmentation; the validation set is the training set. Python, NumPy and TensorFlow are
seeded with 123. The weights are saved as logs/<run>/final.h5.

Usage (TF environment, from the repository root):
    python src/training/my_train_3.py --init logs/00220190520T1522/final.h5

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
import skimage.io

from mrcnn import model as modellib
from mrcnn.config import Config
from my_bowl_dataset import BowlDataset


# --- Default Settings ---
SEED = 123
TRAIN_DIR = "dataset/kaggle-dsbowl-2018-dataset-fixes/stage1_train"
INIT_WEIGHTS = "logs/00220190520T1522/final.h5"   # stage-2 weights


class BowlConfig(Config):
    """
    Training configuration of stage 3 (2 classes: background, nuclei; 512 x 512 random crops; stronger
    scaling, rotation and channel shift than stages 1 and 2).
    """
    # Give the configuration a recognizable name
    NAME = "003"

    IMAGE_RESIZE_MODE = "crop"
    # Augmentation parameters
    ASPECT_RATIO = 1.5 # Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1.5 # Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.7 # Maximum zoom per image, note that this will be randomized
    IMAGE_MIN_SCALE = False # not used

    ROT_RANGE = 22.5
    CHANNEL_SHIFT_RANGE = 20

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
    VALIDATION_STEPS = 2//IMAGES_PER_GPU

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


def balance_by_image_size(train_dir, train_ids):
    """
    Repeat the images of each image size so that every size is about equally represented: the images of
    a size with `count` images are repeated int(max count / count) times (as in my_train_2.py).

    Args:
        train_dir: training set folder
        train_ids: image ids (folder names)

    Returns:
        List of image folders, with repetitions
    """
    # image size of every image
    train_paths = [os.path.join(train_dir, train_id,'images','{}.png'.format(train_id)) for train_id in train_ids]
    img_sizes=[]
    for i in train_paths:
        image = skimage.io.imread(i)
        img_sizes.append(image.shape)

    # number of images per size
    counts = []
    for i in set(img_sizes):
        count = 0
        for j in np.arange(len(img_sizes)):
            if img_sizes[j] == i:
                count = count +1
        print(i,count)
        counts.append(count)

    # repeat the images of each size
    long_train_ids = []
    replic_n = max(counts)
    for i in set(img_sizes):
        train_ids_size = []
        count = 0
        for j in np.arange(len(img_sizes)):
            if img_sizes[j] == i:
                train_ids_size.append(train_ids[j])
                count = count +1
        repeticiones = int(replic_n/count)
        print(i,count,repeticiones)
        train_ids_size = np.repeat(train_ids_size,repeticiones)
        long_train_ids.extend(train_ids_size.tolist())

    return [os.path.join(train_dir, train_id) for train_id in long_train_ids]


def main():
    parser = argparse.ArgumentParser(
        description="Nuclei recipe, stage 3: size-balanced fine-tuning from the stage-2 weights.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--train_dir", default=TRAIN_DIR, help="stage-1 training set (default: %(default)s)")
    parser.add_argument("--init", default=INIT_WEIGHTS, help="stage-2 weights (default: %(default)s)")
    args = parser.parse_args()

    # --- 1. Seeds and configuration ---
    set_seeds(SEED)
    bowl_config = BowlConfig()
    bowl_config.display()

    # --- 2. Size-balanced dataset; the validation set is the training set ---
    root_dir = os.getcwd()
    train_dir = os.path.join(root_dir, args.train_dir)

    # Get train IDs
    train_ids = next(os.walk(train_dir))[1]
    train_ids = balance_by_image_size(train_dir, train_ids)

    dataset_train = BowlDataset()
    dataset_train.load_bowl(train_ids)
    dataset_train.prepare()

    dataset_val = BowlDataset()
    dataset_val.load_bowl(train_ids)
    dataset_val.prepare()

    # --- 3. Model from the stage-2 weights ---
    model_dir = os.path.join(root_dir, "logs")
    model = modellib.MaskRCNN(mode="training", config=bowl_config,
                              model_dir=model_dir)
    model.load_weights(os.path.join(root_dir, args.init), by_name=True)

    start_time = time.time()

    # --- 4. Training, no imgaug augmentation ---
    augmentation=False

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE/30,
                epochs=5,
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
#   python src/training/my_train_3.py --init logs/00220190520T1522/final.h5
#   -> logs/003<timestamp>/final.h5
