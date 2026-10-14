"""
Colon Crypt Segmentation - Train the Crypt/Gland Mask R-CNN (TF/Keras)

This script trains the crypt/gland Mask R-CNN used for the CDX2 results (ResNet-50 FPN; classes
background, crypt, gland; configuration CryptsConfig in my_crypts_dataset.py).

Data: <dataset>/{train,valid}/Images/<name>.png with one mask per instance in Annotation/<name>_<k>.png
(crypts) and Annotation_gland/<name>_<k>.png (glands). Schedule:
    1. heads only, to epoch 1, with light imgaug augmentation (flips, rotation, crop/pad, noise)
    2. ResNet stage 4 and up, to epoch 20, no augmentation
    3. all layers, to epoch 30, with strong imgaug augmentation
Python, NumPy and TensorFlow are seeded with 123. Checkpoints go to logs/crypt<timestamp>/ and the final
weights to logs/crypt<timestamp>/<dest>. Paths are relative to the working directory.

Usage (TF environment, from the repository root):
    python src/training/my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5
    python src/training/my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5 --model models/notransfer.h5

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
from my_crypts_dataset import CryptsDataset, CryptsConfig


# --- Default Settings ---
SEED = 123


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
    Load the training and validation sets of a dataset folder.

    Args:
        root_dir: working directory
        base_dir: dataset folder with train/ and valid/

    Returns:
        Tuple of (training CryptsDataset, validation CryptsDataset), prepared
    """
    train_dir = os.path.join(root_dir, base_dir + '/train/Images')
    print(train_dir)

    # Get train IDs
    train_ids = next(os.walk(train_dir))[2]

    # Training dataset
    dataset_train = CryptsDataset()
    dataset_train.load_bowl(train_ids, base_dir + '/train')
    dataset_train.prepare()

    # Validation dataset
    val_dir = os.path.join(root_dir, base_dir + '/valid/Images/')
    valid_ids = next(os.walk(val_dir))[2]
    dataset_val = CryptsDataset()
    dataset_val.load_bowl(valid_ids, base_dir + '/valid')
    dataset_val.prepare()
    return dataset_train, dataset_val


def _load_augmentation_light():
    """
    Light augmentation of stage 1 (http://imgaug.readthedocs.io/en/latest/source/augmenters.html).

    Returns:
        imgaug pipeline: flips, rotation, crop/pad and noise, in random order
    """
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Sometimes(0.5,iaa.Affine(rotate=(-180,180))),
        iaa.Sometimes(0.5,iaa.CropAndPad(percent=(-0.25, 0.25))),
        iaa.Sometimes(0.5, iaa.AddElementwise((-10, 10), per_channel=0.5)),
    ], random_order=True)


def _load_augmentation_aug_all():
    """
    Strong augmentation of stage 3.

    Returns:
        imgaug pipeline: flips, crop/pad and affine transforms, then 0 to 5 of blur, sharpen, emboss, edges,
        noise, dropout, colour, contrast, elastic, piecewise-affine and perspective augmenters
    """

    def sometimes(aug):
        return iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.Flipud(0.5),  # vertically flip 50% of all images
            # crop images by -5% to 10% of their height/width
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode='constant',
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                # translate by -20 to +20 percent (per axis)
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                rotate=(-45, 45),  # rotate by -45 to +45 degrees
                shear=(-16, 16),  # shear by -16 to +16 degrees
                # use nearest neighbour or bilinear interpolation (fast)
                order=[0, 1],
                # if mode is constant, use a cval between 0 and 255
                cval=(0, 255),
                # use any of scikit-image's warping modes
                # (see 2nd image from the top for examples)
                mode='constant'
            )),
            # execute 0 to 5 of the following (less important) augmenters per
            # image don't execute all of them, as that would often be way too
            # strong
            iaa.SomeOf((0, 5),
                       [
                # convert images into their superpixel representation
                sometimes(iaa.Superpixels(
                    p_replace=(0, 1.0), n_segments=(20, 200))),
                iaa.OneOf([
                    # blur images with a sigma between 0 and 3.0
                    iaa.GaussianBlur((0, 3.0)),
                    # blur image using local means with kernel sizes
                    # between 2 and 7
                    iaa.AverageBlur(k=(2, 7)),
                    # blur image using local medians with kernel sizes
                    # between 3 and 11
                    iaa.MedianBlur(k=(3, 11)),
                ]),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(
                            0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(
                    0, 2.0)),  # emboss images
                # search either for all edges or for directed edges,
                # blend the result with the original image using a blobby mask
                iaa.SimplexNoiseAlpha(iaa.OneOf([
                    iaa.EdgeDetect(alpha=(0.5, 1.0)),
                    iaa.DirectedEdgeDetect(
                        alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                ])),
                # add gaussian noise to images
                iaa.AdditiveGaussianNoise(loc=0, scale=(
                    0.0, 0.05*255), per_channel=0.5),
                iaa.OneOf([
                    # randomly remove up to 10% of the pixels
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(
                        0.02, 0.05), per_channel=0.2),
                ]),
                # invert color channels
                iaa.Invert(0.05, per_channel=True),
                # change brightness of images (by -10 to 10 of original value)
                iaa.Add((-10, 10), per_channel=0.5),
                # change hue and saturation
                iaa.AddToHueAndSaturation((-20, 20)),
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply(
                                (0.5, 1.5), per_channel=0.5),
                    iaa.FrequencyNoiseAlpha(
                        exponent=(-4, 0),
                        first=iaa.Multiply(
                            (0.5, 1.5), per_channel=True),
                        second=iaa.ContrastNormalization(
                            (0.5, 2.0))
                    )
                ]),
                # improve or worsen the contrast
                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                iaa.Grayscale(alpha=(0.0, 1.0)),
                # move pixels locally around (with random strengths)
                sometimes(iaa.ElasticTransformation(
                    alpha=(0.5, 3.5), sigma=0.25)),
                # sometimes move parts of the image around
                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))),
                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
            ],
                random_order=True
            )
        ],
        random_order=True
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train the crypt/gland Mask R-CNN (heads -> stage 4+ -> all layers).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data and output
    parser.add_argument("--dataset", help="path to the dataset, Exp: dataset/Normalized_Images", type=str, required=True)
    parser.add_argument("--dest", help="name of the output model, Exp:final.h5", type=str, required=True)

    # Initial weights
    parser.add_argument("--model", help="path to the model, Exp: logs/no_transfer/mask_rcnn_crypt_0060.h5", type=str, required=False)
    args = parser.parse_args()

    # --- 1. Seeds and configuration ---
    set_seeds(SEED)
    bowl_config = CryptsConfig()
    bowl_config.display()

    # --- 2. Datasets ---
    root_dir = os.getcwd()
    dataset_train, dataset_val = load_datasets(root_dir, args.dataset)

    # --- 3. Model (logs and checkpoints in logs/), optional initial weights ---
    model_dir = os.path.join(root_dir, "logs")
    model = modellib.MaskRCNN(mode="training", config=bowl_config,
                              model_dir=model_dir)
    if args.model:
        model.load_weights(args.model, by_name=True)

    start_time = time.time()

    # --- 4. Training schedule ---
    augmentation = _load_augmentation_light()
    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=1,
                augmentation=augmentation,
                layers="heads")

    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=20,
                layers="4+")

    augmentation = _load_augmentation_aug_all()
    model.train(dataset_train, dataset_val,
                learning_rate=bowl_config.LEARNING_RATE,
                epochs=30,
                augmentation=augmentation,
                layers="all")

    # --- 5. Final weights ---
    elapsed_time = (time.time() - start_time) / 3600
    print(f"training time: {elapsed_time:.2f} hours")
    print(model.log_dir)
    model_path = os.path.join(model.log_dir, args.dest)
    model.keras_model.save_weights(model_path)


if __name__ == "__main__":
    main()


# Example usage (TF environment, from the repository root):
#
#   # from randomly initialised weights (no --model)
#   python src/training/my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5
#
#   # continue from existing weights
#   python src/training/my_train_crypt.py --dataset dataset/Normalized_Images --dest final.h5 --model models/notransfer.h5
