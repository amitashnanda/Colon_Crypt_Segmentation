"""
Colon Crypt Segmentation - Offline Augmentation of a Training Set

This script augments the large images of a training set once, offline, with imgaug: images whose height
+ width is at least 4000 pixels are flipped left-right (p = 0.5), scaled by 1 to 1.5, padded to
2000 x 1500 and cropped to 1500 x 1000; smaller images are skipped. The crypt (Annotation/) and gland
(Annotation_gland/) masks get the same transform (only mask-safe augmenters are applied to them).

    dataset/<read_dir>/Images/<id>.png, Annotation/<id>_*.png, Annotation_gland/<id>_*.png
        -> dataset/aug_images/<read_dir>/Images/<id><aug_string>.png,
           Annotation/<id><aug_string>_<i>.png, Annotation_gland/<id><aug_string>_<i>.png

The augmentation is random (imgaug, unseeded), so every run gives new images.

Usage (TF environment, from the repository root):
    python src/preprocess/img_aug.py --read_dir Mix_crypts/train/ --aug_string orginin

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os

import numpy as np
import imgaug
import imgaug.augmenters as iaa
import skimage.io


# --- Default Settings ---
# Augmenters that are safe to apply to masks
# Some, such as Affine, have settings that make them unsafe, so always
# test your augmentation on masks
MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                   "Fliplr", "Flipud", "CropAndPad",
                   "Affine", "PiecewiseAffine"]
READ_DIR = "Mix_crypts/train/"   # training set under dataset/
AUG_STRING = "orginin"           # suffix of the augmented file names
MIN_SIZE_SUM = 4000              # only images with height + width >= this are augmented


def hook(images, augmenter, parents, default):
    """Determines which augmenters to apply to masks."""
    return augmenter.__class__.__name__ in MASK_AUGMENTERS


def load_mask(folder_path, image_id):
    """
    Load the instance masks of one training image (crypts, then glands).

    Args:
        folder_path: training set folder with Annotation/ and Annotation_gland/
        image_id: image file name

    Returns:
        Tuple of (H x W x N mask in {0, 1}, int32 class ids: 1 crypt, 2 gland)
    """
    print(image_id)
    mask_paths = os.path.join(folder_path, 'Annotation', os.path.splitext(image_id)[0]) + '_*'
    gland_path = os.path.join(folder_path, 'Annotation_gland', os.path.splitext(image_id)[0]) + '_*'

    mask = skimage.io.imread_collection(mask_paths).concatenate()
    try:
        gland_mask = skimage.io.imread_collection(gland_path).concatenate()
        class_ids = np.array([1]* mask.shape[0] + [2] * gland_mask.shape[0])
        mask = np.concatenate((mask, gland_mask))
    except:
        class_ids = np.array([1] * mask.shape[0])
    mask = np.rollaxis(mask,0,3)
    mask = np.clip(mask,0,1)
    return mask, class_ids.astype(np.int32)


def build_augmentation():
    """
    Returns:
        imgaug pipeline for the large images (flip, scale, pad, crop)
    """
    return iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Affine(scale=(1, 1.5)),
        iaa.PadToFixedSize(width=2000, height=1500),
        iaa.CropToFixedSize(width=1500, height=1000),
    ], random_order=False)


def augment_training_set(read_dir, aug_string):
    """
    Augment every large image of dataset/<read_dir> and its masks, writing to dataset/aug_images/<read_dir>.

    Args:
        read_dir: training set under dataset/ (with Images/, Annotation/, Annotation_gland/), ending in /
        aug_string: suffix of the augmented file names
    """
    root_dir = os.getcwd()

    ## Change this dir to the stage 1 training data
    train_dir = os.path.join(root_dir, 'dataset/' + read_dir)

    # Get train IDs
    train_ids = next(os.walk(train_dir + '/Images'))[2]

    for sub in ("Images/", "Annotation/", "Annotation_gland/"):
        os.makedirs("dataset/aug_images/" + read_dir + sub, exist_ok=True)

    for train_id in train_ids:
        image_path = os.path.join(train_dir, 'Images', train_id)
        image = skimage.io.imread(image_path)
        image = image[:, :, :3]

        mask, class_ids = load_mask(train_dir, train_id)
        mask_shape = mask.shape

        # only the large images are augmented
        if image.shape[0] + image.shape[1] < MIN_SIZE_SUM:
            continue
        augmentation = build_augmentation()

        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))

        skimage.io.imsave("dataset/aug_images/" + read_dir + "Images/" + os.path.splitext(train_id)[0] + aug_string + '.png', image)
        for i in range(mask_shape[2]):
            if class_ids[i] == 1:
                skimage.io.imsave("dataset/aug_images/" + read_dir + "Annotation/" + os.path.splitext(train_id)[0] + aug_string + '_' + str(i) + '.png', mask[:, :, i]*255)
            else:
                skimage.io.imsave("dataset/aug_images/" + read_dir + "Annotation_gland/" + os.path.splitext(train_id)[0]+ aug_string  + '_' + str(i) + '.png',
                                  mask[:, :, i] * 255)


def main():
    parser = argparse.ArgumentParser(
        description="Offline imgaug augmentation of the large images of a training set and their masks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--read_dir", default=READ_DIR,
                        help="training set under dataset/, ending in / (default: %(default)s)")
    parser.add_argument("--aug_string", default=AUG_STRING,
                        help="suffix of the augmented file names (default: %(default)s)")
    args = parser.parse_args()

    augment_training_set(args.read_dir, args.aug_string)


if __name__ == "__main__":
    main()


# Example usage (from the repository root):
#
#   python src/preprocess/img_aug.py --read_dir Mix_crypts/train/ --aug_string orginin
#   -> dataset/aug_images/Mix_crypts/train/{Images,Annotation,Annotation_gland}/
