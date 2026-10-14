"""
Colon Crypt Segmentation - Crypt/Gland Training Configuration and Dataset

This module holds the training configuration (CryptsConfig) and the data loader (CryptsDataset) of the
crypt/gland Mask R-CNN of the paper, trained by my_train_crypt.py. Layout of a dataset folder:
    <folder>/Images/<name>.png                  tile
    <folder>/Annotation/<name>_<k>.png          one mask per crypt (class 1)
    <folder>/Annotation_gland/<name>_<k>.png    one mask per gland (class 2)

Usage (imported; see my_train_crypt.py):
    from my_crypts_dataset import CryptsConfig, CryptsDataset
    dataset = CryptsDataset()
    dataset.load_bowl(os.listdir("dataset/Normalized_Images/train/Images"), "dataset/Normalized_Images/train")
    dataset.prepare()

Author: Amitash Nanda
Date: 2026-10-09
"""

import os

import numpy as np
import skimage.io

from mrcnn.config import Config
from mrcnn.utils import Dataset


class CryptsConfig(Config):
    """
    Training configuration of the crypt/gland model (ResNet-50 FPN, 3 classes: background, crypt, gland).

    Derives from the base Config class of mrcnn/config.py; the tiles are molded to 1024 x 1024 squares
    (smallest side 768) and trained on 4 GPUs with one image each.
    """
    # Give the configuration a recognizable name
    NAME = "crypt"
    BACKBONE = "resnet50"
    WEIGHT_DECAY = 0.000001

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 768
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = False  # not used

    IMAGE_CHANNEL_COUNT = 3
    # Augmentation parameters
    ASPECT_RATIO = 1  # Maximum aspect ratio modification when scaling
    MIN_ENLARGE = 1  # Minimum enlarging of images, note that this will be randomized
    ZOOM = 1.5  # Maximum zoom per image, note that this will be randomized

    ROT_RANGE = 10.
    CHANNEL_SHIFT_RANGE = 15

    LEARNING_RATE = 0.001
    # Train on 4 GPUs with 1 image per GPU (batch size 4)
    GPU_COUNT = 4
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 2   # background + u-shape + circular glands

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 512

    STEPS_PER_EPOCH = 600 // IMAGES_PER_GPU
    VALIDATION_STEPS = 50

    USE_MINI_MASK = True

    MAX_GT_INSTANCES = 256

    DETECTION_MAX_INSTANCES = 512


class CryptsDataset(Dataset):
    """
    Crypt/gland dataset: tiles in <folder>/Images, one mask file per instance in <folder>/Annotation
    (crypts, class 1) and <folder>/Annotation_gland (glands, class 2).
    """

    def load_bowl(self, folderpaths, folderpath):
        """
        Register the images of one dataset folder.

        Args:
            folderpaths: image file names (in <folderpath>/Images)
            folderpath: dataset folder
        """
        # Add classes
        self.add_class("bowl", 1, "crypt")
        self.add_class("bowl", 2, "gland")

        self.folder_path = folderpath
        # Add images
        for i in range(len(folderpaths)):
            self.add_image("bowl", image_id=i, path=folderpaths[i])

    def load_image(self, image_id):
        """
        Load one tile as RGB (an alpha channel is dropped).

        Args:
            image_id: internal image id

        Returns:
            H x W x 3 image
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image_path = os.path.join(self.folder_path, 'Images', image_path)
        image = skimage.io.imread(image_path)
        image = image[:,:,:3]
        return image

    def image_reference(self, image_id):
        """Return the image info of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bowl":
            return info
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """
        Load the instance masks of one tile: crypts first (class 1), then glands (class 2).

        Args:
            image_id: internal image id

        Returns:
            Tuple of (H x W x N mask in {0, 1}, int32 class ids)
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_paths = os.path.join(self.folder_path, 'Annotation', os.path.splitext(image_path)[0])+ '_*'
        gland_path = os.path.join(self.folder_path, 'Annotation_gland', os.path.splitext(image_path)[0])+ '_*'
        class_ids = np.array([])
        mask=np.array([])
        try:
            mask = skimage.io.imread_collection(mask_paths).concatenate()
            class_ids = np.array([1] * mask.shape[0])
        except:
            pass
        try:
            gland_mask = skimage.io.imread_collection(gland_path).concatenate()
            class_ids = np.concatenate((class_ids, [2] * gland_mask.shape[0]))
            # with NumPy 1.16 (TF environment) this comparison is True for a non-empty mask and
            # False for the empty start value
            if mask != np.array([]):
                mask = np.concatenate((mask, gland_mask))
            else:
                mask = gland_mask
        except:
            class_ids = np.array([1] * mask.shape[0])
        mask = np.rollaxis(mask,0,3)
        mask = np.clip(mask,0,1)
        return mask, class_ids.astype(np.int32)
