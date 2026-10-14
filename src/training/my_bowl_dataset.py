"""
Colon Crypt Segmentation - Nuclei Dataset (Kaggle Data Science Bowl 2018 Layout)

This module holds the data loader of the nuclei training recipe this project started from
(my_train_1.py, my_train_2.py, my_train_3.py). One class (nuclei); the Kaggle DSB 2018 layout:
    <id>/images/<id>.png     image
    <id>/masks/*.png         one mask per nucleus

Usage (imported; see my_train_1.py):
    from my_bowl_dataset import BowlDataset
    dataset = BowlDataset()
    dataset.load_bowl([os.path.join(train_dir, i) for i in os.listdir(train_dir)])
    dataset.prepare()

Author: Amitash Nanda
Date: 2026-10-09
"""

import os

import numpy as np
import skimage.io

from mrcnn.utils import Dataset


class BowlDataset(Dataset):
    """Nuclei dataset in the Kaggle DSB 2018 layout (<id>/images/<id>.png, <id>/masks/*.png)."""

    def load_bowl(self, folderpaths):
        """
        Register the images.

        Args:
            folderpaths: one folder per image (<id>/)
        """
        # Add classes
        self.add_class("bowl", 1, "nuclei")

        # Add images
        for i in range(len(folderpaths)):
            self.add_image("bowl", image_id=i, path=folderpaths[i])

    def load_image(self, image_id):
        """
        Load one image as RGB (an alpha channel is dropped).

        Args:
            image_id: internal image id

        Returns:
            H x W x 3 image
        """
        info = self.image_info[image_id]
        image_path = info['path']
        image_path = os.path.join(image_path, 'images', '{}.png'.format(os.path.basename(image_path)))
        image = skimage.io.imread(image_path)
        image = image[:,:,:3]
        return image

    def image_reference(self, image_id):
        """Return the image info of the image."""
        info = self.image_info[image_id]
        if info["source"] == "bowl":
            return info["bowl"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """
        Load the nucleus masks of one image (<id>/masks/*.png).

        Args:
            image_id: internal image id

        Returns:
            Tuple of (H x W x N mask in {0, 1}, int32 class ids, all 1)
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_paths = os.path.join(image_path, 'masks', '*.png')
        mask = skimage.io.imread_collection(mask_paths).concatenate()
        mask = np.rollaxis(mask,0,3)
        mask = np.clip(mask,0,1)
        class_ids = np.array([1]* mask.shape[2])
        return mask, class_ids.astype(np.int32)
