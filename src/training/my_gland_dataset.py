"""
Colon Crypt Segmentation - Gland Dataset (Gland-Only Experiment)

This module holds the data loader of the earlier gland-only experiment (my_train_gland.py): one class
(gland), tiles in <folder>/Images/<name>.png and the masks of each tile in <folder>/Annotation/.

Usage (imported; see my_train_gland.py):
    from my_gland_dataset import GlandDataset
    dataset = GlandDataset()
    dataset.load_bowl(os.listdir("dataset/gland/train/Images"), "dataset/gland/train")
    dataset.prepare()

Author: Amitash Nanda
Date: 2026-10-09
"""

import os

import numpy as np
import skimage.io

from mrcnn.utils import Dataset


class GlandDataset(Dataset):
    """Gland dataset: tiles in <folder>/Images, masks in <folder>/Annotation (one class, gland)."""

    def load_bowl(self, folderpaths, folderpath):
        """
        Register the images of one dataset folder.

        Args:
            folderpaths: image file names (in <folderpath>/Images)
            folderpath: dataset folder
        """
        # Add classes
        self.add_class("bowl", 1, "gland")

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
            return info["bowl"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """
        Load the gland masks of one tile (<folder>/Annotation/<image file name>, a file pattern).

        Args:
            image_id: internal image id

        Returns:
            Tuple of (H x W x N mask in {0, 1}, int32 class ids, all 1)
        """
        info = self.image_info[image_id]
        image_path = info['path']
        mask_paths = os.path.join(self.folder_path, 'Annotation', image_path)
        mask = skimage.io.imread_collection(mask_paths).concatenate()
        mask = np.rollaxis(mask,0,3)
        mask = np.clip(mask,0,1)
        class_ids = np.array([1]* mask.shape[2])
        return mask, class_ids.astype(np.int32)
