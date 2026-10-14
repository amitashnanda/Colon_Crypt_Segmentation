"""
Colon Crypt Segmentation - Convert Colour Annotation Masks to Single-Channel Masks

This script turns every colour (multi-channel) mask of a folder into a single-channel mask, in place:
each pixel becomes the maximum over its channels. Single-channel masks are left untouched. It was used to
clean the gland annotations (Annotation_gland/) before training.

Usage:
    python dataset_preparation/to_grayscale.py --folder dataset/Normalized_Images/test/Annotation_gland

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import os

import cv2


def to_single_channel(folder):
    """
    Replace every multi-channel image of a folder by its channel-wise maximum (in place).

    Args:
        folder: folder with the masks
    """
    for fil in glob.glob(os.path.join(folder, "*")):
        print(fil)
        image = cv2.imread(fil, cv2.IMREAD_UNCHANGED)
        if image is None or len(image.shape) == 2:
            continue
        image = image.max(axis=2)   # maximum over the channels
        cv2.imwrite(fil, image)     # write to location with same name


def main():
    parser = argparse.ArgumentParser(
        description="Convert colour annotation masks to single-channel masks, in place.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--folder", required=True, help="folder with the masks, e.g. .../Annotation_gland")
    args = parser.parse_args()

    to_single_channel(args.folder)


if __name__ == "__main__":
    main()
