"""
Colon Crypt Segmentation - Watershed Segmentation Exploration (Not Part of the Pipeline)

This script runs OpenCV's marker-based watershed on one crypt image and shows every step: inverted Otsu
threshold, opening (noise removal), sure background (dilation), sure foreground (distance transform
> 6% of its maximum), unknown region, markers, watershed. It was an early alternative to Mask R-CNN.

Usage:
    python src/preprocess/segmentation.py --image <crypt image.png>

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse

import cv2
import numpy as np
from matplotlib import pyplot as plt


# --- Default Settings ---
FG_FRACTION = 0.06     # sure foreground: distance transform above this fraction of its maximum
MARK_LABEL = 25        # watershed label painted green in the last figure


def show_image(im1, im2):
    """Show two images side by side."""
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    ax1.imshow(im1)
    ax2.imshow(im2)
    plt.show()


def watershed_steps(img):
    """
    Marker-based watershed of one BGR image, showing every step.

    Args:
        img: BGR image (modified: the region MARK_LABEL is painted green)

    Returns:
        Watershed markers (-1 on boundaries)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    show_image(thresh, opening)
    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    show_image(thresh, sure_bg)
    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 3)
    ret, sure_fg = cv2.threshold(dist_transform, FG_FRACTION * dist_transform.max(), 255, 0)
    show_image(thresh, sure_fg)
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    show_image(thresh, sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    show_image(img, unknown)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1
    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(img, markers)
    img[markers == MARK_LABEL] = [0, 255, 0]
    show_image(img, markers)
    return markers


def main():
    parser = argparse.ArgumentParser(
        description="Marker-based watershed of one crypt image, showing every step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--image", required=True, help="crypt image")
    args = parser.parse_args()

    watershed_steps(cv2.imread(args.image))


if __name__ == "__main__":
    main()
