"""
Colon Crypt Segmentation - Image-Technique Exploration (Not Part of the Pipeline)

This script collects three early explorations, each shown with matplotlib:
  1. Canny edges (scikit-image, sigma 7) of a grey image, closed with a binary closing,
  2. an ellipse fitted to the first contour of an annotation mask (the idea behind the U-shape step),
  3. a Reinhard colour normalisation implemented directly in OpenCV's Lab space (the pipeline uses HistomicsTK,
     src/preprocess/color_normalize_pipe.py).
Each part runs only if its input is given.

Usage (from the repository root):
    python src/preprocess/cv_technique.py --edges_image dataset/1.png --mask <annotation mask.png> \\
        --image <tile.png> --ref dataset/color_reference.png

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import feature, io, morphology


def show_canny_edges(image_path):
    """
    Show a grey image and its Canny edges (scikit-image, sigma 7, thresholds 0.05 / 0.15, binary closing).

    Args:
        image_path: image file (read as grey)
    """
    im = io.imread(image_path, 1)
    # Increase the smoothing for better results
    edges2 = feature.canny(im, sigma=7, low_threshold=0.05, high_threshold=.15)
    edges2 = morphology.binary_closing(edges2)

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    ax1.imshow(im, cmap=plt.cm.gray)
    ax1.axis('off')
    ax1.set_title('image', fontsize=20)
    ax2.imshow(edges2, cmap=plt.cm.gray)
    ax2.axis('off')
    ax2.set_title(r'Canny filter, $\sigma=7$', fontsize=20)
    fig.tight_layout()
    plt.show()


def show_mask_ellipse(mask_path):
    """
    Show an annotation mask with the ellipse fitted to its first contour.

    Args:
        mask_path: mask file (read as grey)
    """
    img = cv2.imread(mask_path, 0)
    ret, thresh = cv2.threshold(img, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, 1, 2)[-2:]   # works with OpenCV 3 and 4
    cnt = contours[0]
    ellipse = cv2.fitEllipse(cnt)
    plt.imshow(cv2.ellipse(img, ellipse, (255, 255, 0), 20))
    plt.show()


def reinhard_mat(img, ref_path):
    """
    Reinhard colour normalisation in OpenCV's Lab space: per channel, (x - mean) / std of the image, times
    the std of the reference, plus its mean.

    Args:
        img: BGR uint8 image
        ref_path: reference image

    Returns:
        Normalised BGR image
    """
    refimg = cv2.imread(ref_path)
    refimg = refimg.astype('uint8')
    im_lab = cv2.cvtColor(refimg, cv2.COLOR_BGR2LAB)
    target_mu = np.zeros(3)
    target_sigma = np.zeros(3)
    src_mu = np.zeros(3)
    src_sigma = np.zeros(3)

    for i in range(3):
        target_mu[i] = im_lab[:, :, i].mean()
        target_sigma[i] = (im_lab[:, :, i] - target_mu[i]).std()

    im_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    for i in range(3):
        src_mu[i] = im_lab[:, :, i].mean()
        src_sigma[i] = (im_lab[:, :, i] - src_mu[i]).std()

    for i in range(3):
        im_lab[:, :, i] = (im_lab[:, :, i] - src_mu[i]) / src_sigma[i] * target_sigma[i] + target_mu[i]

    im_norm = cv2.cvtColor(im_lab, cv2.COLOR_LAB2BGR)
    return im_norm


def show_reinhard(image_path, ref_path):
    """
    Show an image before and after reinhard_mat.

    Args:
        image_path: BGR image to normalise
        ref_path: reference image
    """
    img = cv2.imread(image_path).astype('uint8')
    normalized_img = reinhard_mat(img, ref_path)
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(8, 3))
    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax1.set_title('image')
    ax2.imshow(cv2.cvtColor(normalized_img, cv2.COLOR_BGR2RGB))
    ax2.set_title('Reinhard (Lab)')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Explorations: Canny edges, ellipse fit on a mask, Reinhard normalisation in Lab space.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--edges_image", default=None, help="image for the Canny edges")
    parser.add_argument("--mask", default=None, help="annotation mask for the ellipse fit")
    parser.add_argument("--image", default=None, help="image for the Reinhard normalisation (needs --ref)")
    parser.add_argument("--ref", default=None, help="reference image of the Reinhard normalisation")
    args = parser.parse_args()

    if args.edges_image:
        show_canny_edges(args.edges_image)
    if args.mask:
        show_mask_ellipse(args.mask)
    if args.image and args.ref:
        show_reinhard(args.image, args.ref)


if __name__ == "__main__":
    main()


# Example usage: see the header; each part runs only when its input is given.
