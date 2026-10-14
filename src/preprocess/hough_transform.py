"""
Colon Crypt Segmentation - Preprocessing Step 2: Hough Alignment of the Tiles

This script straightens every colour-normalised tile using the tissue orientation found with Hough
lines: the black border is cropped, edges are found with Canny (thresholds from the median grey value)
on a bilateral-filtered grey image, line segments with the probabilistic Hough transform, and the tile
is rotated by minus the mean line angle (outliers removed) with imgaug Rotate(fit_output=True).

It mirrors <src>/**/*.png under <dest>. The aligned tiles are the input of the inference pipeline
(src/inference/run_inference_pipeline.py), in place of dataset/Whole_Slides_Segments_Processed/CDX2_Data.
With --show it also displays the grey image, the filtered image, the edges and the lines of every tile.

Usage (TF environment, from the repository root):
    python src/preprocess/hough_transform.py --src dataset/Color_Normalized/ --dest dataset/Hough_Aligned/

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import math
import os

import imgaug.augmenters as iaa
import skimage.io
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


# --- Default Settings ---
CANNY_SIGMA = 0.4           # Canny thresholds at (1 -+ sigma) x median grey value
MAX_LINES = 100             # Hough segments used for the angle
OUTLIER_M = 0.5             # keep angles within m standard deviations of the mean (1 if none is left)
ANNOTATION_FOLDERS = ['/Annotation_gland/', '/Annotation/']   # mask folders rotated with the training images


def show_image(img, cmap=None):
    """Display one image (debugging, --show)."""
    plt.imshow(img, cmap=cmap)
    plt.show()


def reject_outliers(data, m=2):
    """
    Keep the values within m standard deviations of the mean.

    Args:
        data: 1-D array
        m: number of standard deviations

    Returns:
        The kept values
    """
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def get_hough_angle(img, show=False):
    """
    Tissue orientation of a tile from probabilistic Hough lines.

    Args:
        img: BGR tile with the black border cropped
        show: display the intermediate images and the lines (draws the lines into `img`)

    Returns:
        Rotation angle in radians (minus the mean line angle; 0 if no line is found)
    """
    imgray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if show:
        show_image(imgray, cmap='gray')
    imgray = cv.bilateralFilter(imgray, 10, 10, 10)
    if show:
        show_image(imgray, cmap='gray')
    v = np.median(imgray)
    sigma = CANNY_SIGMA
    lower_thresh = int(max(0, (1.0 - sigma) * v))
    upper_thresh = int(min(255, (1.0 + sigma) * v))

    # edges, then line segments (relaxed parameters if fewer than 5 are found)
    edges = cv.Canny(imgray, lower_thresh, upper_thresh)
    lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 60, 10)
    if lines is None or len(lines) < 5:
        lines = cv.HoughLinesP(edges, 1, math.pi / 128, 40, None, 80, 20)
        if lines is None or len(lines) < 5:
            lines = cv.HoughLinesP(edges, 1, math.pi / 128, 20, None, 80, 20)
    if show:
        show_image(edges)
    if lines is None:
        return 0

    # angle of every segment (the small offset avoids division by zero for vertical segments)
    lines_theta = np.array([np.arctan((line[0][3] - line[0][1]) / (line[0][2] - line[0][0]+.00001)) for line in lines[:MAX_LINES]])

    filter_lines_theta = reject_outliers(lines_theta, OUTLIER_M)
    if len(filter_lines_theta) == 0:
        filter_lines_theta = reject_outliers(lines_theta, 1)

    if show:
        for l in lines[:MAX_LINES]:
            for line in l:
                pt1 = (line[0], line[1])
                pt2 = (line[2], line[3])
                cv.line(img, pt1, pt2, (0, 0, 255), 3)
        show_image(img)
    res = -np.mean(filter_lines_theta)
    if np.isnan(res):
        res = 0
    return res


def cut_black_area(img):
    """
    Bounding box of the pixels that are non-zero in all channels (the tile without its black border).

    Args:
        img: H x W x C image

    Returns:
        Tuple (x0, x1, y0, y1): rows x0:x1, columns y0:y1
    """
    mask = img > 0
    mask = mask.all(2)
    coords = np.argwhere(mask)
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1
    return x0, x1, y0, y1


def rotate_image(img, theta):
    """
    Rotate an image by theta (radians) with imgaug, enlarging the canvas to keep the whole image.

    Args:
        img: image
        theta: angle in radians

    Returns:
        Rotated image
    """
    augmentation = iaa.Sequential([
        iaa.Rotate((theta * 180 / math.pi), fit_output=True)
    ], random_order=False)
    img = augmentation.augment_image(img)
    return img


def save_hough_transform(image_path, output_path, show=False):
    """
    Align one tile and save it.

    The angle is computed on the tile without its black border (read with OpenCV); the rotation is applied
    to the full tile as read by scikit-image.

    Args:
        image_path: colour-normalised tile
        output_path: aligned tile to write
        show: display the intermediate images
    """
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    x0, x1, y0, y1 = cut_black_area(img)
    img = img[x0:x1, y0:y1]
    theta = get_hough_angle(img, show=show)
    img = skimage.io.imread(image_path)
    img = rotate_image(img, theta)
    skimage.io.imsave(output_path, img)


def save_hough_transform_annotation(image_path, dst_folder, label_folder):
    """
    Align one training image and its annotation masks by the same angle (training data preparation).

    Writes <dst_folder>/Images/<image>.png and, for every mask <label_folder>/<Annotation folder>/<image>_*.png,
    <dst_folder>/<Annotation folder>/<mask>.png.

    Args:
        image_path: training image
        dst_folder: output root with Images/, Annotation/ and Annotation_gland/
        label_folder: root with Annotation/ and Annotation_gland/
    """
    img = cv.imread(image_path, cv.IMREAD_UNCHANGED)
    x0, x1, y0, y1 = cut_black_area(img)
    img = img[x0:x1, y0:y1]
    theta = get_hough_angle(img)
    img = skimage.io.imread(image_path)
    img = rotate_image(img, theta)
    skimage.io.imsave(dst_folder + "/Images/" + os.path.splitext(os.path.basename(image_path))[0]+ '.png', img)

    for i, mask_folder in enumerate(ANNOTATION_FOLDERS):
        mask_names = glob.glob(label_folder + mask_folder + os.path.splitext(os.path.basename(image_path))[0] + '_*.png')
        for mask_name in mask_names:
            mask = skimage.io.imread(mask_name)
            mask = rotate_image(mask, theta)
            skimage.io.imsave(dst_folder + mask_folder + os.path.splitext(os.path.basename(mask_name))[0]+ '.png', mask)


def main():
    parser = argparse.ArgumentParser(
        description="Align every colour-normalised tile with the Hough-line orientation, mirrored under --dest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input / output
    parser.add_argument("--dest", help="Root folder path of the dest images Exp: images/", type=str,
                        required=True)
    parser.add_argument("--src",
                        help="Root folder path of the source images (will only process png files) Exp: if you have images in images/data1/1.png  images/data2/3.png you should pass: images/",
                        type=str, required=True)

    # Options
    parser.add_argument("--show", action="store_true",
                        help="display the intermediate images of every tile (needs a display)")
    args = parser.parse_args()

    src_folder = args.src
    dst_folder = args.dest

    search_pattern = os.path.join(src_folder, '**', '*.png')
    file_paths = glob.glob(search_pattern, recursive=True)
    print(f"Found {len(file_paths)} .png files to process.")
    for file_path in file_paths:
        print(f"Processing: {file_path}")

        # mirror the folder structure: <src>/<rel>/<tile>.png -> <dest>/<rel>/<tile>.png
        relative_path = os.path.relpath(file_path, src_folder)
        relative_path_png = os.path.splitext(relative_path)[0] + '.png'
        output_path = os.path.join(dst_folder, relative_path_png)
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        save_hough_transform(file_path, output_path, show=args.show)


if __name__ == '__main__':
    main()


# Example usage (TF environment, from the repository root):
#
#   python src/preprocess/hough_transform.py --src dataset/Color_Normalized/ --dest dataset/Hough_Aligned/
#   -> dataset/Hough_Aligned/<protocol>/<slide>/<tile>.png, the input of src/inference/run_inference_pipeline.py --src
