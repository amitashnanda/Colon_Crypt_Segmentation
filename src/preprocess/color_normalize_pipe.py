"""
Colon Crypt Segmentation - Preprocessing Step 1: Reinhard Colour Normalisation

This script normalises the colour of every image under --src to a reference image with the Reinhard
method of HistomicsTK (mean and standard deviation matched in Lab space). It mirrors the folder tree of
--src under --dest and writes every image as .png (raw tiles are .JPG).

The next step is the Hough alignment (src/preprocess/hough_transform.py). HistomicsTK does not install
into the Python 3.7 TF environment, so run this step in environment_histomicstk.yml.

Usage (HistomicsTK environment, from the repository root):
    python src/preprocess/color_normalize_pipe.py --ref dataset/color_reference.png \\
        --src dataset/Whole_Slides_Segments/ --dest dataset/Color_Normalized/

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os

import cv2
import histomicstk as htk


def reinhard_normalization_recursive(src_root, dest_root, ref_img_path):
    """
    Applies Reinhard color normalization to all images within a source directory
    and its subdirectories, saving them to a destination directory with a mirrored
    structure.

    Args:
        src_root: root folder of the source images (.jpg, .jpeg, .png, .tif, .tiff)
        dest_root: root folder of the normalised .png images
        ref_img_path: reference image of the normalisation
    """
    # --- Step 1: Prepare the reference image statistics (do this only once) ---
    print("Reading reference image...")
    ref_img = cv2.imread(ref_img_path)
    if ref_img is None:
        print(f"Error: Could not read reference image at {ref_img_path}")
        return
    mean_ref, std_ref = htk.preprocessing.color_conversion.lab_mean_std(ref_img)
    print("Reference statistics calculated.")

    # Define a tuple of valid image extensions (case-insensitive)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')

    # --- Step 2: Walk through the source directory tree ---
    for dirpath, dirnames, filenames in os.walk(src_root):
        # Filter for image files
        image_files = [f for f in filenames if f.lower().endswith(valid_extensions)]

        if not image_files:
            continue

        # --- Step 3: Create the corresponding destination directory ---
        # Get the relative path from the source root
        relative_path = os.path.relpath(dirpath, src_root)
        # Create the full destination path
        dest_dir = os.path.join(dest_root, relative_path)
        # Create the directory if it doesn't exist
        os.makedirs(dest_dir, exist_ok=True)
        print(f"\nProcessing directory: {dirpath}")

        # --- Step 4: Process each image in the current directory ---
        for filename in image_files:
            input_path = os.path.join(dirpath, filename)

            # Create the new filename with a .png extension
            base_filename = os.path.splitext(filename)[0]
            output_filename = f"{base_filename}.png"
            output_path = os.path.join(dest_dir, output_filename)

            print(f"  -> Normalizing {filename}...")

            # Read the input image
            img_input = cv2.imread(input_path)
            if img_input is None:
                print(f"    - Warning: Could not read {input_path}. Skipping.")
                continue

            # Apply Reinhard normalization
            im_nmzd = htk.preprocessing.color_normalization.reinhard(img_input, mean_ref, std_ref)

            # Save the normalized image
            cv2.imwrite(output_path, im_nmzd)

    print("\nNormalization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Reinhard colour normalisation (HistomicsTK) of every image under --src, mirrored under --dest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Reference and input / output
    parser.add_argument("--ref", help="Path to the reference image. Exp: dataset/color_reference.png", type=str, required=True)
    parser.add_argument("--dest", help="Root folder for the output images. Exp: dataset/Color_Normalized/", type=str, required=True)
    parser.add_argument("--src", help="Root folder of the source images. Exp: dataset/Raw_images/", type=str, required=True)
    args = parser.parse_args()

    reinhard_normalization_recursive(args.src, args.dest, args.ref)


if __name__ == '__main__':
    main()


# Example usage (HistomicsTK environment, from the repository root):
#
#   python src/preprocess/color_normalize_pipe.py --ref dataset/color_reference.png \
#       --src dataset/Whole_Slides_Segments/ --dest dataset/Color_Normalized/
#   -> dataset/Color_Normalized/<protocol>/<slide>/<tile>.png, then src/preprocess/hough_transform.py
