"""
Colon Crypt Segmentation - Resize the Images of a Folder

This script resizes every image of a folder (.png, .jpg, .jpeg, .bmp, .gif) to a fixed size with PIL
(Lanczos, the former Image.ANTIALIAS) and saves it under the same name in the destination folder. It was
used to bring the mouse tiles to 2048 x 1536.

Usage:
    python src/preprocess/resizeimg.py --src <image folder> --dest <output folder> --width 2048 --height 1536

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import os

from PIL import Image


# --- Default Settings ---
SIZE = (2048, 1536)   # (width, height)
EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')


def resize_images_in_folder(source_folder, destination_folder, size=SIZE):
    """
    Resize every image of a folder and save it under the same name.

    Args:
        source_folder: folder with the images
        destination_folder: output folder (created if needed)
        size: (width, height)
    """
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for filename in os.listdir(source_folder):
        print(filename)
        if filename.endswith(EXTENSIONS):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            with Image.open(source_path) as img:
                img = img.resize(size, Image.LANCZOS)
                img.save(destination_path)
                print(f"Resized {filename} and saved to {destination_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Resize every image of a folder to a fixed size (PIL, Lanczos).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True, help="folder with the images")
    parser.add_argument("--dest", required=True, help="output folder")
    parser.add_argument("--width", type=int, default=SIZE[0], help="width (default: %(default)s)")
    parser.add_argument("--height", type=int, default=SIZE[1], help="height (default: %(default)s)")
    args = parser.parse_args()

    resize_images_in_folder(args.src, args.dest, size=(args.width, args.height))


if __name__ == "__main__":
    main()
