"""
Colon Crypt Segmentation - Read a CRCHistoPhenotypes .mat File (Exploration)

This script loads one .mat annotation file of the CRCHistoPhenotypes dataset (for example
Detection/img10/img10_detection.mat, nucleus centres) and prints its variables with their shapes.
It was used to look at that dataset; it is not part of the pipeline.

Usage:
    python src/preprocess/mat_read.py --mat CRCHistoPhenotypes_2016_04_28/Detection/img10/img10_detection.mat

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse

import scipy.io


def main():
    parser = argparse.ArgumentParser(
        description="Load a CRCHistoPhenotypes .mat file and print its variables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mat", required=True, help=".mat file")
    args = parser.parse_args()

    mat = scipy.io.loadmat(args.mat)
    for key, value in mat.items():
        if not key.startswith("__"):
            print(f"{key}: {getattr(value, 'shape', type(value).__name__)}")


if __name__ == "__main__":
    main()
