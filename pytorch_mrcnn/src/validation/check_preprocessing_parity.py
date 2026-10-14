"""
Colon Crypt Segmentation (PyTorch) - Validation: Preprocessing Parity

This script checks that the PyTorch preprocessing gives exactly the TF input image for every tile.
    TF (src/inference/my_inference.py):  skimage.io.imread -> imgaug PadToAspectRatio(1.5, "center")
                                         -> Resize(768x1024)
    PyTorch (tfparity_maskrcnn.load_and_preprocess): the same read, then an exact cv2 replica of the
                                         imgaug ops
Each mode hashes the preprocessed image of every tile into a JSON file; --compare compares two of them.

Usage (from pytorch_mrcnn/, with CDX=dataset/Whole_Slides_Segments_Processed/CDX2_Data, C=validation/compare):
    # step 1, cdx2_tf1_ref (has imgaug)
    python src/validation/check_preprocessing_parity.py --mode imgaug  --src $CDX --out $C/preprocess_tf_imgaug.json
    # step 2, torch_env
    python src/validation/check_preprocessing_parity.py --mode replica --src $CDX --out $C/preprocess_torch_replica.json
    # step 3
    python src/validation/check_preprocessing_parity.py --compare $C/preprocess_tf_imgaug.json $C/preprocess_torch_replica.json

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import glob
import hashlib
import json
import os
import sys


def tiles(src, skip=("Mouse-DSS",)):
    """
    List the tiles under a root, leaving out the slide folders in `skip`.

    Args:
        src: tile root
        skip: slide folder names to leave out

    Returns:
        Sorted list of tile paths
    """
    paths = sorted(glob.glob(os.path.join(src, "**", "*.png"), recursive=True))
    return [p for p in paths if not any(f"{os.sep}{s}{os.sep}" in p for s in skip)]


def compare(json_a, json_b):
    """
    Print how many tiles have the same preprocessed-image hash in two JSON files.

    Args:
        json_a, json_b: hash files written by --mode imgaug / --mode replica
    """
    a, b = (json.load(open(f)) for f in (json_a, json_b))
    common = sorted(set(a) & set(b))
    same = sum(a[k] == b[k] for k in common)
    print(f"tiles in both: {len(common)}; identical preprocessed images: {same}; different: {len(common) - same}")
    for k in [k for k in common if a[k] != b[k]][:10]:
        print("   differs:", k)


def main():
    parser = argparse.ArgumentParser(
        description="Hash the TF (imgaug) or PyTorch (replica) preprocessed image of every tile, or compare two hash files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Hashing
    parser.add_argument("--mode", choices=["imgaug", "replica"], help="imgaug (TF env) or replica (torch_env)")
    parser.add_argument("--src", help="tile root")
    parser.add_argument("--out", help="JSON file to write")

    # Comparison
    parser.add_argument("--compare", nargs=2, help="two JSON files to compare")
    args = parser.parse_args()

    if args.compare:
        compare(*args.compare)
        return

    # --- preprocessing function of the chosen side (imported here: each side runs in its own environment) ---
    import numpy as np
    import skimage.io
    from skimage import img_as_ubyte
    if args.mode == "imgaug":
        import imgaug.augmenters as iaa
        aug = iaa.Sequential([iaa.PadToAspectRatio(1.5, position="center"),
                              iaa.Resize({"height": 768, "width": 1024})], random_order=False)

        def prep(p):
            img = skimage.io.imread(p)
            if len(img.shape) < 3:
                img = img_as_ubyte(img)
                img = np.expand_dims(img, 2)
                img = img[:, :, [0, 0, 0]]
            img = img[:, :, :3]
            return aug.augment_image(img)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))
        from tfparity_maskrcnn import load_and_preprocess as prep

    # --- hash every preprocessed tile ---
    out = {}
    for p in tiles(args.src):
        arr = np.ascontiguousarray(prep(p))
        out[os.path.relpath(p, args.src)] = hashlib.md5(arr.tobytes() + str(arr.shape).encode()).hexdigest()
    json.dump(out, open(args.out, "w"), indent=0)
    print(f"{args.mode}: hashed {len(out)} tiles -> {args.out}")


if __name__ == "__main__":
    main()


# Example usage: see the three steps in the header (run_checks.sh runs them all).
