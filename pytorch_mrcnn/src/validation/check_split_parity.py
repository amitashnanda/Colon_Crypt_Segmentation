"""
Colon Crypt Segmentation (PyTorch) - Validation: U-Shape Split Parity

This script checks that the U-shape split gives TF's categories in this environment. It classifies
every TF crypt mask (inf/*/predicted_mask) with the functions of src/inference/u_shape_split.py and
compares the result with the folder TF put the mask in (us_split/accepted, fragmented, rejected).

Differences come from library versions: src/inference/u_shape_split.py uses the scikit-image 0.15
remove_small_objects; --no_compat swaps in the installed scikit-image version to show the raw difference.

Usage (torch_env, from pytorch_mrcnn/):
    python src/validation/check_split_parity.py --root ../inference

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse
import collections
import os
import sys

# U-shape classifier of pytorch_mrcnn/src/inference/u_shape_split.py
HERE = os.path.dirname(os.path.abspath(__file__))
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(HERE, "..", "inference"))
import u_shape_split as us  # noqa: E402  (pytorch_mrcnn/src/inference/u_shape_split.py)


# --- Default Settings ---
CAT = {"good_u": "accepted", "fragmented_u": "fragmented", "o_shape": "rejected", "rejected": "rejected"}   # class -> folder


def main():
    parser = argparse.ArgumentParser(
        description="Classify every TF crypt mask with u_shape_split.py and compare with TF's folders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root", required=True, help="TF results root with inf/ and us_split/")
    parser.add_argument("--no_compat", action="store_true",
                        help="use the installed scikit-image remove_small_objects instead of the 0.15 behaviour")
    args = parser.parse_args()

    if args.no_compat:
        from skimage.morphology import remove_small_objects
        us.remove_small_objects = remove_small_objects

    stats = collections.Counter()
    diffs = []
    for p in sorted(os.listdir(os.path.join(args.root, "us_split"))):
        for s in sorted(os.listdir(os.path.join(args.root, "us_split", p))):
            if s == "Mouse-DSS":
                continue
            # folder TF put each mask in
            truth = {}
            for c in ("accepted", "fragmented", "rejected"):
                d = os.path.join(args.root, "us_split", p, s, c)
                for f in (os.listdir(d) if os.path.isdir(d) else []):
                    truth[f] = c
            # category given here
            mdir = os.path.join(args.root, "inf", p, s, "predicted_mask")
            for f, c in truth.items():
                label, _ = us.classify_mask(us.read_mask_binary(os.path.join(mdir, f)))
                got = CAT[label]
                stats["masks"] += 1
                stats["same"] += int(got == c)
                if got != c:
                    diffs.append((p, s, f, c, got))
    print(f"masks classified: {stats['masks']}; same category as TF: {stats['same']}; different: {len(diffs)}")
    for d in diffs[:20]:
        print("   ", d)


if __name__ == "__main__":
    main()


# Example usage (torch_env, from pytorch_mrcnn/):
#
#   python src/validation/check_split_parity.py --root ../inference               # 3852 of 3852 the same
#   python src/validation/check_split_parity.py --root ../inference --no_compat   # installed scikit-image
