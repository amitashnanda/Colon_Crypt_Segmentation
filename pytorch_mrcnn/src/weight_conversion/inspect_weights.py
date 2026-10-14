"""
Colon Crypt Segmentation (PyTorch) - Inspect Keras Weights

This script lists the tensors of a Matterport Mask R-CNN Keras .h5 weights file: layer, weight name,
shape, dtype and parameter count, with the total. weight_map.txt in this folder is the --dump listing
of notransfer.h5, used to write the conversion rules of convert_weights_to_pytorch.py.

Usage (any environment with h5py, from pytorch_mrcnn/):
    python src/weight_conversion/inspect_weights.py --weights models/notransfer.h5
    python src/weight_conversion/inspect_weights.py --weights models/notransfer.h5 --filter res2
    python src/weight_conversion/inspect_weights.py --weights models/notransfer.h5 --dump src/weight_conversion/weight_map.txt

Author: Amitash Nanda
Date: 2026-10-09
"""

import argparse

import h5py
import numpy as np


def walk_weights(h5_path, name_filter=None):
    """
    Yield every tensor of a Keras .h5 weights file.

    Args:
        h5_path: Keras weights (.h5)
        name_filter: only layers whose name contains this substring

    Yields:
        Tuples of (layer_name, weight_name, shape, dtype, n_params)
    """
    with h5py.File(h5_path, "r") as f:
        # Matterport stores weights under a top-level group. Usually "model_weights"
        # but some older Keras files put layer groups at the root. Handle both.
        root = f["model_weights"] if "model_weights" in f else f
        for layer_name in root.keys():
            if name_filter and name_filter not in layer_name:
                continue
            group = root[layer_name]

            # A layer group may be empty (no trainable weights) or nested;
            # h5py's visititems doesn't yield, so recurse manually.
            def recurse(g, prefix=""):
                for k, v in g.items():
                    p = f"{prefix}/{k}" if prefix else k
                    if isinstance(v, h5py.Dataset):
                        yield (
                            layer_name,
                            p,
                            tuple(v.shape),
                            str(v.dtype),
                            int(np.prod(v.shape)),
                        )
                    else:
                        yield from recurse(v, p)

            yield from recurse(group)


def main():
    parser = argparse.ArgumentParser(
        description="List the tensors of a Matterport Mask R-CNN Keras .h5 weights file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input
    parser.add_argument("--weights", required=True, help="Keras weights, e.g. models/notransfer.h5")

    # Options
    parser.add_argument("--filter", default=None, help="Only show layers containing this substring")
    parser.add_argument("--dump", default=None, help="Write the full listing to this file")
    args = parser.parse_args()

    rows = list(walk_weights(args.weights, args.filter))

    # aligned listing with the total
    total = 0
    header = f"{'LAYER':45} {'WEIGHT':30} {'SHAPE':25} {'DTYPE':10} {'PARAMS':>12}"
    lines = [header, "-" * len(header)]
    for layer, weight, shape, dtype, n in rows:
        lines.append(f"{layer:45} {weight:30} {str(shape):25} {dtype:10} {n:>12,}")
        total += n
    lines.append("-" * len(header))
    lines.append(f"{'TOTAL':45} {'':30} {'':25} {'':10} {total:>12,}")
    lines.append(f"({len(rows)} tensors)")

    out = "\n".join(lines)
    print(out)
    if args.dump:
        with open(args.dump, "w") as f:
            f.write(out)
        print(f"\nWrote {args.dump}")


if __name__ == "__main__":
    main()


# Example usage: see the header (the --dump listing of notransfer.h5 is weight_map.txt).
